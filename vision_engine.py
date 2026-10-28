"""Step 3: Computer Vision & Spatial Shelf Logic.

YOLOv8 inference + OpenCV spatial geometry over 3 vertical brand lanes.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Class id -> brand name (matches data.yaml)
CLASS_NAMES = {0: "coca", 1: "fanta", 2: "sprite"}


class ShelfVisionEngine:
    """Runs YOLOv8 detection and assigns every bottle to a shelf zone."""

    def __init__(self, weights_path, conf=0.40, iou=0.45, device=0):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device

        # Normalized zone polygons: (x_min, x_max, y_min, y_max, expected_brand, max_capacity)
        # Capacity is 2: YOLO sees 2 front-facing bottles per lane, so a full shelf is 2/2.
        self.zones = {
            "Sprite_Zone": (0.04, 0.37, 0.02, 0.96, "sprite", 2),
            "Coca_Zone": (0.37, 0.70, 0.02, 0.96, "coca", 2),
            "Fanta_Zone": (0.70, 0.98, 0.02, 0.96, "fanta", 2),
        }

        # Interactive simulation state
        self.simulated_removed = {"Sprite_Zone": 0, "Coca_Zone": 0, "Fanta_Zone": 0}
        self.simulate_misplaced = False

        self._polygons = {}  # zone -> scaled np.int32 polygon
        self._colors = {
            "Sprite_Zone": (0, 255, 0),
            "Coca_Zone": (0, 255, 0),
            "Fanta_Zone": (0, 255, 0),
        }

    def _build_polygons(self, w, h):
        """Scale normalized zone rects to pixel polygons for this frame."""
        self._polygons = {}
        for zone, (x0, x1, y0, y1, _brand, _cap) in self.zones.items():
            pts = np.array(
                [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]],
                dtype=np.int32,
            )
            self._polygons[zone] = pts

    def _detect(self, frame):
        """Run YOLOv8, return list of dicts with boxes/conf/brand."""
        results = self.model(frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        out = []
        for box, conf, cls in zip(
            results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls
        ):
            x1, y1, x2, y2 = box.tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            brand = CLASS_NAMES.get(int(cls), "unknown")
            out.append(
                {
                    "box": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "conf": float(conf),
                    "brand": brand,
                    "class_id": int(cls),
                }
            )
        return out

    def _assign_zones(self, detections):
        """Assign each detection to exactly one zone via centroid pointPolygonTest."""
        for d in detections:
            d["zone"] = None
            for zone, poly in self._polygons.items():
                if cv2.pointPolygonTest(poly, d["center"], False) >= 0:
                    d["zone"] = zone
                    break

    def _simulate_removal(self, zone, detections):
        """Drop the last N detections in a zone to simulate a customer taking bottles.

        Detections are sorted left-to-right before dropping so removal is
        deterministic across frames instead of flickering with detection order.
        """
        if self.simulated_removed.get(zone, 0) <= 0:
            return detections
        in_zone = [d for d in detections if d["zone"] == zone]
        others = [d for d in detections if d["zone"] != zone]
        in_zone.sort(key=lambda d: d["box"][0])  # left-to-right, stable
        drop = min(self.simulated_removed[zone], len(in_zone))
        in_zone = in_zone[: len(in_zone) - drop]
        return others + in_zone

    def _inject_misplaced(self, detections):
        """When simulating, relabel the first Coca_Zone bottle as sprite so the
        MISPLACED ITEM alert is visible in the Coca-Cola lane."""
        if not self.simulate_misplaced:
            return detections
        for d in detections:
            if d["zone"] == "Coca_Zone":
                d["brand"] = "sprite"
                d["class_id"] = 2
                break
        return detections

    def _zone_math(self, detections):
        """Per-zone stock, empty space, occupancy, planogram compliance, status."""
        telemetry = {}
        
        # Use list() to avoid issues if we update the dictionary size during the loop
        for zone, (x0, x1, y0, y1, expected, capacity) in list(self.zones.items()):
            items = [d for d in detections if d["zone"] == zone]
            stock = len(items)
            
            # --- DYNAMIC CAPACITY PATCH ---
            # If a new image has more bottles than the current max capacity, expand the capacity!
            if stock > capacity:
                capacity = stock
                self.zones[zone] = (x0, x1, y0, y1, expected, capacity)
            # ------------------------------
            
            brands = {"coca": 0, "fanta": 0, "sprite": 0}
            misplaced = 0
            for d in items:
                brands[d["brand"]] = brands.get(d["brand"], 0) + 1
                if d["brand"] != expected:
                    misplaced += 1
                    
            empty = max(0, capacity - stock)
            # Prevent division by zero if capacity is somehow 0
            occupancy = round((stock / capacity) * 100, 1) if capacity > 0 else 0.0
            
            if stock == 0:
                status = "OUT OF STOCK"
            elif misplaced > 0:
                status = "MISPLACED ITEM"
            elif empty >= 1:
                status = "LOW STOCK"
            else:
                status = "HEALTHY"
                
            telemetry[zone] = {
                "expected_brand": expected,
                "max_capacity": capacity,
                "current_stock_count": stock,
                "brand_breakdown": brands,
                "empty_space_count": empty,
                "occupancy_pct": occupancy,
                "misplaced_count": misplaced,
                "status": status,
            }
        return telemetry

    @staticmethod
    def _status_color(status):
        return {
            "OUT OF STOCK": (0, 0, 255),
            "LOW STOCK": (0, 165, 255),
            "MISPLACED ITEM": (0, 165, 255),
        }.get(status, (0, 255, 0))  # HEALTHY -> green

    @staticmethod
    def _hud_status(status):
        """Compact HUD label only — telemetry keeps the full status string."""
        return {"OUT OF STOCK": "OUT", "MISPLACED ITEM": "MISPLACED"}.get(status, status)

    def _draw_zones(self, frame, telemetry):
        """Semi-transparent polygon fill + crisp border, colored by status."""
        overlay = frame.copy()
        for zone, poly in self._polygons.items():
            color = self._status_color(telemetry[zone]["status"])
            self._colors[zone] = color
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2)
        return cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    def _draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cx, cy = d["center"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
            # Pill near the centroid so the top HUD bar cannot cover it.
            label = f"{d['brand']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            lx, ly = cx - tw // 2, cy - 10
            cv2.rectangle(frame, (lx, ly), (lx + tw, ly + th), (0, 255, 255), -1)
            cv2.putText(frame, label, (lx, ly + th - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def _draw_hud(self, frame, telemetry):
        h, w = frame.shape[:2]
        bar_h = int(h * 0.07)
        cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
        for idx, (zone, tz) in enumerate(telemetry.items()):
            name = zone.replace("_Zone", "")
            text = f"{name}: {tz['current_stock_count']}/{tz['max_capacity']} (E:{tz['empty_space_count']}) {self._hud_status(tz['status'])}"
            x = int(idx * (w / 3)) + 8
            color = self._status_color(tz["status"])
            cv2.putText(frame, text, (x, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def process_frame(self, frame):
        """Main entry: detect -> assign -> filter -> simulate -> math -> draw -> telemetry."""
        self._build_polygons(frame.shape[1], frame.shape[0])
        detections = self._detect(frame)
        self._assign_zones(detections)
        # Drop detections whose centroid falls outside every zone polygon.
        detections = [d for d in detections if d["zone"] is not None]
        detections = self._inject_misplaced(detections)
        for zone in self.zones:
            detections = self._simulate_removal(zone, detections)
        telemetry = self._zone_math(detections)
        annotated = frame.copy()
        annotated = self._draw_zones(annotated, telemetry)
        self._draw_detections(annotated, detections)
        self._draw_hud(annotated, telemetry)
        return annotated, telemetry