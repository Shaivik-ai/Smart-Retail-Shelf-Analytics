"""Step 4: End-to-End CV + Data Science verification.

Interactive mode: python test_step3.py  (keys: 1/2/3 remove, m misplaced, r reset, s snapshot, q quit)
Headless mode:    python test_step4.py --headless-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from analytics_engine import InventoryAnalyticsEngine
from vision_engine import ShelfVisionEngine

ZONE_KEYS = ["Sprite_Zone", "Coca_Zone", "Fanta_Zone"]


def parse_args():
    p = argparse.ArgumentParser(description="Step 4: CV + data science engine")
    p.add_argument("--video", default="Dataset/Bevrage_Shelf.mp4")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--headless-test", action="store_true")
    p.add_argument("--seconds", type=int, default=12)
    p.add_argument("--save-output", default=None)
    return p.parse_args()


def run_interactive(args):
    analytics = InventoryAnalyticsEngine(log_interval_sec=5.0, reset_db=True)
    vision = ShelfVisionEngine(args.weights, device=args.device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save_output:
        writer = cv2.VideoWriter(args.save_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_frame = None
    print("Controls: 1/2/3 remove | r reset | m misplaced | s snapshot+JSON | q quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        last_frame = frame.copy()
        annotated, telemetry = vision.process_frame(frame)
        analytics.maybe_log(telemetry)

        if writer:
            writer.write(annotated)
        cv2.imshow("Step 4 - Shelf Vision", annotated)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            vision.simulated_removed[ZONE_KEYS[key - ord("1")]] += 1
            _, telemetry = vision.process_frame(last_frame)
            analytics.maybe_log(telemetry, force=True)
        elif key == ord("r"):
            vision.simulated_removed = {z: 0 for z in ZONE_KEYS}
            vision.simulate_misplaced = False
            _, telemetry = vision.process_frame(last_frame)
            analytics.maybe_log(telemetry, force=True)
        elif key == ord("m"):
            vision.simulate_misplaced = not vision.simulate_misplaced
            _, telemetry = vision.process_frame(last_frame)
            analytics.maybe_log(telemetry, force=True)
        elif key == ord("s"):
            cv2.imwrite("step4_verification.jpg", annotated)
            analytics.export_json("inventory_snapshot.json")
            print(json.dumps(analytics.get_zone_analytics(), indent=2))

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    analytics.close()


def run_headless(args):
    """Automated verification: 150 frames, progressive removals, print + save."""
    analytics = InventoryAnalyticsEngine(log_interval_sec=5.0, reset_db=True)
    vision = ShelfVisionEngine(args.weights, device=args.device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    last_frame = None
    last_annotated = None
    frame_no = 0

    while frame_no < 150:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        last_frame = frame.copy()
        frame_no += 1

        # Progressive depletion: remove bottles at frames 30, 60, 90, 120.
        if frame_no == 30:
            vision.simulated_removed["Coca_Zone"] = 1
        elif frame_no == 60:
            vision.simulated_removed["Sprite_Zone"] = 1
        elif frame_no == 90:
            vision.simulated_removed["Coca_Zone"] = 2
        elif frame_no == 120:
            vision.simulated_removed["Fanta_Zone"] = 1

        annotated, telemetry = vision.process_frame(frame)
        last_annotated = annotated
        analytics.maybe_log(telemetry)

    cap.release()

    # Force-log the final depleted state so the analytics are non-zero.
    last_annotated, telemetry = vision.process_frame(last_frame)
    analytics.maybe_log(telemetry, force=True)
    cv2.imwrite("step4_verification.jpg", last_annotated)

    print("\n================ STEP 4 VERIFICATION ================")
    print(f"frames processed: {frame_no}")
    print("\n--- Zone analytics (Pandas summary) ---")
    import pandas as pd
    df = pd.DataFrame(analytics.get_zone_analytics())
    print(df.to_string(index=False))
    print("\n--- Out-of-stock warnings ---")
    print(json.dumps(analytics.out_of_stock_warnings(), indent=2))
    print("\n--- ETSO (est. time to stockout) ---")
    for z in ZONE_KEYS:
        print(f"  {z}: {analytics.est_time_to_stockout(z)} min")

    out = analytics.export_json("inventory_snapshot.json")
    print(f"\nexported: {out}")
    analytics.close()
    return 0


def main():
    args = parse_args()
    if args.headless_test:
        return run_headless(args)
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())