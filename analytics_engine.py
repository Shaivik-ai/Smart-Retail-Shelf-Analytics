"""Step 4: Data Science Engine — SQLite time-series logging + forecasting.

Consumes the telemetry dict from ShelfVisionEngine.process_frame() directly.
"""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "inventory.db"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS inventory_logs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT,
    epoch              REAL,
    shelf_zone         TEXT,
    expected_brand     TEXT,
    current_stock_count INTEGER,
    empty_space_count  INTEGER,
    max_capacity       INTEGER,
    occupancy_pct      REAL,
    misplaced_count    INTEGER,
    status             TEXT
)
"""

CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_logs_epoch ON inventory_logs(epoch);"


class InventoryAnalyticsEngine:
    """Logs per-zone telemetry to SQLite and forecasts out-of-stock risk."""

    def __init__(self, db_path=DB_PATH, log_interval_sec=5.0, reset_db=False):
        self.db_path = Path(db_path)
        self.log_interval_sec = log_interval_sec
        self.last_logged_time = 0.0
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(CREATE_SQL)
        self._conn.execute(CREATE_INDEX_SQL)
        self._conn.commit()
        if reset_db:
            self.reset_db()

    def reset_db(self):
        """Wipe all logged rows for a clean demo run."""
        self._conn.execute("DELETE FROM inventory_logs")
        self._conn.commit()
        self.last_logged_time = 0.0

    def maybe_log(self, telemetry, force=False):
        """Insert 1 row per zone when the interval elapsed or forced.

        Returns the number of rows written (3 per snapshot, 0 when throttled).
        """
        now = time.time()
        if not force and (now - self.last_logged_time) < self.log_interval_sec:
            return 0
        self.last_logged_time = now
        ts = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        rows = []
        for zone, tz in telemetry.items():
            rows.append(
                (
                    ts,
                    now,
                    zone,
                    tz["expected_brand"],
                    int(tz["current_stock_count"]),
                    int(tz["empty_space_count"]),
                    int(tz["max_capacity"]),
                    float(tz["occupancy_pct"]),
                    int(tz["misplaced_count"]),
                    tz["status"],
                )
            )
        self._conn.executemany(
            "INSERT INTO inventory_logs "
            "(timestamp, epoch, shelf_zone, expected_brand, current_stock_count, "
            "empty_space_count, max_capacity, occupancy_pct, misplaced_count, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # ------------------------------------------------------------------ queries

    def _history_df(self, zone=None, window_sec=180.0, limit=None):
        """Recent log rows for one zone (or all zones) as a DataFrame."""
        cutoff = time.time() - window_sec
        sql = "SELECT * FROM inventory_logs WHERE epoch >= ?"
        params = [cutoff]
        if zone:
            sql += " AND shelf_zone = ?"
            params.append(zone)
        sql += " ORDER BY epoch ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        df = pd.read_sql_query(sql, self._conn, params=params)
        return df

    @staticmethod
    def _depletion_rate(df):
        """Linear-regression slope of stock over elapsed minutes (units/min)."""
        if df.empty or len(df) < 2:
            return 0.0
        t = (df["epoch"] - df["epoch"].iloc[0]) / 60.0
        s = df["current_stock_count"].astype(float)
        slope = float(np.polyfit(t, s, 1)[0])
        return round(abs(slope), 2) if slope < 0 else 0.0

    def est_time_to_stockout(self, zone):
        """Minutes until a zone hits zero stock, or None if the trend is stable."""
        df = self._history_df(zone=zone)
        if df.empty:
            return None
        current = int(df["current_stock_count"].iloc[-1])
        if current == 0:
            return 0.0
        rate = self._depletion_rate(df)
        if rate > 0:
            return round(current / rate, 1)
        return None

    def get_zone_analytics(self, window_sec=180.0):
        """Per-zone depletion rate, ETSO, and out-of-stock warning flag."""
        zones = ["Sprite_Zone", "Coca_Zone", "Fanta_Zone"]
        out = []
        for zone in zones:
            df = self._history_df(zone=zone, window_sec=window_sec)
            rate = self._depletion_rate(df)
            current = int(df["current_stock_count"].iloc[-1]) if not df.empty else 0
            empty = int(df["empty_space_count"].iloc[-1]) if not df.empty else 0
            status = str(df["status"].iloc[-1]) if not df.empty else "HEALTHY"
            ets = 0.0 if current == 0 else (round(current / rate, 1) if rate > 0 else None)
            oos = (
                status == "OUT OF STOCK"
                or empty >= 1
                or (ets is not None and ets <= 2.0)
            )
            out.append(
                {
                    "shelf_zone": zone,
                    "expected_brand": df["expected_brand"].iloc[-1] if not df.empty else "",
                    "current_stock_count": current,
                    "empty_space_count": empty,
                    "max_capacity": int(df["max_capacity"].iloc[-1]) if not df.empty else 0,
                    "occupancy_pct": float(df["occupancy_pct"].iloc[-1]) if not df.empty else 0.0,
                    "misplaced_count": int(df["misplaced_count"].iloc[-1]) if not df.empty else 0,
                    "status": status,
                    "depletion_rate_per_min": rate,
                    "est_time_to_stockout_min": ets,
                    "oos_warning": oos,
                }
            )
        return out

    def snapshot(self, zone):
        """Latest row for a zone, enriched with depletion and ETSO."""
        df = self._history_df(zone=zone, window_sec=180.0, limit=1)
        if df.empty:
            return {}
        r = df.iloc[-1]
        rate = self._depletion_rate(self._history_df(zone=zone))
        current = int(r["current_stock_count"])
        ets = 0.0 if current == 0 else (round(current / rate, 1) if rate > 0 else None)
        return {
            "timestamp": r["timestamp"],
            "shelf_zone": r["shelf_zone"],
            "expected_brand": r["expected_brand"],
            "current_stock_count": current,
            "empty_space_count": int(r["empty_space_count"]),
            "max_capacity": int(r["max_capacity"]),
            "occupancy_pct": float(r["occupancy_pct"]),
            "misplaced_count": int(r["misplaced_count"]),
            "status": r["status"],
            "depletion_rate_per_min": rate,
            "est_time_to_stockout_min": ets,
        }

    def out_of_stock_warnings(self, empty_threshold=1):
        """Zones at risk: empty, non-healthy, or stockout within 2 minutes."""
        warnings = []
        for a in self.get_zone_analytics():
            if (
                a["empty_space_count"] >= empty_threshold
                or a["status"] != "HEALTHY"
                or (a["est_time_to_stockout_min"] is not None and a["est_time_to_stockout_min"] <= 2.0)
            ):
                warnings.append(
                    {
                        "shelf_zone": a["shelf_zone"],
                        "expected_brand": a["expected_brand"],
                        "current_stock_count": a["current_stock_count"],
                        "empty_space_count": a["empty_space_count"],
                        "status": a["status"],
                        "depletion_rate_per_min": a["depletion_rate_per_min"],
                        "est_time_to_stockout_min": a["est_time_to_stockout_min"],
                    }
                )
        return warnings

    def get_timeseries_history(self, limit_per_zone=20):
        """JSON-serializable per-zone timestamps + stock arrays for Chart.js."""
        series = {}
        for zone in ["Sprite_Zone", "Coca_Zone", "Fanta_Zone"]:
            df = self._history_df(zone=zone, window_sec=3600.0, limit=limit_per_zone)
            series[zone] = {
                "timestamps": df["timestamp"].tolist(),
                "stock": df["current_stock_count"].astype(int).tolist(),
            }
        return series

    def export_json(self, path="inventory_snapshot.json"):
        """Full JSON summary: generated_at, zones, warnings, timeseries."""
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "log_interval_sec": self.log_interval_sec,
            "zones": self.get_zone_analytics(),
            "warnings": self.out_of_stock_warnings(),
            "timeseries": self.get_timeseries_history(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        return path

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()