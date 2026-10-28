"""Step 5 & 6 (Scaled): FastAPI backend + WebSocket streaming + Multi-Source Upload.

Run Locally: python main.py
Open:        http://localhost:8000
"""

import asyncio
import base64
import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from analytics_engine import InventoryAnalyticsEngine
from vision_engine import ShelfVisionEngine

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = BASE_DIR / "Dataset" / "Bevrage_Shelf.mp4"

# Use local directory if writable, otherwise fallback to /tmp for read-only cloud containers
WRITABLE_DIR = BASE_DIR if os.access(BASE_DIR, os.W_OK) else Path(tempfile.gettempdir())
UPLOAD_DIR = WRITABLE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = WRITABLE_DIR / "inventory.db"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vision = ShelfVisionEngine(str(BASE_DIR / "best.pt"), device="cpu")
    app.state.analytics = InventoryAnalyticsEngine(db_path=DB_PATH, log_interval_sec=5.0, reset_db=True)

    # Dynamic media source state: "video" or "image"
    app.state.source_mode = "video"
    app.state.source_name = DEFAULT_VIDEO_PATH.name
    app.state.cap = cv2.VideoCapture(str(DEFAULT_VIDEO_PATH))
    app.state.static_frame = None
    app.state.lock = asyncio.Lock()

    app.state.connections = set()
    app.state.force_log = True  # Force-log first frame immediately on startup
    app.state.broadcast_task = asyncio.create_task(broadcast_loop(app))
    yield
    app.state.broadcast_task.cancel()
    if app.state.cap is not None:
        app.state.cap.release()
    app.state.analytics.close()


app = FastAPI(title="Smart Retail Shelf Analytics", lifespan=lifespan)


# ------------------------------------------------------------------ helpers

def _encode_frame(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


async def broadcast_loop(app: FastAPI):
    """Continuously process active video OR uploaded static image and broadcast over WS."""
    while True:
        frame = None
        interval = 0.04  # Default ~25 FPS

        async with app.state.lock:
            if app.state.source_mode == "image" and app.state.static_frame is not None:
                frame = app.state.static_frame.copy()
                interval = 0.25  # 4 FPS refresh for static images so interactive buttons still work!
            elif app.state.cap is not None and app.state.cap.isOpened():
                fps = app.state.cap.get(cv2.CAP_PROP_FPS) or 25
                interval = 1.0 / max(1.0, min(fps, 30.0))
                ok, read_frame = app.state.cap.read()
                if not ok:
                    app.state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    frame = read_frame

        if frame is None:
            await asyncio.sleep(0.1)
            continue

        # Run YOLOv8 + OpenCV spatial math in a worker thread so FastAPI never blocks
        annotated, telemetry = await asyncio.to_thread(app.state.vision.process_frame, frame)

        force = app.state.force_log
        app.state.force_log = False
        app.state.analytics.maybe_log(telemetry, force=force)

        payload = {
            "type": "frame",
            "source_mode": app.state.source_mode,
            "source_name": app.state.source_name,
            "image": _encode_frame(annotated),
            "telemetry": telemetry,
            "analytics": app.state.analytics.get_zone_analytics(),
            "warnings": app.state.analytics.out_of_stock_warnings(),
            "timeseries": app.state.analytics.get_timeseries_history(limit_per_zone=20),
        }
        await _broadcast(app, json.dumps(payload))
        await asyncio.sleep(interval)


async def _broadcast(app: FastAPI, message: str):
    dead = []
    for ws in list(app.state.connections):
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        app.state.connections.discard(ws)


# ------------------------------------------------------------------ REST & Upload Endpoints

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Error: templates/index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_media(file: UploadFile = File(...)):
    """Hot-swap the live vision engine to a newly uploaded shelf video or image."""
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Upload a video ({', '.join(VIDEO_EXTS)}) or image ({', '.join(IMAGE_EXTS)}).",
        )

    save_path = UPLOAD_DIR / safe_name
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    async with app.state.lock:
        # Reset interactive simulation counters and database logs for the new media
        app.state.vision.simulated_removed = {z: 0 for z in app.state.vision.zones}
        app.state.vision.simulate_misplaced = False
        app.state.analytics.reset_db()

        if ext in IMAGE_EXTS:
            img = cv2.imread(str(save_path))
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
            if app.state.cap is not None:
                app.state.cap.release()
                app.state.cap = None
            app.state.static_frame = img
            app.state.source_mode = "image"
            app.state.source_name = safe_name
        else:
            new_cap = cv2.VideoCapture(str(save_path))
            if not new_cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open uploaded video file.")
            if app.state.cap is not None:
                app.state.cap.release()
            app.state.cap = new_cap
            app.state.static_frame = None
            app.state.source_mode = "video"
            app.state.source_name = safe_name

        app.state.force_log = True

    return JSONResponse({
        "status": "ok",
        "source_mode": app.state.source_mode,
        "source_name": app.state.source_name,
    })


@app.post("/api/source/default")
async def restore_default_source():
    """Switch back to the default Bevrage_Shelf.mp4 demo video."""
    if not DEFAULT_VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Default video not found at {DEFAULT_VIDEO_PATH}")

    async with app.state.lock:
        if app.state.cap is not None:
            app.state.cap.release()
        app.state.cap = cv2.VideoCapture(str(DEFAULT_VIDEO_PATH))
        app.state.static_frame = None
        app.state.source_mode = "video"
        app.state.source_name = DEFAULT_VIDEO_PATH.name
        app.state.vision.simulated_removed = {z: 0 for z in app.state.vision.zones}
        app.state.vision.simulate_misplaced = False
        app.state.analytics.reset_db()
        app.state.force_log = True

    return JSONResponse({
        "status": "ok",
        "source_mode": "video",
        "source_name": DEFAULT_VIDEO_PATH.name,
    })


@app.get("/api/analytics")
async def get_analytics():
    return JSONResponse(app.state.analytics.get_zone_analytics())


@app.get("/api/history")
async def get_history():
    return JSONResponse(app.state.analytics.get_timeseries_history(limit_per_zone=20))


@app.get("/api/snapshot")
async def get_snapshot():
    out_path = WRITABLE_DIR / "inventory_snapshot.json"
    return JSONResponse(app.state.analytics.export_json(str(out_path)))


# ------------------------------------------------------------------ WebSocket

@app.websocket("/ws/inventory")
async def ws_inventory(ws: WebSocket):
    await ws.accept()
    app.state.connections.add(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                action = msg.get("action")
                if action == "remove":
                    zone = msg.get("zone")
                    if zone in app.state.vision.simulated_removed:
                        app.state.vision.simulated_removed[zone] += 1
                        app.state.force_log = True
                elif action == "misplace":
                    app.state.vision.simulate_misplaced = not app.state.vision.simulate_misplaced
                    app.state.force_log = True
                elif action == "reset":
                    app.state.vision.simulated_removed = {z: 0 for z in app.state.vision.zones}
                    app.state.vision.simulate_misplaced = False
                    app.state.force_log = True
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        app.state.connections.discard(ws)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)