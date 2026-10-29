# 🛒 Smart Retail: Shelf Inventory Analytics & Out-of-Stock Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-Spatial_Vision-green?logo=opencv)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSockets-009688?logo=fastapi&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-%2B_Pandas-003B57?logo=sqlite)

An end-to-end **Computer Vision and Predictive Retail Analytics** platform that transforms raw shelf camera feeds into real-time inventory telemetry, planogram compliance alerts, and time-series out-of-stock forecasting.

---

## 🎯 Business Problem & Value Proposition

Retailers lose billions annually to **Out-of-Stock (OOS)** events and misplaced merchandise (planogram non-compliance), while manual shelf auditing is slow and reactive. 

Instead of stopping at raw bounding boxes, **Smart Retail Analytics** treats a custom-trained **YOLOv8s** object detector as a real-time data generation engine:
1. **Computer Vision Layer (Left Panel):** Detects individual beverage SKUs (`Coca-Cola`, `Fanta`, `Sprite`), maps bottle centroids into physical shelf lane polygons via OpenCV, and highlights empty slots or misplaced items in real time.
2. **Data Science & Business Intelligence Layer (Right Panel):** Logs time-series snapshots into SQLite, computes rolling **Stock Depletion Velocity (units/min)** via Pandas linear regression, and forecasts **Estimated Time-to-Stockout (ETSO)** before shelves hit zero.

---

## ✨ Key Features

* **Custom Fine-Tuned YOLOv8s Detector:** Trained for 50 epochs on Tesla T4 GPU achieving **83.9% mAP50** and **87.5% Precision** across 3 beverage classes (`coca`, `fanta`, `sprite`).
* **OpenCV Spatial Polygon Zoning:** Uses normalized Region-of-Interest (ROI) polygons and `cv2.pointPolygonTest` centroid filtering to assign products strictly to designated brand lanes while ignoring background clutter.
* **Planogram Compliance Auditing:** Automatically flags `MISPLACED ITEM` alerts when a SKU is detected inside another brand's designated lane.
* **Predictive Depletion & ETSO Engine:** Throttled 5-second SQLite time-series logging paired with Pandas linear regression (`np.polyfit`) to calculate depletion rates and predict minutes remaining until stockout.
* **Multi-Source Hot-Swapping:** Drag-and-drop any custom `.mp4` shelf video or `.jpg/.png` test image in the browser to dynamically recalibrate shelf capacity and evaluate model generalization on the fly.
* **Interactive Simulation Controls:** Simulate customer bottle removals (`Keys 1, 2, 3`), planogram misplacements (`Key M`), and restocking (`Key R`) live during executive demonstrations.

---

## 🏗️ System Architecture

```text
[Video Loop / Custom Uploaded Image]
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Step 3: Vision Engine (vision_engine.py)│
│ • YOLOv8s Inference (best.pt)           │
│ • OpenCV Polygon ROIs & Centroid Filter │
│ • Dynamic Capacity & Planogram Math     │
└────────────────┬────────────────────────┘
                 │ (Annotated Frame + Telemetry Dict)
                 ▼
┌─────────────────────────────────────────┐
│ Step 4: Data Engine (analytics_engine.py)│
│ • SQLite Time-Series Logging (5s / Event)│
│ • Pandas Linear Regression Slope (R_d)  │
│ • Estimated Time-to-Stockout (ETSO)     │
└────────────────┬────────────────────────┘
                 │ (Async Worker Thread + JSON)
                 ▼
┌─────────────────────────────────────────┐
│ Step 5 & 6: FastAPI + Dashboard UI      │
│ • Unified WebSocket (/ws/inventory)     │
│ • Base64 MJPEG Stream + Chart.js Graphs │
└─────────────────────────────────────────┘
```
 
---

## 📂 Repository Structure


```text
Smart_Retail_Analytics/
├── main.py                      # Step 5: FastAPI server, WebSocket broadcaster & upload API
├── vision_engine.py             # Step 3: YOLOv8s inference & OpenCV spatial polygon engine
├── analytics_engine.py          # Step 4: SQLite time-series logger & Pandas forecasting math
├── best.pt                      # Fine-tuned YOLOv8s model weights (83.9% mAP50)
├── requirements.txt             # Production Python dependencies (CPU-optimized PyTorch)
├── Dockerfile                   # Container specification for cloud deployment
├── Dataset/
│   └── Bevrage_Shelf.mp4        # Default retail cooler video feed
├── templates/
│   └── index.html               # Step 6: Tailwind CSS + Chart.js executive dashboard
├── notebooks/
│   └── train_bottle_yolov8s.ipynb # Google Colab GPU training notebook & curves
└── tests/
    ├── test_step3.py            # Standalone OpenCV CLI runner
    └── test_step4.py            # Automated headless analytics verification suite
```
---
## 📊 Model Performance (best.pt)
Trained on the `bottle-on-shelves` dataset using `yolov8s.pt` (50 epochs, imgsz=640, `AdamW`):

| Class | Precision (P) | Recall (R) | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes** | **0.875** | **0.791** | **0.839** | **0.685** |
| `coca` | 0.954 | 0.808 | 0.908 | 0.715 |
| `fanta` | 0.821 | 0.856 | 0.849 | 0.720 |
| `sprite` | 0.848 | 0.710 | 0.758 | 0.621 |

---

## 🚀 Quickstart (Run Locally)

### 1. Clone the Repository & Install Dependencies

    git clone https://github.com/Shaivik-ai/Smart-Retail-Shelf-Analytics.git
    cd Smart-Retail-Shelf-Analytics
    pip install -r requirements.txt

### 2. Start the FastAPI Server

    python main.py

### 3. Open the Live Dashboard
Open **`http://localhost:8000`** in your browser.
* **Keyboard Shortcuts:** Press `1`, `2`, `3` to simulate customers removing bottles from Lane 1, Lane 2, or Lane 3; press `M` to toggle a simulated misplaced bottle alert; press `R` to reset all lanes to full stock.
* **Custom Media Testing:** Drag and drop any retail shelf `.mp4` video or `.jpg`/`.png` image into the upload box to test real-time generalization, or click **Default Video** to return to the baseline cooler stream.

---

## 🔌 API & WebSocket Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the split-screen executive web dashboard (`templates/index.html`). |
| `WS` | `/ws/inventory` | Bi-directional WebSocket streaming annotated frames, zone telemetry, warnings, and Chart.js time-series arrays; receives `remove`, `misplace`, and `reset` commands. |
| `POST` | `/api/upload` | Hot-swaps the vision engine to a newly uploaded video or image and auto-calibrates shelf capacity & expected lane brands. |
| `POST` | `/api/source/default` | Restores the default `Bevrage_Shelf.mp4` video loop and resets zone capacities to baseline (`2/2` per lane). |
| `GET` | `/api/analytics` | Returns current per-zone stock counts, depletion rates ($\text{units/min}$), and ETSO forecasts. |
| `GET` | `/api/history` | Returns chronological time-series arrays from SQLite for each shelf zone. |
| `GET` | `/api/snapshot` | Exports the current analytics state to `inventory_snapshot.json`. |


