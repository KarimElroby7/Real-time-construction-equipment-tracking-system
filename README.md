# 🚜 Eagle Vision — Real-Time Construction Equipment Activity Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-blueviolet)](https://github.com/ultralytics/ultralytics)

A full production-grade system for **real-time construction equipment monitoring and activity analysis** using computer vision, multi-object tracking, and streaming technologies.

Eagle Vision detects, tracks, and analyzes excavators and dump trucks from video feeds, then computes class-aware utilization metrics and streams the results to a live dashboard.

---

## 🎥 Demo

![Tracking Demo](assets/demo_tracking.gif)

![Tracking2 Demo](assets/demo_tracking2.gif)

---

## 🎯 Overview

Eagle Vision is an end-to-end AI system that turns raw construction site footage into actionable analytics:

- Continuous detection and identity-stable tracking of equipment
- Activity classification (ACTIVE / IDLE / LOADING) based on motion and proximity
- Class-aware utilization metrics (excavator vs. dump truck)
- Real-time streaming through Kafka, FastAPI, and TimescaleDB
- Live dashboard with final analytics auto-display

---

## ✨ Features

- 🔍 **Real-time detection** of excavators and dump trucks using YOLO11 (yolo11l)
- 🎯 **Stable multi-object tracking** with BoT-SORT and persistent friendly IDs (EX-001, TR-001)
- ⚡ **Activity classification**:
  - ACTIVE — equipment is operating
  - IDLE — equipment is stationary
  - LOADING — dump truck is near an active excavator
- 📊 **Class-aware utilization**:
  - Excavator → utilization based on Active Time
  - Dump Truck → utilization based on Loading Time
- 📈 **Detailed time-based analytics** per equipment
- 🧩 **Modular architecture** — clean separation between detection, tracking, analysis, and UI
- 🛑 **Graceful shutdown** via ESC key — guarantees analytics are written to disk
- 📡 **End-to-end streaming pipeline** with Kafka, FastAPI, and TimescaleDB
- 🖥️ **Live Streamlit dashboard** with auto-refresh and final analytics display

---

## 🧠 Core Technologies

| Layer            | Technology                       |
| ---------------- | -------------------------------- |
| Detection        | YOLO11 (yolo11l)                 |
| Tracking         | BoT-SORT                         |
| Motion Analysis  | Custom optical-flow analyzer     |
| Time Analytics   | Custom time accumulator          |
| Streaming        | Apache Kafka                     |
| Backend API      | FastAPI                          |
| Storage          | TimescaleDB                      |
| Dashboard        | Streamlit                        |
| Video Display    | OpenCV                           |

---

## 🏗️ System Architecture

```text
Video Input
   ↓
YOLO11 Detection
   ↓
BoT-SORT Tracking
   ↓
Motion Analyzer (Activity Classification)
   ↓
Time Analyzer (Metrics Calculation)
   ↓
Kafka → FastAPI → TimescaleDB
   ↓
Streamlit Dashboard
```

---

## 📊 Sample Results

| Equipment ID | Type       | Total Time | Moving Time | Idle Time | Loading Time | Utilization | Loading Ratio |
| ------------ | ---------- | ---------- | ----------- | --------- | ------------ | ----------- | ------------- |
| EX-001       | Excavator  | 4.1s       | 0.03s       | 4.07s     | 0.0s         | 0.8%        | 0%            |
| TR-001       | Dump Truck | 4.1s       | 3.97s       | 0.13s     | 0.0s         | 96.7%       | 0%            |

---

## 📖 Metrics Explanation

| Metric          | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| Total Time      | Total duration the equipment was tracked                                    |
| Moving Time     | Time when a dump truck is actively moving                                   |
| Active Time     | Time when an excavator is operating                                         |
| Idle Time       | Time when the equipment is stationary                                       |
| Loading Time    | Time when a dump truck is being loaded by an active excavator               |
| Utilization     | Class-aware efficiency metric (Active for excavator, Loading for truck)     |
| Loading Ratio   | Percentage of total tracked time spent in the LOADING state                 |

---

## 📂 Project Structure

```
.
├── assets/
│   ├── demo_tracking.gif          # Demo animation (used in README)
│   └── demo_tracking2.gif
│
├── eagle_vision/
│   ├── cv_service/
│   │   ├── detector.py            # YOLO detection wrapper
│   │   ├── tracker.py             # BoT-SORT tracking + friendly IDs
│   │   ├── motion_analyzer.py     # Optical flow + activity classification
│   │   ├── time_analyzer.py       # Time accumulation + utilization metrics
│   │   └── pipeline.py            # Unified detect → track → analyze pipeline
│   │
│   ├── kafka_consumers/
│   │   ├── api_server.py          # FastAPI live state endpoint
│   │   └── db_writer.py           # Kafka → TimescaleDB consumer
│   │
│   ├── ui/
│   │   └── app.py                 # Streamlit analytics dashboard
│   │
│   ├── input/                     # Place input videos here (gitignored)
│   ├── output/
│   │   └── analytics.sample.json  # Reference output schema (committed)
│   ├── models/                    # Place YOLO weights here (gitignored)
│   │
│   ├── main.py                    # Single-entry system launcher
│   ├── run_pipeline.py            # CV pipeline entry point
│   ├── docker-compose.yml         # Kafka + TimescaleDB stack
│   └── .env.example               # Environment template
│
├── README.md
├── LICENSE                        # MIT
├── requirements.txt               # Consolidated Python dependencies
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/KarimElroby7/eagle-vision.git
cd eagle-vision
```

### 2. Create the Python environment

```bash
conda create -n eagle_vision python=3.10
conda activate eagle_vision
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and edit it with your local settings (database credentials, Kafka broker, etc.):

```bash
cp eagle_vision/.env.example eagle_vision/.env
```

### 5. Add the YOLO model weights

Place your trained model at:

```
eagle_vision/models/best.pt
```

> **Note**: Model weights are not committed to the repository. Use your own trained model or a base YOLO checkpoint.

### 6. Add an input video

Place your test video at:

```
eagle_vision/input/sample.mp4
```

### 7. Start Docker services (Kafka + TimescaleDB)

```bash
cd eagle_vision
docker compose up -d
```

---

## 🚀 Usage

> The main entry point of the system is located inside the `eagle_vision/` directory.

Run the entire system from a single entry point:

```bash
cd eagle_vision
python main.py
```

You can also pass a custom video:

```bash
cd eagle_vision
python main.py --video input/sample5.mp4
```

### What happens

- 🐳 Docker services start (Kafka + TimescaleDB)
- 🚀 FastAPI server launches on port 8000
- 💾 Kafka → DB writer starts consuming events
- 🎥 OpenCV window opens with the live annotated video
- 🖥️ Streamlit dashboard launches at http://localhost:8502
- 🌐 Browser opens automatically

### Controls

- Press **ESC** in the OpenCV window → pipeline stops gracefully
- The `finally` block writes `analytics.json` with `"status": "completed"`
- The Streamlit dashboard detects the new status and auto-displays the final analytics

---

## 📊 Output

### 1. JSON Output

```
eagle_vision/output/analytics.json
```

A reference example is committed at `eagle_vision/output/analytics.sample.json` so you can preview the schema before running the pipeline.

Structure:

```json
{
  "status": "completed",
  "video_info": { ... },
  "summary": {
    "EX-001": {
      "equipment_class": "excavator",
      "total_time": 4.1,
      "active_time": 0.03,
      "idle_time": 4.07,
      "loading_time": 0.0,
      "utilization": 0.008,
      "loading_ratio": 0.0
    },
    "TR-001": { ... }
  }
}
```

### 2. Dashboard Metrics

For each tracked piece of equipment the dashboard shows:

- ⏱ Total Tracked Time
- 🔵 Active Time (excavator) / Moving Time (dump truck)
- 🔴 Idle Time
- 🟡 Loading Time
- 📊 Utilization (class-aware)
- 📈 Loading Ratio

---

## 🧠 Activity Logic

### LOADING Detection

A dump truck is classified as **LOADING** when:

1. There is at least one **active** excavator in the frame
2. The center-to-center distance between the truck and the nearest active excavator is below the loading threshold

```text
if active_excavator_nearby AND distance < loading_threshold:
    activity = LOADING
else:
    activity = WAITING
```

The loading distance threshold is configurable via the `Pipeline` constructor.

### Class-Aware Utilization

| Equipment   | Utilization Formula                |
| ----------- | ---------------------------------- |
| Excavator   | `active_time / total_time`         |
| Dump Truck  | `loading_time / total_time`        |

This reflects how each machine type contributes value on a real construction site:
an excavator is productive when it is operating, while a dump truck is productive when it is being loaded.

---

## Challenges & Solutions

| Challenge                                     | Root Cause                                  | Solution                                                      | Trade-offs                                                      |
| --------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- |
| Identity switching under occlusion            | Weak Re-ID + visually similar equipment     | Custom tracker with spatial/geometric ID matching             | Requires per-camera tuning                                      |
| Class flickering (`excavator` ↔ `dump_truck`) | Model confusion under occlusion             | Majority voting + class locking after threshold               | Wrong locks require track reset                                 |
| Missed `LOADING` events                       | IoU too restrictive for realistic scenarios | Switched to center-distance metric                            | Threshold sensitive to camera setup                             |
| False motion detection                        | Optical flow noise on stationary objects    | Class-specific motion thresholds + motion-type classification | Very slow motion may be misclassified                           |
| Misleading utilization metrics                | Same formula applied across equipment types | Class-aware utilization (movement vs loading)                 | Must interpret results according to equipment class             |
| Memory / scalability issues                   | Storing all frames in memory                | Split summary vs full data files                              | Full data still in memory; optimization needed for long streams |
---


## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


