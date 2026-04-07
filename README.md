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

The following table summarizes the principal technical challenges encountered during implementation, their root causes, and the engineering decisions adopted to resolve them.

| Challenge | Root Cause | Solution | Trade-offs |
|-----------|------------|----------|------------|
| Identity switching between excavators and dump trucks under occlusion. | BoT-SORT's appearance Re-ID is weak for visually similar heavy equipment; transient confidence drops cause tracks to re-spawn with new IDs. | Custom tracker wrapper with persistent friendly IDs, greedy spatial ID locking, and geometric Re-ID fallback using position and size similarity. | Adds per-frame matching overhead; `lock_distance` requires per-camera tuning. |
| Class flicker — objects oscillating between `excavator` and `dump_truck` across frames. | YOLO misclassifies visually similar equipment under partial occlusion, propagating into downstream logic. | Class voting with a majority-vote warmup phase; the dominant class is permanently locked after `class_lock_threshold` frames. | Locked classes cannot be corrected later; wrong locks require a track reset. |
| `LOADING` false negatives when trucks and excavators were close but not overlapping. | Initial proximity logic used IoU, which is zero unless boxes intersect — too restrictive for realistic loading scenes. | Replaced IoU with Euclidean center-to-center distance, gated on the nearest *active* excavator and a configurable `loading_distance` threshold. | Threshold is camera- and zoom-dependent; no single global value generalizes. |
| Spurious optical-flow magnitude on stationary excavators producing false `ACTIVE` states. | Farneback dense flow over the full bbox captures background pixels and arm-tip jitter, yielding non-zero averages on idle frames. | Empirically tuned per-class magnitude thresholds plus a motion-source classifier (`static` / `arm only` / `full body`) over magnitude bands. | Thresholds are dataset-dependent; very slow motion may be classified as static. |
| Class-agnostic utilization formula produced misleadingly low values for dump trucks. | Productive time semantics differ by class: excavators are productive when moving, dump trucks when being loaded (i.e., stationary). | Class-aware utilization in `TimeAnalyzer.get_stats()`: excavator uses `active_time / total_time`, dump truck uses `loading_time / total_time`; UI labels adapt accordingly. | The `utilization` field carries class-conditional semantics and must be interpreted alongside `equipment_class`. |
| Frame-level analytics output grew unboundedly for long videos, slowing the final shutdown write. | All per-frame detections were accumulated in memory and serialized in one operation, yielding linear growth with video length. | Split persistence into a compact `analytics.json` (status + summary, atomic write) and a larger best-effort `analytics_full.json` written afterward; the UI never blocks on the larger file. | The full frame list is still held in memory; multi-hour streams require further refactoring. |
| The dashboard intermittently read truncated `analytics.json` during pipeline writes. | `json.dump()` is non-atomic on Windows; concurrent readers could observe a half-written document. | Atomic write helper using a temporary file → `flush()` → `fsync()` → `os.replace()`; reader retries with structural validation (trailing-brace check). | Marginal disk overhead per write; the atomic rename is single-filesystem only. |
| Pipeline subprocess was force-killed before its `finally` block could persist analytics. | The orchestrator's shutdown handler called `terminate()` immediately on exit, racing the in-progress write. | Orchestrator now `wait()`s on the pipeline for up to 30 s before any forced termination; the pipeline writes analytics *before* any OpenCV or Kafka cleanup. | Adds a 30 s grace period to shutdown; truly wedged pipelines are still force-killed after the timeout. |
| `cv2.destroyAllWindows()` deadlocked the shutdown path on Windows. | OpenCV's Win32 window teardown can hang when invoked after Streamlit and Kafka have torn down their event loops. | Reordered the `finally` block so analytics persistence and Kafka flushing complete before OpenCV cleanup. | Window cleanup is now best-effort; the OpenCV window may persist briefly if teardown stalls. |
| TimescaleDB `INSERT` statements failed with `NOT NULL` violations on `time` and `motion_source`. | The `INSERT` omitted the `time` column relying on `DEFAULT NOW()` against a non-nullable column; `dict.get("motion_source", "")` returns `None` when the key exists with a `None` value. | Include `time` explicitly with a `datetime.now(timezone.utc)` fallback; coerce `None` via `obj.get("motion_source") or ""`. | None. |
| Child processes leaked when the orchestrator was terminated on Windows. | `subprocess.terminate()` signals only the immediate child, not its descendants; `psutil` is not guaranteed to be installed. | `kill_proc_tree()` helper prefers `psutil.Process.children(recursive=True)` and falls back to `taskkill /F /T /PID` on Windows. | The fallback is Windows-specific and would require adaptation for UNIX-only deployments. |
| Streamlit duplicate-element errors and race-condition rendering (showing `processing` and `completed` simultaneously). | The original UI used a `while True` polling loop; Streamlit re-executes the script on every interaction, producing concurrent render passes with duplicate widget keys. | Refactored to a single-pass render model: read state once, render once, then trigger `st.rerun()` after a sleep interval while still processing. All transient state migrated to `st.session_state`. | Status transitions incur ~1 s latency, equal to one rerun cycle. |

---

## 🔮 Future Improvements

- 🧠 Smarter LOADING detection (distance + bucket motion + scene context)
- ⚡ Real-time Kafka → UI streaming (eliminate JSON polling)
- 📷 Multi-camera support and cross-camera ID handoff
- 📊 Advanced metrics (cycle time, loads per hour, fleet efficiency)
- 🤖 Model optimization with TensorRT / ONNX
- 🏗️ Support for additional equipment classes (loaders, bulldozers, cranes)

---

## 🏆 Highlights

- End-to-end AI system: **Detection → Tracking → Activity → Analytics → UI**
- Production-grade architecture with modular, testable components
- Class-aware utilization that reflects real construction workflows
- Graceful shutdown guarantees analytics persistence
- Single-command launch of the entire stack

---

## 📌 Notes

- Optimized for typical construction site camera setups
- Tracking quality depends on camera angle, resolution, and lighting
- The loading distance threshold should be tuned per dataset / camera zoom
- Excavator detection benefits from higher-confidence thresholds than dump trucks

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Built as a real-world AI system for construction equipment monitoring and analytics.

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to open issues or contribute!
