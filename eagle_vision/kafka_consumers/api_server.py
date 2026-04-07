"""FastAPI server: REST + WebSocket for latest equipment state.

Consumes Kafka events in a background thread and exposes:
  - GET  /state       → latest state of all equipment
  - GET  /state/{id}  → latest state of one equipment
  - WS   /ws          → real-time push on every update

Usage:
    uvicorn kafka_consumers.api_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
import threading
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eagle_vision.api")

# ── Config ───────────────────────────────────────────────────
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "equipment-events")

# ── In-memory state ─────────────────────────────────────────
latest_state: Dict[str, dict] = {}
state_lock = threading.Lock()

# ── WebSocket clients ───────────────────────────────────────
ws_clients: List[WebSocket] = []

# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(title="Eagle Vision API", version="2.0.0")


@app.get("/state")
def get_state():
    """Return latest state of all tracked equipment."""
    with state_lock:
        return {"equipment": list(latest_state.values())}


@app.get("/state/{equipment_id}")
def get_equipment_state(equipment_id: str):
    """Return latest state of a specific equipment."""
    with state_lock:
        data = latest_state.get(equipment_id)
    if data is None:
        return {"error": f"Equipment '{equipment_id}' not found"}
    return data


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Push real-time state updates to connected clients."""
    await ws.accept()
    ws_clients.append(ws)
    logger.info("WebSocket client connected. Total: %d", len(ws_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.remove(ws)
        logger.info("WebSocket client disconnected. Total: %d", len(ws_clients))


async def broadcast(data: dict) -> None:
    """Send data to all connected WebSocket clients."""
    disconnected = []
    for ws in ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_clients.remove(ws)


def _kafka_consumer_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Background thread: consume Kafka and update state."""
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
    except Exception as e:
        logger.error("Kafka unavailable: %s. API will serve stale state.", e)
        return

    logger.info("Kafka consumer started: %s -> '%s'", KAFKA_SERVERS, KAFKA_TOPIC)

    for message in consumer:
        data = message.value
        frame_id = data.get("frame_id")
        objects = data.get("objects", [])

        updated = {}
        with state_lock:
            for obj in objects:
                eid = obj.get("equipment_id")
                if not eid:
                    continue
                obj["frame_id"] = frame_id
                latest_state[eid] = obj
                updated[eid] = obj

        if updated and ws_clients:
            payload = {"frame_id": frame_id, "equipment": list(updated.values())}
            asyncio.run_coroutine_threadsafe(broadcast(payload), loop)


@app.on_event("startup")
async def startup():
    """Start Kafka consumer in a background thread."""
    loop = asyncio.get_event_loop()
    thread = threading.Thread(
        target=_kafka_consumer_loop,
        args=(loop,),
        daemon=True,
    )
    thread.start()
    logger.info("API server started. Kafka consumer running in background.")
