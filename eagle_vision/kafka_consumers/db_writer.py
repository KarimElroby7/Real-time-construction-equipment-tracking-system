"""Kafka consumer that writes equipment events to TimescaleDB.

Consumes JSON messages from the "equipment-events" topic and inserts
each object's state into the equipment_events table.

Usage:
    python -m kafka_consumers.db_writer
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import psycopg2
from kafka import KafkaConsumer

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eagle_vision.db_writer")

# ── Config from .env ─────────────────────────────────────────
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "equipment-events")

DB_HOST = os.getenv("TIMESCALE_HOST", "localhost")
DB_PORT = int(os.getenv("TIMESCALE_PORT", "5432"))
DB_NAME = os.getenv("TIMESCALE_DB", "eagle_vision")
DB_USER = os.getenv("TIMESCALE_USER", "eagle")
DB_PASS = os.getenv("TIMESCALE_PASSWORD", "eagle_pass")

# ── SQL ──────────────────────────────────────────────────────
# Table is created by database/init.sql via Docker.
# This is a fallback in case init.sql didn't run.
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS equipment_events (
    time                TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    frame_id            INTEGER          NOT NULL,
    equipment_id        VARCHAR(16)      NOT NULL,
    equipment_class     VARCHAR(32)      NOT NULL,
    video_timestamp     VARCHAR(16)      NOT NULL DEFAULT '',
    current_state       VARCHAR(16)      NOT NULL,
    current_activity    VARCHAR(16)      NOT NULL,
    motion_source       VARCHAR(16)      NOT NULL DEFAULT '',
    total_tracked_seconds   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_active_seconds    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_idle_seconds      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    utilization_percent     DOUBLE PRECISION NOT NULL DEFAULT 0.0
);
"""

INSERT_EVENT = """
INSERT INTO equipment_events
    (time, frame_id, equipment_id, equipment_class, video_timestamp,
     current_state, current_activity, motion_source)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
"""


def connect_db():
    """Connect to TimescaleDB and ensure table exists."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE)
    logger.info("DB connected: %s@%s:%s/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME)
    return conn


def process_message(cur, data: dict) -> int:
    """Insert objects from a single Kafka message. Returns rows inserted."""
    frame_id = data.get("frame_id")
    event_time = data.get("timestamp") or datetime.now(timezone.utc)
    video_ts = data.get("video_timestamp", "")
    objects = data.get("objects", [])
    count = 0

    for obj in objects:
        cur.execute(INSERT_EVENT, (
            event_time,
            frame_id,
            obj.get("equipment_id"),
            obj.get("equipment_class"),
            video_ts,
            obj.get("status"),
            obj.get("activity"),
            obj.get("motion_source") or "",
        ))
        count += 1

    return count


def main():
    # Connect to DB
    try:
        conn = connect_db()
    except Exception as e:
        logger.error("Cannot connect to DB: %s", e)
        sys.exit(1)

    # Connect to Kafka
    logger.info("Connecting to Kafka: %s -> topic '%s'", KAFKA_SERVERS, KAFKA_TOPIC)
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
    except Exception as e:
        logger.error("Cannot connect to Kafka: %s", e)
        conn.close()
        sys.exit(1)

    logger.info("Listening for events... (Ctrl+C to stop)")
    total_rows = 0

    try:
        with conn.cursor() as cur:
            for message in consumer:
                data = message.value
                rows = process_message(cur, data)
                total_rows += rows

                if total_rows % 50 == 0 and total_rows > 0:
                    logger.info("Inserted %d rows total", total_rows)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.close()
        conn.close()
        logger.info("Done. Total rows inserted: %d", total_rows)


if __name__ == "__main__":
    main()
