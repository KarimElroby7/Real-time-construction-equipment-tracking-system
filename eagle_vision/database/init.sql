-- ═══════════════════════════════════════════════════════════
-- Eagle Vision — TimescaleDB Schema
-- ═══════════════════════════════════════════════════════════

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Equipment Events Table ──────────────────────────────
-- Stores every published event from the CV pipeline.
-- One row per equipment per published frame.
CREATE TABLE IF NOT EXISTS equipment_events (
    time                TIMESTAMPTZ      NOT NULL,
    frame_id            INTEGER          NOT NULL,
    equipment_id        VARCHAR(16)      NOT NULL,
    equipment_class     VARCHAR(32)      NOT NULL,
    video_timestamp     VARCHAR(16)      NOT NULL,
    current_state       VARCHAR(16)      NOT NULL,
    current_activity    VARCHAR(16)      NOT NULL,
    motion_source       VARCHAR(16)      NOT NULL,
    total_tracked_seconds   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_active_seconds    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_idle_seconds      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    utilization_percent     DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

-- Convert to TimescaleDB hypertable partitioned by ingestion time
SELECT create_hypertable('equipment_events', 'time', if_not_exists => TRUE);

-- ── Indexes ─────────────────────────────────────────────
-- Fast lookups by equipment and by state
CREATE INDEX IF NOT EXISTS idx_events_equipment_id
    ON equipment_events (equipment_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_events_state
    ON equipment_events (current_state, time DESC);
