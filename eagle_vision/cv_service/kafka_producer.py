"""Kafka producer for streaming CV pipeline events.

Sends per-frame equipment state to a Kafka topic for downstream
consumers (DB writer, API, dashboard).
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("eagle_vision.kafka")


class EventProducer:
    """Sends equipment events to Kafka.

    Wraps kafka-python's KafkaProducer with JSON serialization
    and graceful fallback if Kafka is unavailable.

    Usage::

        producer = EventProducer()
        producer.send({"equipment_id": "EX-001", "status": "ACTIVE", ...})
        producer.flush()
        producer.close()
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> None:
        self._servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092",
        )
        self._topic = topic or os.getenv("KAFKA_TOPIC", "equipment-events")
        self._producer = None

        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self._servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info("Kafka connected: %s -> topic '%s'", self._servers, self._topic)
        except Exception as e:
            logger.warning("Kafka unavailable (%s). Events will be skipped.", e)

    @property
    def connected(self) -> bool:
        return self._producer is not None

    def send(self, payload: dict) -> None:
        """Send a JSON payload to the configured topic."""
        if not self._producer:
            return
        self._producer.send(self._topic, payload)

    def flush(self) -> None:
        """Flush pending messages."""
        if self._producer:
            self._producer.flush()

    def close(self) -> None:
        """Flush and close the producer."""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Kafka producer closed.")
