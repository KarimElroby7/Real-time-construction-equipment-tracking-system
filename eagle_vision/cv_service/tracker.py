"""BoT-SORT wrapper with friendly ID assignment for construction equipment.

Tracking and appearance-based Re-ID are handled by Ultralytics' BoT-SORT
via ``model.track()``.  This module adds:
  - Persistent friendly IDs (EX-001, TR-001) per tracked object.
  - Class voting with majority-vote warmup and permanent lock.
  - Fallback geometric Re-ID for when BoT-SORT loses a track
    (catches obvious same-position reappearances).
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("eagle_vision.tracker")


# Prefix mapping for friendly IDs
CLASS_PREFIX: Dict[str, str] = {
    "excavator": "EX",
    "dump_truck": "TR",
}


@dataclass
class TrackedEquipment:
    """A tracked piece of equipment with persistent identity."""

    tracker_id: int        # BoT-SORT internal numeric ID
    equipment_id: str      # Friendly ID, e.g. "EX-001", "TR-002"
    equipment_class: str   # "excavator" or "dump_truck"
    bbox: np.ndarray       # [x1, y1, x2, y2] absolute pixel coords
    confidence: float


@dataclass
class _LostTrack:
    """Fallback buffer entry for a recently lost track."""

    equipment_id: str
    equipment_class: str
    last_bbox: np.ndarray
    last_center: np.ndarray
    last_area: float
    lost_time: float


class EquipmentTracker:
    """Friendly ID + class voting + fallback Re-ID wrapper on top of BoT-SORT.

    BoT-SORT handles tracking, appearance Re-ID, and global motion
    compensation internally via ``model.track()``.  This wrapper adds:
      - Persistent friendly IDs (EX-001, TR-001).
      - Class voting with majority-vote warmup and permanent lock.
      - Fallback geometric Re-ID: when BoT-SORT assigns a new tracker_id,
        checks if it matches a recently-lost track by position and size.
        This catches cases where BoT-SORT's appearance Re-ID fails
        (weak model features) but the object is obviously the same.
    """

    def __init__(
        self,
        class_lock_threshold: int = 15,
        lost_timeout: float = 30.0,
        max_match_distance: float = 200.0,
        min_size_similarity: float = 0.5,
        lock_distance: float = 100.0,
    ) -> None:
        """
        Args:
            class_lock_threshold: Frames of class votes before locking.
            lost_timeout: Seconds to keep lost tracks for fallback Re-ID.
            max_match_distance: Max center distance (px) for fallback match.
            min_size_similarity: Min area ratio for fallback match (0-1).
            lock_distance: Max center distance (px) for spatial ID locking.
        """
        self.class_lock_threshold = class_lock_threshold
        self.lost_timeout = lost_timeout
        self.max_match_distance = max_match_distance
        self.min_size_similarity = min_size_similarity
        self.lock_distance = lock_distance

        # State
        self._class_counters: Dict[str, int] = {}
        self._id_map: Dict[int, str] = {}                 # tracker_id -> friendly_id
        self._class_map: Dict[int, str] = {}               # tracker_id -> equipment_class
        self._class_votes: Dict[int, Dict[str, int]] = {}  # tid -> {class: count}
        self._last_bboxes: Dict[int, np.ndarray] = {}      # tracker_id -> last bbox
        self._active_ids: set = set()                       # tracker_ids active last frame
        self._lost_tracks: Dict[str, _LostTrack] = {}      # friendly_id -> LostTrack

        # Spatial ID lock state
        self._locked_positions: Dict[str, np.ndarray] = {}  # eq_id -> last center
        self._locked_last_seen: Dict[str, float] = {}       # eq_id -> timestamp

        # Per-object history for class persistence + motion checks
        self._history: Dict[str, dict] = {}  # eq_id -> {class, last_bbox, frames_seen}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, results, class_names: dict) -> List[TrackedEquipment]:
        """Process ``model.track()`` results into TrackedEquipment.

        Args:
            results: Raw results from ``model.track()``.
            class_names: Model class name mapping ``{id: name}``.

        Returns:
            List of ``TrackedEquipment`` with persistent friendly IDs.
        """
        tracked_results: List[TrackedEquipment] = []
        current_active: set = set()

        if not results or results[0].boxes is None:
            self._mark_lost(set())
            return tracked_results

        boxes = results[0].boxes

        # No tracker IDs yet (first few frames while tracker initializes)
        if boxes.id is None:
            self._mark_lost(set())
            return tracked_results

        for i in range(len(boxes)):
            tid = int(boxes.id[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
            conf = float(boxes.conf[i].item())
            class_id = int(boxes.cls[i].item())
            eq_class = class_names.get(class_id, "unknown")

            # Skip non-equipment classes
            if eq_class not in CLASS_PREFIX:
                continue

            current_active.add(tid)
            self._last_bboxes[tid] = bbox.copy()

            if tid in self._id_map:
                # Known track — remove from lost buffer if present
                self._lost_tracks.pop(self._id_map[tid], None)
                # Class voting
                votes = self._class_votes.setdefault(tid, {})
                votes[eq_class] = votes.get(eq_class, 0) + 1
                total = sum(votes.values())
                if total <= self.class_lock_threshold:
                    majority = max(votes, key=votes.get)
                    if majority != self._class_map[tid]:
                        old_class = self._class_map[tid]
                        self._class_map[tid] = majority
                        self._id_map[tid] = self._next_id(majority)
                        logger.info(
                            "Class flip tid=%d: %s -> %s (votes=%s)",
                            tid, old_class, majority, dict(votes),
                        )
            else:
                # New tracker_id — try fallback geometric Re-ID first
                reid_result = self._try_fallback_reid(eq_class, bbox)
                if reid_result:
                    self._id_map[tid] = reid_result[0]
                    self._class_map[tid] = reid_result[1]
                    self._class_votes[tid] = {
                        reid_result[1]: self.class_lock_threshold,
                    }
                    logger.info(
                        "Fallback Re-ID tid=%d -> %s (%s) conf=%.2f",
                        tid, reid_result[0], reid_result[1], conf,
                    )
                else:
                    self._id_map[tid] = self._next_id(eq_class)
                    self._class_map[tid] = eq_class
                    self._class_votes[tid] = {eq_class: 1}
                    logger.info(
                        "New track tid=%d -> %s (%s) conf=%.2f",
                        tid, self._id_map[tid], eq_class, conf,
                    )

            tracked_results.append(TrackedEquipment(
                tracker_id=tid,
                equipment_id=self._id_map[tid],
                equipment_class=self._class_map[tid],
                bbox=bbox,
                confidence=conf,
            ))

        self._mark_lost(current_active)
        return self._apply_id_lock(tracked_results)

    def reset(self) -> None:
        """Reset all tracker state."""
        self._class_counters.clear()
        self._id_map.clear()
        self._class_map.clear()
        self._class_votes.clear()
        self._last_bboxes.clear()
        self._active_ids.clear()
        self._lost_tracks.clear()
        self._locked_positions.clear()
        self._locked_last_seen.clear()
        self._history.clear()

    # ------------------------------------------------------------------
    # Spatial ID lock (runs on top of BoT-SORT output)
    # ------------------------------------------------------------------

    def _apply_id_lock(
        self, tracked_results: List[TrackedEquipment],
    ) -> List[TrackedEquipment]:
        """Override IDs using spatial consistency to prevent switching.

        For each detection, find the closest previously-locked identity
        by center distance.  If within ``lock_distance`` pixels, reuse
        that identity regardless of what BoT-SORT assigned internally.
        """
        now = time.time()

        if not tracked_results:
            return tracked_results

        # Compute centers for current detections
        current: List[tuple] = []
        for t in tracked_results:
            cx = (t.bbox[0] + t.bbox[2]) / 2
            cy = (t.bbox[1] + t.bbox[3]) / 2
            current.append((t, np.array([cx, cy], dtype=np.float32)))

        # First frame — seed locked identities from BoT-SORT output
        if not self._locked_positions:
            for t, center in current:
                self._locked_positions[t.equipment_id] = center
                self._locked_last_seen[t.equipment_id] = now
            return tracked_results

        # Expire stale locked entries
        expired = [
            lid for lid, ts in self._locked_last_seen.items()
            if now - ts > self.lost_timeout
        ]
        for lid in expired:
            del self._locked_positions[lid]
            del self._locked_last_seen[lid]

        if not self._locked_positions:
            # All expired — re-seed
            for t, center in current:
                self._locked_positions[t.equipment_id] = center
                self._locked_last_seen[t.equipment_id] = now
            return tracked_results

        # Build (distance, current_idx, locked_id) pairs, sorted by distance
        locked_ids = list(self._locked_positions.keys())
        pairs: List[tuple] = []
        for ci, (_, center) in enumerate(current):
            for lid in locked_ids:
                dist = float(np.linalg.norm(center - self._locked_positions[lid]))
                pairs.append((dist, ci, lid))
        pairs.sort()

        # Greedy assignment — closest pair first, no double-assignment
        used_curr: set = set()
        used_locked: set = set()
        assignments: Dict[int, str] = {}

        for dist, ci, lid in pairs:
            if ci in used_curr or lid in used_locked:
                continue
            if dist < self.lock_distance:
                assignments[ci] = lid
                used_curr.add(ci)
                used_locked.add(lid)

        # Build result with overridden IDs + history checks
        result: List[TrackedEquipment] = []
        for ci, (t, center) in enumerate(current):
            if ci in assignments:
                locked_id = assignments[ci]
                if locked_id != t.equipment_id:
                    dist = float(np.linalg.norm(
                        center - self._locked_positions[locked_id],
                    ))
                    logger.info(
                        "ID lock: %s -> %s (dist=%.0fpx)",
                        t.equipment_id, locked_id, dist,
                    )
                    t = TrackedEquipment(
                        tracker_id=t.tracker_id,
                        equipment_id=locked_id,
                        equipment_class=t.equipment_class,
                        bbox=t.bbox,
                        confidence=t.confidence,
                    )
                self._locked_positions[locked_id] = center
                self._locked_last_seen[locked_id] = now
            else:
                # No match — register as a new locked identity
                self._locked_positions[t.equipment_id] = center
                self._locked_last_seen[t.equipment_id] = now

            # --- History: class persistence + motion guard ---
            eid = t.equipment_id
            if eid in self._history:
                hist = self._history[eid]

                # Motion consistency: relative threshold (50% of bbox diagonal)
                prev_bbox = hist["last_bbox"]
                prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                diag = float(np.sqrt(
                    (prev_bbox[2] - prev_bbox[0]) ** 2
                    + (prev_bbox[3] - prev_bbox[1]) ** 2
                ))
                max_move = max(diag * 0.5, 50.0)

                move_dist = float(np.linalg.norm(
                    center - np.array([prev_cx, prev_cy], dtype=np.float32),
                ))
                if move_dist > max_move:
                    # Soft freeze: keep the detection but don't update history
                    logger.info(
                        "Motion freeze %s: moved %.0fpx (limit %.0fpx), "
                        "keeping previous position",
                        eid, move_dist, max_move,
                    )
                    hist["frames_seen"] += 1
                    result.append(t)
                    continue

                # Class persistence: keep established class
                if t.equipment_class != hist["class"]:
                    logger.info(
                        "Class override %s: %s -> %s (keeping %s, seen %d frames)",
                        eid, t.equipment_class, hist["class"],
                        hist["class"], hist["frames_seen"],
                    )
                    t = TrackedEquipment(
                        tracker_id=t.tracker_id,
                        equipment_id=eid,
                        equipment_class=hist["class"],
                        bbox=t.bbox,
                        confidence=t.confidence,
                    )

                hist["last_bbox"] = t.bbox.copy()
                hist["frames_seen"] += 1
            else:
                self._history[eid] = {
                    "class": t.equipment_class,
                    "last_bbox": t.bbox.copy(),
                    "frames_seen": 1,
                }

            result.append(t)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_id(self, equipment_class: str) -> str:
        prefix = CLASS_PREFIX.get(equipment_class, "UK")
        count = self._class_counters.get(equipment_class, 0) + 1
        self._class_counters[equipment_class] = count
        return f"{prefix}-{count:03d}"

    def _try_fallback_reid(
        self, equipment_class: str, bbox: np.ndarray,
    ) -> Optional[tuple]:
        """Geometric fallback Re-ID: match by position + size.

        Only fires when BoT-SORT has already failed to re-identify.
        Returns ``(friendly_id, locked_class)`` or ``None``.
        """
        now = time.time()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        best_id: Optional[str] = None
        best_class: Optional[str] = None
        best_dist = float("inf")
        expired: List[str] = []

        for eq_id, lost in self._lost_tracks.items():
            age = now - lost.lost_time
            if age > self.lost_timeout:
                expired.append(eq_id)
                continue

            # Center distance check
            dist = np.sqrt(
                (cx - lost.last_center[0]) ** 2
                + (cy - lost.last_center[1]) ** 2
            )
            if dist > self.max_match_distance:
                continue

            # Size similarity check
            max_area = max(area, lost.last_area)
            size_ratio = min(area, lost.last_area) / max_area if max_area > 0 else 0
            if size_ratio < self.min_size_similarity:
                continue

            # Pick the closest match
            if dist < best_dist:
                best_dist = dist
                best_id = eq_id
                best_class = lost.equipment_class

        for eq_id in expired:
            del self._lost_tracks[eq_id]

        if best_id is not None:
            del self._lost_tracks[best_id]
            return best_id, best_class

        return None

    def _mark_lost(self, current_active: set) -> None:
        """Move newly-disappeared tracks into the lost buffer."""
        now = time.time()
        newly_lost = self._active_ids - current_active

        for tid in newly_lost:
            eq_id = self._id_map.get(tid)
            if eq_id and eq_id not in self._lost_tracks:
                bbox = self._last_bboxes.get(tid, np.zeros(4))
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                self._lost_tracks[eq_id] = _LostTrack(
                    equipment_id=eq_id,
                    equipment_class=self._class_map.get(tid, "unknown"),
                    last_bbox=bbox,
                    last_center=np.array([cx, cy]),
                    last_area=float(area),
                    lost_time=now,
                )

        # Expire old entries
        expired = [
            eid for eid, lt in self._lost_tracks.items()
            if now - lt.lost_time > self.lost_timeout
        ]
        for eid in expired:
            del self._lost_tracks[eid]

        self._active_ids = current_active
