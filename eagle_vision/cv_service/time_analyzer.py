"""Time-based analytics for tracked construction equipment.

Accumulates time spent in each state (active, idle, loading) per
equipment and computes utilization metrics for dashboard use.
"""

from typing import Dict, List

from cv_service.motion_analyzer import MotionResult


class TimeAnalyzer:
    """Track cumulative time per state for each piece of equipment."""

    def __init__(self) -> None:
        self.data: Dict[str, dict] = {}

    def update(self, motion_results: List[MotionResult], dt: float) -> None:
        """Accumulate time for each equipment based on current state.

        Args:
            motion_results: Current frame's motion analysis output.
            dt: Time elapsed since last frame (seconds).
        """
        for m in motion_results:
            eid = m.equipment_id
            if eid not in self.data:
                self.data[eid] = {
                    "equipment_class": m.equipment_class,
                    "total_time": 0.0,
                    "active_time": 0.0,
                    "idle_time": 0.0,
                    "loading_time": 0.0,
                }

            d = self.data[eid]
            d["total_time"] += dt

            if m.status in ("ACTIVE", "MOVE"):
                d["active_time"] += dt
            elif m.status in ("INACTIVE", "IDLE"):
                d["idle_time"] += dt

            if m.activity == "LOADING":
                d["loading_time"] += dt

    def get_stats(self) -> Dict[str, dict]:
        """Return per-equipment stats with utilization ratios.

        Returns:
            Dict mapping equipment_id to stats including:
              - total_time, active_time, idle_time, loading_time
              - utilization (active_time / total_time)
              - loading_ratio (loading_time / total_time)
        """
        stats: Dict[str, dict] = {}

        for eid, d in self.data.items():
            total = d["total_time"]
            eq_class = d["equipment_class"]

            # Excavator: utilization = active_time / total
            # Dump truck: utilization = loading_time / total
            if eq_class == "dump_truck":
                util_numerator = d["loading_time"]
            else:
                util_numerator = d["active_time"]

            stats[eid] = {
                "equipment_class": eq_class,
                "total_time": round(d["total_time"], 2),
                "active_time": round(d["active_time"], 2),
                "idle_time": round(d["idle_time"], 2),
                "loading_time": round(d["loading_time"], 2),
                "utilization": round(util_numerator / total, 4) if total > 0 else 0.0,
                "loading_ratio": round(d["loading_time"] / total, 4) if total > 0 else 0.0,
            }

        return stats

    def reset(self) -> None:
        """Clear all accumulated data."""
        self.data.clear()
