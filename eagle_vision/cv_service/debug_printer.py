"""Debug output for motion and time analytics.

Prints structured JSON + human-readable explanation per equipment,
designed for debugging and technical assessment submission.
"""

import json
from typing import Dict, List, Optional

from cv_service.motion_analyzer import MotionResult, DEFAULT_FLOW_THRESHOLDS


def _format_timestamp(frame_idx: int, fps: float) -> str:
    """Convert frame index to HH:MM:SS.mmm timestamp."""
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def print_debug(
    frame_idx: int,
    fps: float,
    motion_results: List[MotionResult],
    time_stats: Dict[str, dict],
    loading_distance: float = 400.0,
) -> None:
    """Print JSON debug + explanation for each tracked equipment.

    Args:
        frame_idx: Current frame number.
        fps: Video FPS (for timestamp calculation).
        motion_results: Current frame's motion analysis output.
        time_stats: Output of TimeAnalyzer.get_stats().
        loading_distance: Loading distance threshold (for explanation).
    """
    timestamp = _format_timestamp(frame_idx, fps)

    print(f"\n{'='*60}")
    print(f"  DEBUG OUTPUT — Frame {frame_idx} | {timestamp}")
    print(f"{'='*60}")

    for m in motion_results:
        thresh = DEFAULT_FLOW_THRESHOLDS.get(m.equipment_class, 0.5)
        stats = time_stats.get(m.equipment_id, {})

        _print_json(frame_idx, timestamp, m, thresh, stats)
        _print_explanation(m, thresh, stats, loading_distance)


def _print_json(
    frame_idx: int,
    timestamp: str,
    m: MotionResult,
    threshold: float,
    stats: dict,
) -> None:
    """Print structured JSON block for one equipment."""
    data = {
        "frame_id": frame_idx,
        "equipment_id": m.equipment_id,
        "equipment_class": m.equipment_class,
        "timestamp": timestamp,
        "motion_debug": {
            "magnitude": m.magnitude,
            "threshold_used": threshold,
            "status_logic": "mag > threshold",
            "status": m.status,
            "activity": m.activity,
            "motion_source": m.motion_source,
        },
        "time_analytics": {
            "total_time": stats.get("total_time", 0.0),
            "active_time": stats.get("active_time", 0.0),
            "idle_time": stats.get("idle_time", 0.0),
            "loading_time": stats.get("loading_time", 0.0),
            "utilization_percent": round(stats.get("utilization", 0.0) * 100, 1),
            "loading_percent": round(stats.get("loading_ratio", 0.0) * 100, 1),
        },
    }
    print(json.dumps(data, indent=2))


def _print_explanation(
    m: MotionResult,
    threshold: float,
    stats: dict,
    loading_distance: float,
) -> None:
    """Print human-readable explanation for one equipment."""
    print(f"\n[{m.equipment_id} - {m.equipment_class}]\n")
    print(f"- mag = {m.magnitude}")
    print(f"- threshold = {threshold} -> {m.status}")

    if m.equipment_class == "excavator":
        if m.status == "ACTIVE":
            print(f"- activity = {m.activity} (excavator active)")
            print(f"- motion_source = {m.motion_source} "
                  f"({'0 < mag < 0.5' if m.motion_source == 'Arm only' else 'mag >= 0.5'})")
        else:
            print(f"- activity = {m.activity} (excavator inactive)")

    elif m.equipment_class == "dump_truck":
        print(f"\n- nearest excavator:")
        if m.activity == "LOADING":
            print(f"    distance < {loading_distance:.0f}px")
            print(f"    excavator ACTIVE -> TRUE")
            print(f"\n-> activity = LOADING")
        else:
            print(f"    condition not met (excavator inactive or too far)")
            print(f"\n-> activity = WAITING")

    total = stats.get("total_time", 0.0)
    active = stats.get("active_time", 0.0)
    idle = stats.get("idle_time", 0.0)
    loading = stats.get("loading_time", 0.0)
    util_pct = round(stats.get("utilization", 0.0) * 100, 1)
    load_pct = round(stats.get("loading_ratio", 0.0) * 100, 1)

    print(f"\nTime:")
    print(f"- total = {total:.1f}s")
    print(f"- active = {active:.1f}s")
    print(f"- idle = {idle:.1f}s")
    if m.equipment_class == "dump_truck":
        print(f"- loading = {loading:.1f}s")

    print(f"\n-> utilization = {util_pct}%")
    if m.equipment_class == "dump_truck" and loading > 0:
        print(f"-> loading = {load_pct}%")

    print(f"\n{'-'*40}")
