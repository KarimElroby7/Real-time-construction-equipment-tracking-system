"""Eagle Vision — Streamlit Analytics Dashboard.

Displays live equipment status from FastAPI and final analytics
from the pipeline JSON output. Video is shown in a separate
OpenCV window — this UI is analytics-only.

Usage:
    streamlit run ui/app.py
"""

import json
import os
import time

import requests
import streamlit as st

API_URL = "http://localhost:8000/state"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYTICS_PATH = os.path.join(ROOT_DIR, "output", "analytics.json")

st.set_page_config(page_title="Eagle Vision", page_icon="🚜", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
.analytics-card {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.analytics-card h3 {
    margin: 0 0 12px 0;
    color: #e0e0e0;
    font-size: 1.2rem;
}
.util-big {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 4px 0;
}
.util-green  { color: #4caf50; }
.util-orange { color: #ff9800; }
.util-red    { color: #f44336; }
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #2a2a3a;
    font-size: 0.95rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-label { color: #aaa; }
.metric-val-blue   { color: #42a5f5; font-weight: 600; }
.metric-val-red    { color: #ef5350; font-weight: 600; }
.metric-val-yellow { color: #ffca28; font-weight: 600; }
.metric-val-white  { color: #e0e0e0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🚜 Eagle Vision Dashboard")
st.caption("Real-time construction equipment monitoring — video in OpenCV window")


# ── Helpers ─────────────────────────────────────────────────

def _util_color_class(util: float) -> str:
    if util > 0.70:
        return "util-green"
    if util >= 0.40:
        return "util-orange"
    return "util-red"


def _format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    if m > 0:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.1f}s"


def render_analytics(summary: dict):
    """Render the final analytics cards."""
    st.subheader("📊 Final Analytics")

    ids = list(summary.keys())
    cols = st.columns(min(len(ids), 2))

    for i, eid in enumerate(ids):
        stats = summary[eid]
        eq_class = stats.get("equipment_class", "unknown")
        icon = "⛏️" if eq_class == "excavator" else "🚛"
        util = stats.get("utilization", 0.0)
        util_pct = f"{util * 100:.1f}%"
        util_cls = _util_color_class(util)

        active = stats.get("active_time", 0.0)
        idle = stats.get("idle_time", 0.0)
        loading = stats.get("loading_time", 0.0)
        total = stats.get("total_time", 0.0)
        loading_ratio = stats.get("loading_ratio", 0.0)

        # Dump truck: "Moving Time"  |  Excavator: "Active Time"
        active_label = "Moving Time" if eq_class == "dump_truck" else "Active Time"

        card_html = f"""
        <div class="analytics-card">
            <h3>{icon} {eid} <span style="color:#777;font-size:0.85rem;">({eq_class})</span></h3>
            <div style="text-align:center; margin:8px 0 16px;">
                <div style="color:#888;font-size:0.85rem;">UTILIZATION</div>
                <div class="util-big {util_cls}">{util_pct}</div>
            </div>
            <div class="metric-row">
                <span class="metric-label">Total Tracked</span>
                <span class="metric-val-white">{_format_time(total)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">🔵 {active_label}</span>
                <span class="metric-val-blue">{_format_time(active)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">🔴 Idle Time</span>
                <span class="metric-val-red">{_format_time(idle)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">🟡 Loading Time</span>
                <span class="metric-val-yellow">{_format_time(loading)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Loading Ratio</span>
                <span class="metric-val-white">{loading_ratio * 100:.1f}%</span>
            </div>
        </div>
        """

        with cols[i % 2]:
            st.markdown(card_html, unsafe_allow_html=True)
            st.progress(min(util, 1.0))


# ════════════════════════════════════════════════════════════
#  SECTION 2 — Live Equipment Status
# ════════════════════════════════════════════════════════════

equipment = []
try:
    resp = requests.get(API_URL, timeout=2)
    data = resp.json()
    equipment = data.get("equipment", [])
except Exception:
    st.caption("Waiting for API server...")

if equipment:
    eq_cols = st.columns(len(equipment))

    for col, eq in zip(eq_cols, equipment):
        eid = eq.get("equipment_id", "---")
        eq_class = eq.get("equipment_class", "---")
        eq_status = eq.get("status", "---")
        activity = eq.get("activity", "---")
        magnitude = eq.get("magnitude", 0.0)
        motion_src = eq.get("motion_source")
        confidence = eq.get("confidence", 0.0)
        frame_id = eq.get("frame_id", 0)

        is_active = eq_status in ("ACTIVE", "MOVE")
        status_color = "🟢" if is_active else "🔴"
        class_icon = "⛏️" if eq_class == "excavator" else "🚛"

        with col:
            st.subheader(f"{class_icon} {eid}")
            st.markdown(f"**Class:** {eq_class}")
            st.markdown(f"**Status:** {status_color} {eq_status}")
            st.markdown(f"**Activity:** {activity}")
            st.markdown(f"**Magnitude:** {magnitude:.2f}")

            if motion_src:
                st.markdown(f"**Motion:** {motion_src}")

            st.markdown(f"**Confidence:** {confidence:.1%}")
            st.caption(f"Frame: {frame_id}")
            st.divider()

    st.subheader("📊 Summary Table")
    table_data = []
    for eq in equipment:
        row = {
            "ID": eq.get("equipment_id"),
            "Class": eq.get("equipment_class"),
            "Status": eq.get("status"),
            "Activity": eq.get("activity"),
            "Magnitude": eq.get("magnitude", 0.0),
            "Confidence": f"{eq.get('confidence', 0.0):.1%}",
        }
        table_data.append(row)
    st.table(table_data)
else:
    st.info("Waiting for equipment data...")

# ════════════════════════════════════════════════════════════
#  SECTION 3 — Final Analytics (auto-display, no button)
# ════════════════════════════════════════════════════════════

st.divider()


def read_analytics_safe(path, retries=5, delay=0.2):
    """Read analytics JSON with retry logic to handle partial writes."""
    for _ in range(retries):
        try:
            if not os.path.exists(path):
                return {"status": "processing"}

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                time.sleep(delay)
                continue

            if not content.strip().endswith("}"):
                time.sleep(delay)
                continue

            data = json.loads(content)

            if "status" not in data:
                time.sleep(delay)
                continue

            return data

        except (json.JSONDecodeError, OSError):
            time.sleep(delay)

    return {"status": "processing"}


analytics_data = read_analytics_safe(ANALYTICS_PATH)
status = analytics_data.get("status", "processing")

if status == "completed":
    st.success("Processing completed!")

    summary = analytics_data.get("summary", {})
    if summary:
        render_analytics(summary)
    else:
        st.info("Analytics file is empty. Run the pipeline first.")

else:
    st.warning("Processing is still running. Please wait...")
    time.sleep(1)
    st.rerun()
