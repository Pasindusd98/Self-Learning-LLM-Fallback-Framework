"""
Minimal Streamlit dashboard showing per-task stage, accuracy, and
estimated LLM calls saved. Optional -- the framework works fully without
this, but it's useful for a quick visual check.

Run with:
    pip install streamlit
    streamlit run dashboard/app.py
"""
import streamlit as st

from cascade.config import load_all_task_configs
from cascade.monitoring.metrics import task_summary
from cascade.stages.stage_manager import StageManager
from cascade.storage.db import session_scope

st.set_page_config(page_title="Cascade Framework Dashboard", layout="wide")
st.title("Cascade Framework — Task Overview")

configs = load_all_task_configs("configs/tasks")

if not configs:
    st.warning("No task configs found in configs/tasks/. Add one to get started.")
else:
    with session_scope() as session:
        manager = StageManager(session)

        for task_id, config in configs.items():
            st.subheader(task_id)
            stage = manager.current_stage(task_id)
            summary = task_summary(session, task_id)

            cols = st.columns(4)
            cols[0].metric("Stage", stage.value.capitalize())
            cols[1].metric("Total requests", summary["total_requests"])
            cols[2].metric("Served by student", f"{summary['student_share']:.1%}")
            fallback = summary["recent_fallback_rate"]
            cols[3].metric(
                "Recent fallback rate",
                f"{fallback:.1%}" if fallback is not None else "n/a",
            )

            state = manager.get_or_create_state(task_id)
            st.caption(
                f"Samples seen: {state.samples_seen} | "
                f"Recent accuracy: {state.recent_accuracy} | "
                f"Stable cycles: {state.consecutive_stable_cycles} | "
                f"Last retrain: {state.last_retrain_at}"
            )
            st.divider()
