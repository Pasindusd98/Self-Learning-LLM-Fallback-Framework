"""
Prints current shadow/assisted/autonomous status and key metrics for every
configured task. Useful as a quick CLI health check, or to run periodically
in CI/monitoring.

Usage:
    python scripts/run_shadow_mode.py
"""
from cascade.config import load_all_task_configs
from cascade.monitoring.metrics import task_summary
from cascade.stages.stage_manager import StageManager
from cascade.storage.db import session_scope


def main():
    configs = load_all_task_configs("configs/tasks")
    if not configs:
        print("No task configs found in configs/tasks/")
        return

    with session_scope() as session:
        manager = StageManager(session)
        for task_id, config in configs.items():
            stage = manager.current_stage(task_id)
            summary = task_summary(session, task_id)
            print(f"\nTask: {task_id}")
            print(f"  Stage: {stage.value}")
            print(f"  Total requests logged: {summary['total_requests']}")
            print(f"  Served by student: {summary['served_by_student']} "
                  f"({summary['student_share']:.1%})")
            print(f"  Recent fallback rate: {summary['recent_fallback_rate']}")


if __name__ == "__main__":
    main()
