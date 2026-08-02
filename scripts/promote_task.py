"""
Manual override to force a task's stage. Use sparingly -- this bypasses
the safety checks in promotion_rules.py, so only use it for debugging or
a deliberate, reviewed rollback/rollforward, not routine operation.

Usage:
    python scripts/promote_task.py ticket_classification assisted
"""
import sys

from cascade.storage.db import session_scope
from cascade.storage.models import Stage
from cascade.stages.stage_manager import StageManager


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/promote_task.py <task_id> <stage>")
        print("  stage must be one of: shadow, assisted, autonomous")
        sys.exit(1)

    task_id, stage_str = sys.argv[1], sys.argv[2]
    try:
        stage = Stage(stage_str)
    except ValueError:
        print(f"Invalid stage '{stage_str}'. Must be one of: shadow, assisted, autonomous")
        sys.exit(1)

    with session_scope() as session:
        manager = StageManager(session)
        state = manager.get_or_create_state(task_id)
        old_stage = state.current_stage
        state.current_stage = stage
        session.commit()
        print(f"Task '{task_id}': {old_stage.value} -> {stage.value} (manual override)")


if __name__ == "__main__":
    main()
