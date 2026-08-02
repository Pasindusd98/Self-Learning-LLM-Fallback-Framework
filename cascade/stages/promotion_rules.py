"""
Pure decision logic for stage transitions. Deliberately has zero ML in
it -- just threshold/sample-size/stability checks against metrics that
stage_manager.py maintains. Kept separate and dependency-free so it's easy
to unit test and easy to audit (governance-relevant: you want to be able
to show exactly why a task was promoted).
"""
from __future__ import annotations

from cascade.config import PromotionRules
from cascade.storage.models import Stage


def check_promotion(
    current_stage: Stage,
    recent_accuracy: float | None,
    samples_seen: int,
    consecutive_stable_cycles: int,
    rules: PromotionRules,
    thresholds,  # cascade.config.Thresholds
) -> Stage:
    """
    Returns the stage the task SHOULD be in, given current metrics.
    Only ever moves one stage at a time, even if metrics would justify
    skipping a stage -- promotion should be earned incrementally.
    """
    if recent_accuracy is None or samples_seen < rules.min_samples_before_promotion:
        return current_stage

    if current_stage == Stage.SHADOW:
        if (
            recent_accuracy >= thresholds.assisted_mode
            and consecutive_stable_cycles >= rules.min_stable_cycles
        ):
            return Stage.ASSISTED
        return Stage.SHADOW

    if current_stage == Stage.ASSISTED:
        if (
            recent_accuracy >= thresholds.autonomous_mode
            and consecutive_stable_cycles >= rules.min_stable_cycles
        ):
            return Stage.AUTONOMOUS
        return Stage.ASSISTED

    return Stage.AUTONOMOUS  # already at the top


def check_demotion(
    current_stage: Stage,
    rolling_window_accuracy: float,
    rules: PromotionRules,
) -> Stage:
    """
    Checks recent (windowed) accuracy against a floor -- this is the drift
    guard. Demotes by exactly one stage if accuracy has dropped below the
    floor, regardless of how good the historical average looks. Silent
    degradation is the main real-world failure mode for cascades like this.
    """
    if not rules.demotion_enabled:
        return current_stage

    if rolling_window_accuracy >= rules.demotion_accuracy_floor:
        return current_stage

    if current_stage == Stage.AUTONOMOUS:
        return Stage.ASSISTED
    if current_stage == Stage.ASSISTED:
        return Stage.SHADOW
    return Stage.SHADOW
