from cascade.config import PromotionRules, Thresholds
from cascade.stages.promotion_rules import check_demotion, check_promotion
from cascade.storage.models import Stage


def make_thresholds():
    return Thresholds(assisted_mode=0.85, autonomous_mode=0.95)


def make_rules(**overrides):
    defaults = dict(
        min_samples_before_promotion=100,
        min_stable_cycles=3,
        demotion_enabled=True,
        demotion_accuracy_floor=0.80,
        demotion_window_size=100,
    )
    defaults.update(overrides)
    return PromotionRules(**defaults)


def test_no_promotion_without_enough_samples():
    result = check_promotion(
        current_stage=Stage.SHADOW,
        recent_accuracy=0.99,
        samples_seen=10,
        consecutive_stable_cycles=5,
        rules=make_rules(),
        thresholds=make_thresholds(),
    )
    assert result == Stage.SHADOW


def test_no_promotion_without_stable_cycles():
    result = check_promotion(
        current_stage=Stage.SHADOW,
        recent_accuracy=0.99,
        samples_seen=500,
        consecutive_stable_cycles=1,
        rules=make_rules(),
        thresholds=make_thresholds(),
    )
    assert result == Stage.SHADOW


def test_promotion_shadow_to_assisted():
    result = check_promotion(
        current_stage=Stage.SHADOW,
        recent_accuracy=0.90,
        samples_seen=500,
        consecutive_stable_cycles=3,
        rules=make_rules(),
        thresholds=make_thresholds(),
    )
    assert result == Stage.ASSISTED


def test_promotion_only_moves_one_stage_at_a_time():
    """Even with accuracy high enough for autonomous, shadow should only
    move to assisted, never skip straight to autonomous."""
    result = check_promotion(
        current_stage=Stage.SHADOW,
        recent_accuracy=0.99,
        samples_seen=500,
        consecutive_stable_cycles=3,
        rules=make_rules(),
        thresholds=make_thresholds(),
    )
    assert result == Stage.ASSISTED


def test_assisted_to_autonomous_requires_higher_threshold():
    result = check_promotion(
        current_stage=Stage.ASSISTED,
        recent_accuracy=0.90,  # above assisted threshold but below autonomous
        samples_seen=500,
        consecutive_stable_cycles=3,
        rules=make_rules(),
        thresholds=make_thresholds(),
    )
    assert result == Stage.ASSISTED

    result = check_promotion(
        current_stage=Stage.ASSISTED,
        recent_accuracy=0.97,
        samples_seen=500,
        consecutive_stable_cycles=3,
        rules=make_rules(),
        thresholds=make_thresholds(),
    )
    assert result == Stage.AUTONOMOUS


def test_demotion_on_accuracy_drop():
    result = check_demotion(
        current_stage=Stage.AUTONOMOUS,
        rolling_window_accuracy=0.60,
        rules=make_rules(),
    )
    assert result == Stage.ASSISTED


def test_no_demotion_when_healthy():
    result = check_demotion(
        current_stage=Stage.AUTONOMOUS,
        rolling_window_accuracy=0.96,
        rules=make_rules(),
    )
    assert result == Stage.AUTONOMOUS


def test_demotion_disabled_does_nothing():
    result = check_demotion(
        current_stage=Stage.AUTONOMOUS,
        rolling_window_accuracy=0.10,
        rules=make_rules(demotion_enabled=False),
    )
    assert result == Stage.AUTONOMOUS
