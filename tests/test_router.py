from cascade.router.calibration import ConfidenceCalibrator
from cascade.router.confidence import ConfidenceRouter
from cascade.student_models.base import Prediction


def test_untrained_student_always_escalates():
    router = ConfidenceRouter(ConfidenceCalibrator())
    decision = router.decide(
        task_id="t1",
        prediction=Prediction(output=None, raw_confidence=0.0, is_trained=False),
        threshold=0.85,
    )
    assert decision.should_use_student is False


def test_high_confidence_uses_student_once_calibrated():
    # Before the calibrator is fit, it applies a conservative haircut
    # (raw * 0.8) by design -- so we fit it first with data showing this
    # model's confidence is trustworthy, then check routing.
    calibrator = ConfidenceCalibrator()
    calibrator.fit(raw_confidences=[0.97] * 20, was_correct=[True] * 19 + [False])
    router = ConfidenceRouter(calibrator)
    decision = router.decide(
        task_id="t1",
        prediction=Prediction(output="billing", raw_confidence=0.97, is_trained=True),
        threshold=0.85,
    )
    assert decision.should_use_student is True


def test_unfitted_calibrator_applies_conservative_haircut():
    router = ConfidenceRouter(ConfidenceCalibrator())
    decision = router.decide(
        task_id="t1",
        prediction=Prediction(output="billing", raw_confidence=0.97, is_trained=True),
        threshold=0.85,
    )
    # 0.97 * 0.8 = 0.776, below the 0.85 threshold -- escalates to LLM
    # rather than trusting an uncalibrated confidence score.
    assert decision.should_use_student is False


def test_low_confidence_escalates():
    router = ConfidenceRouter(ConfidenceCalibrator())
    decision = router.decide(
        task_id="t1",
        prediction=Prediction(output="billing", raw_confidence=0.40, is_trained=True),
        threshold=0.85,
    )
    assert decision.should_use_student is False


def test_calibrator_fits_and_adjusts_confidence():
    calibrator = ConfidenceCalibrator()
    # Simulate a model that's systematically overconfident: raw 0.9 confidence
    # but only actually correct about half the time.
    raw = [0.9] * 15 + [0.5] * 15
    correct = [True] * 8 + [False] * 7 + [True] * 5 + [False] * 10
    calibrator.fit(raw, correct)
    calibrated = calibrator.calibrate(0.9)
    assert 0.0 <= calibrated <= 1.0
