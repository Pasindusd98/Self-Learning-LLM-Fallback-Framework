import pytest

from cascade.student_models.classifier_student import ClassifierStudent


def test_predict_before_training_returns_untrained():
    model = ClassifierStudent(task_id="t1")
    pred = model.predict({"text": "hello"})
    assert pred.is_trained is False


def test_fit_requires_minimum_examples():
    model = ClassifierStudent(task_id="t1")
    with pytest.raises(ValueError):
        model.fit([{"text": "a"}], ["x"])


def test_fit_and_predict_roundtrip():
    model = ClassifierStudent(task_id="t1")
    examples = [
        {"text": "refund my payment"}, {"text": "invoice is wrong"},
        {"text": "charged twice"}, {"text": "update my card"},
        {"text": "app crashes on launch"}, {"text": "cannot log in"},
        {"text": "page is blank"}, {"text": "upload fails"},
        {"text": "refund please"}, {"text": "billing issue"},
        {"text": "server error"}, {"text": "login broken"},
    ]
    labels = ["billing"] * 5 + ["technical"] * 5 + ["billing", "technical"]
    model.fit(examples, labels)

    pred = model.predict({"text": "refund my subscription"})
    assert pred.is_trained is True
    assert pred.output in ("billing", "technical")
    assert 0.0 <= pred.raw_confidence <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    model = ClassifierStudent(task_id="t1")
    examples = [{"text": f"example {i}"} for i in range(12)]
    labels = ["a"] * 6 + ["b"] * 6
    model.fit(examples, labels)

    path = tmp_path / "model.joblib"
    model.save(str(path))

    reloaded = ClassifierStudent(task_id="t1")
    reloaded.load(str(path))
    pred = reloaded.predict({"text": "example 1"})
    assert pred.is_trained is True
