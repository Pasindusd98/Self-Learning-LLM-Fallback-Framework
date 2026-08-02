"""
Instantiates the right StudentModel subclass for a given task, based on
`student_model_type` in that task's YAML config. This is what lets the
rest of the framework stay generalized across task types.
"""
from __future__ import annotations

from cascade.config import StudentModelType, TaskConfig
from cascade.student_models.base import StudentModel
from cascade.student_models.classifier_student import ClassifierStudent

_MODEL_CACHE: dict[str, StudentModel] = {}


def get_student_model(config: TaskConfig) -> StudentModel:
    """Returns a cached student model instance for this task, creating one if needed."""
    if config.task_id in _MODEL_CACHE:
        return _MODEL_CACHE[config.task_id]

    if config.student_model_type == StudentModelType.CLASSIFIER:
        model = ClassifierStudent(task_id=config.task_id)
    elif config.student_model_type == StudentModelType.GENERATIVE:
        # Imported lazily: requires torch/transformers/peft, which are optional
        # extras (`pip install cascade-framework[generative]`).
        from cascade.student_models.generative_student import GenerativeStudent
        model = GenerativeStudent(task_id=config.task_id)
    elif config.student_model_type == StudentModelType.HYBRID:
        raise NotImplementedError(
            "Hybrid student models are task-specific by nature -- "
            "implement a subclass of StudentModel for your rule set and "
            "register it here rather than using the generic registry."
        )
    else:
        raise ValueError(f"Unknown student_model_type: {config.student_model_type}")

    _MODEL_CACHE[config.task_id] = model
    return model


def clear_cache():
    """Mostly useful for tests."""
    _MODEL_CACHE.clear()
