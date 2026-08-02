"""
Loads and validates per-task YAML configs. This file is the entire
"plug-and-play" surface of the framework: a developer integrating a new
task writes one YAML file, and everything else (routing, staging,
training) reads from it.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class StudentModelType(str, Enum):
    CLASSIFIER = "classifier"       # sklearn / small transformer classifier
    GENERATIVE = "generative"       # LoRA fine-tuned small LLM
    HYBRID = "hybrid"               # rule-based pre-filter + classifier


class Thresholds(BaseModel):
    """Confidence thresholds that gate stage promotion and per-request routing."""
    assisted_mode: float = Field(0.85, ge=0.0, le=1.0)
    autonomous_mode: float = Field(0.95, ge=0.0, le=1.0)

    @field_validator("autonomous_mode")
    @classmethod
    def autonomous_must_exceed_assisted(cls, v, info):
        assisted = info.data.get("assisted_mode")
        if assisted is not None and v < assisted:
            raise ValueError("autonomous_mode threshold must be >= assisted_mode threshold")
        return v


class PromotionRules(BaseModel):
    """Guards against promoting a task on too little or too unstable data."""
    min_samples_before_promotion: int = Field(500, gt=0)
    min_stable_cycles: int = Field(3, gt=0)          # consecutive retrain cycles above threshold
    demotion_enabled: bool = True
    demotion_accuracy_floor: float = Field(0.80, ge=0.0, le=1.0)
    demotion_window_size: int = Field(200, gt=0)      # rolling window of recent requests to check


class TaskConfig(BaseModel):
    task_id: str
    description: str = ""
    student_model_type: StudentModelType
    thresholds: Thresholds = Thresholds()
    promotion: PromotionRules = PromotionRules()
    retrain_schedule: str = "daily"                  # "daily" | "hourly" | "manual"
    llm_fallback_provider: str = "anthropic"
    llm_fallback_model: str = "claude-sonnet-4-6"
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    novelty_check_enabled: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


class FrameworkConfig(BaseModel):
    database_url: str = "sqlite:///./cascade.db"
    vector_store_backend: str = "local"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FrameworkConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


def load_all_task_configs(tasks_dir: str | Path) -> dict[str, TaskConfig]:
    """Load every *.yaml file in configs/tasks/ into a {task_id: TaskConfig} dict."""
    tasks_dir = Path(tasks_dir)
    configs = {}
    for path in tasks_dir.glob("*.yaml"):
        cfg = TaskConfig.from_yaml(path)
        configs[cfg.task_id] = cfg
    return configs
