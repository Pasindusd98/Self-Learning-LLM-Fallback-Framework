"""
The interceptor: the single entry point a developer's system calls
instead of hitting the LLM directly. This is what makes the framework
"plug and play" -- swap your existing `llm.complete(prompt)` call for
`cascade.run(task_id, input_data, prompt_fn)` and everything else
(logging, routing, staging) happens automatically.

Per-request flow:
  1. Load task config + current stage
  2. SHADOW    -> always call LLM; also run student silently, log both, don't
                  block on or expose the student's output.
  3. ASSISTED  -> run student; if confidence >= assisted threshold, serve
                  student output; else call LLM.
  4. AUTONOMOUS-> run student; if confidence >= autonomous threshold, serve
                  student output; else call LLM (rare fallback).
  Every path logs a RequestLog row so the training pipeline and stage
  manager have data to work with.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from cascade.config import TaskConfig
from cascade.llm_providers.base import LLMProvider
from cascade.router.calibration import ConfidenceCalibrator
from cascade.router.confidence import ConfidenceRouter
from cascade.router.novelty_detector import NoveltyDetector
from cascade.stages.stage_manager import StageManager
from cascade.storage.models import Handler, RequestLog, Stage
from cascade.student_models.base import StudentModel


@dataclass
class CascadeResult:
    output: Any
    handler: Handler
    stage: Stage
    student_confidence: Optional[float]
    request_log_id: int


class CascadeRunner:
    """
    One instance per task (or share across tasks -- it's stateless aside
    from the injected dependencies). Construct once at app startup, reuse
    across requests.
    """

    def __init__(
        self,
        config: TaskConfig,
        student_model: StudentModel,
        llm_provider: LLMProvider,
        session: Session,
        calibrator: Optional[ConfidenceCalibrator] = None,
        novelty_detector: Optional[NoveltyDetector] = None,
    ):
        self.config = config
        self.student_model = student_model
        self.llm_provider = llm_provider
        self.session = session
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.novelty_detector = novelty_detector
        self.router = ConfidenceRouter(self.calibrator, self.novelty_detector)
        self.stage_manager = StageManager(session)

    def run(
        self,
        input_data: dict,
        prompt_fn: Callable[[dict], str],
        text_field_for_novelty: str = "text",
    ) -> CascadeResult:
        """
        input_data:   dict of task inputs, matching config.input_fields
        prompt_fn:    function that builds the LLM prompt from input_data,
                      used only when the LLM path is actually taken
        """
        stage = self.stage_manager.current_stage(self.config.task_id)
        prediction = self.student_model.predict(input_data)
        input_text = input_data.get(text_field_for_novelty)

        if stage == Stage.SHADOW:
            llm_output = self.llm_provider.complete(prompt_fn(input_data))
            handler = Handler.LLM
            served_output = llm_output
        else:
            threshold = (
                self.config.thresholds.assisted_mode
                if stage == Stage.ASSISTED
                else self.config.thresholds.autonomous_mode
            )
            decision = self.router.decide(
                task_id=self.config.task_id,
                prediction=prediction,
                input_text=input_text if self.config.novelty_check_enabled else None,
                threshold=threshold,
            )
            if decision.should_use_student:
                handler = Handler.STUDENT
                served_output = prediction.output
                llm_output = None
            else:
                llm_output = self.llm_provider.complete(prompt_fn(input_data))
                handler = Handler.LLM
                served_output = llm_output

        log = RequestLog(
            task_id=self.config.task_id,
            input_payload=json.dumps(input_data),
            student_output=json.dumps(prediction.output) if prediction.is_trained else None,
            student_confidence=prediction.raw_confidence if prediction.is_trained else None,
            llm_output=json.dumps(llm_output) if llm_output is not None else None,
            handler=handler,
            stage_at_request_time=stage,
        )
        self.session.add(log)
        self.session.commit()

        if self.novelty_detector is not None and input_text is not None:
            self.novelty_detector.record(self.config.task_id, input_text)

        return CascadeResult(
            output=served_output,
            handler=handler,
            stage=stage,
            student_confidence=prediction.raw_confidence if prediction.is_trained else None,
            request_log_id=log.id,
        )

    def record_feedback(self, request_log_id: int, was_correct: bool) -> None:
        """
        Call this when you learn (via user feedback, downstream validation,
        or human review) whether a served response was actually correct.
        This is what powers accuracy tracking for stage promotion/demotion
        and calibration fitting.
        """
        import datetime as dt
        log = self.session.get(RequestLog, request_log_id)
        if log is None:
            raise ValueError(f"No RequestLog with id {request_log_id}")
        log.was_correct = was_correct
        log.verified_at = dt.datetime.utcnow()
        self.session.commit()
