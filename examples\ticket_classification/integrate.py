"""
End-to-end demo of the cascade framework on a single task: support ticket
classification. Run this file directly:

    python examples/ticket_classification/integrate.py

It walks through the full lifecycle:
  1. Shadow mode   -- every ticket goes to the "LLM" (mocked here so the
                       demo runs with no API key), student silently predicts
                       alongside it, both get logged.
  2. Train         -- once enough shadow-mode data exists, fit the student
                       model and calibrator.
  3. Promote       -- stage manager checks accuracy/sample-size/stability
                       and promotes the task to Assisted mode.
  4. Serve         -- new tickets now route through the student model when
                       confident, falling back to the LLM otherwise.

To use a REAL LLM instead of the mock, replace MockLLMProvider with:
    from cascade.llm_providers.llama_ollama_provider import LlamaOllamaProvider
    llm = LlamaOllamaProvider(model=config.llm_fallback_model)
This runs locally via Ollama -- no API key, no per-call cost (install
Ollama, then `ollama pull llama3.2`; see llama_ollama_provider.py).

If you'd rather use a hosted LLM (Anthropic, OpenAI, Together, etc.),
implement a provider following the same LLMProvider interface -- see
cascade/llm_providers/anthropic_provider.py for a template.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cascade.config import TaskConfig
from cascade.interceptor.proxy import CascadeRunner
from cascade.llm_providers.base import LLMProvider
from cascade.router.calibration import ConfidenceCalibrator
from cascade.stages.stage_manager import StageManager
from cascade.storage.db import init_db, session_scope
from cascade.storage.models import Stage
from cascade.student_models.registry import get_student_model
from cascade.training.trainer import run_training_cycle

HERE = Path(__file__).parent


class MockLLMProvider(LLMProvider):
    """
    Stands in for a real third-party LLM for this demo, so it runs with no
    API key and no network calls. It "knows" the right answer from the
    sample data's ground-truth label, simulating what an LLM's response
    would be for each ticket. Swap for AnthropicProvider in real usage.
    """

    def __init__(self, ground_truth: dict[str, str]):
        self.ground_truth = ground_truth

    def complete(self, prompt: str, **kwargs) -> str:
        # In this mock, the prompt IS the ticket text -- look up its label.
        return self.ground_truth.get(prompt, "technical")


def main():
    init_db("sqlite:///./cascade_demo.db")
    config = TaskConfig.from_yaml(Path(__file__).parents[2] / "configs/tasks/ticket_classification.yaml")

    tickets = json.loads((HERE / "sample_data/tickets.json").read_text())
    ground_truth = {t["text"]: t["category"] for t in tickets}
    llm = MockLLMProvider(ground_truth)

    with session_scope("sqlite:///./cascade_demo.db") as session:
        student = get_student_model(config)
        calibrator = ConfidenceCalibrator()
        runner = CascadeRunner(
            config=config,
            student_model=student,
            llm_provider=llm,
            session=session,
            calibrator=calibrator,
            novelty_detector=None,  # skipped in this minimal demo; see docs for wiring one in
        )

        # ---- Phase 1: Shadow mode ----
        # Replay the sample tickets several times (with shuffling) to
        # simulate enough repetitive daily-task volume to train on.
        print("Phase 1: running shadow mode...")
        random.seed(7)
        replayed = tickets * 8   # simulate ~256 historical requests
        random.shuffle(replayed)
        for ticket in replayed:
            runner.run(
                input_data={"text": ticket["text"]},
                prompt_fn=lambda d: d["text"],
            )
        print(f"  -> logged {len(replayed)} shadow-mode requests")

        # ---- Phase 2 & 3: Train + evaluate promotion, repeated across
        # several cycles. Promotion requires `min_stable_cycles` consecutive
        # cycles above threshold (see configs/tasks/ticket_classification.yaml)
        # -- this loop simulates that passage of time/retrains.
        print("Phase 2+3: running multiple train/evaluate cycles...")
        manager = StageManager(session)
        for cycle in range(1, config.promotion.min_stable_cycles + 2):
            accuracy = run_training_cycle(
                session=session,
                config=config,
                student_model=student,
                calibrator=calibrator,
                model_save_dir=str(HERE / "models"),
            )
            new_stage = manager.evaluate_and_update(config, accuracy)
            print(f"  cycle {cycle}: accuracy={accuracy}, stage={new_stage.value}")

        # ---- Phase 4: Serve new requests through the cascade ----
        print("Phase 4: serving new tickets through the cascade...")
        new_tickets = [
            "I got billed twice this month, need one refund",
            "The export button gives a server error every time",
            "Can I invite a coworker to my workspace",
            "Would be great to have a dark theme option",
        ]
        for text in new_tickets:
            result = runner.run(
                input_data={"text": text},
                prompt_fn=lambda d: d["text"],
            )
            print(
                f"  '{text[:45]}...' -> handled_by={result.handler.value}, "
                f"stage={result.stage.value}, confidence={result.student_confidence}, "
                f"output={result.output}"
            )


if __name__ == "__main__":
    main()
