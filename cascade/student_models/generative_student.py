"""
Generative student model: a small pretrained LLM (e.g. Phi-3-mini, Llama
3.2 3B, Gemma 2B) fine-tuned with LoRA on logged (prompt, LLM_output)
pairs. Use this for tasks where the output isn't a fixed set of classes
(summarization, structured drafting, short free-text generation).

Requires the optional "generative" extra:
    pip install -e ".[generative]"

This is intentionally a skeleton, not a full implementation -- generative
fine-tuning setups vary a lot by base model and hardware. Fill in
`_load_base_model` and the LoRA config for your chosen model before use.
Start with the ClassifierStudent for your first end-to-end task; come
back to this once that loop is proven out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cascade.student_models.base import Prediction, StudentModel


class GenerativeStudent(StudentModel):
    def __init__(self, task_id: str, base_model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.task_id = task_id
        self.base_model_name = base_model_name
        self._model = None
        self._tokenizer = None
        self._is_trained = False

    def _ensure_deps(self):
        try:
            import torch, transformers, peft  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Generative student models require extra dependencies. "
                "Install with: pip install -e \".[generative]\""
            ) from e

    def predict(self, input_data: dict) -> Prediction:
        if not self._is_trained:
            return Prediction(output=None, raw_confidence=0.0, is_trained=False)

        self._ensure_deps()
        import torch

        prompt = input_data.get("prompt", "")
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=256, output_scores=True,
                return_dict_in_generate=True,
            )
        text = self._tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Mean token probability as a rough confidence proxy. Consider
        # replacing with a proper calibration pass (see router/calibration.py)
        # once you have enough held-out data to fit a temperature parameter.
        scores = outputs.scores
        confidence = 0.5
        if scores:
            probs = [torch.softmax(s, dim=-1).max().item() for s in scores]
            confidence = float(sum(probs) / len(probs))

        return Prediction(output=text, raw_confidence=confidence, is_trained=True)

    def fit(self, examples: list[dict], labels: list[Any]) -> None:
        self._ensure_deps()
        raise NotImplementedError(
            "Fill in your LoRA fine-tuning loop here using peft + trl's "
            "SFTTrainer. Kept as a stub because the right config depends on "
            "your chosen base model and hardware. See docs/architecture.md "
            "for the recommended approach."
        )

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self._ensure_deps()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._model = AutoModelForCausalLM.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._is_trained = True
