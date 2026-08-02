# Adding a new task

This is the whole "plug-and-play" workflow. Five steps.

## 1. Write a task config

Copy `configs/tasks/ticket_classification.yaml` to a new file, e.g.
`configs/tasks/invoice_extraction.yaml`, and fill in:

```yaml
task_id: "invoice_extraction"
description: "Extract vendor, amount, and due date from invoice text."

student_model_type: "classifier"   # or "generative" for open-ended output

thresholds:
  assisted_mode: 0.85
  autonomous_mode: 0.95

promotion:
  min_samples_before_promotion: 500
  min_stable_cycles: 3
  demotion_enabled: true
  demotion_accuracy_floor: 0.80
  demotion_window_size: 200

retrain_schedule: "daily"
llm_fallback_provider: "anthropic"
llm_fallback_model: "claude-sonnet-4-6"

input_fields: ["text"]
output_fields: ["vendor", "amount", "due_date"]
novelty_check_enabled: true
```

`task_id` must be unique across all configs — it's the key used
everywhere (logs, models, stage state).

## 2. Wire the interceptor into your existing call site

Wherever your system currently does something like:

```python
response = anthropic_client.messages.create(...)
```

Replace it with:

```python
from cascade.config import TaskConfig
from cascade.interceptor.proxy import CascadeRunner
from cascade.llm_providers.llama_ollama_provider import LlamaOllamaProvider
from cascade.router.calibration import ConfidenceCalibrator
from cascade.storage.db import session_scope
from cascade.student_models.registry import get_student_model

config = TaskConfig.from_yaml("configs/tasks/invoice_extraction.yaml")

with session_scope() as session:
    runner = CascadeRunner(
        config=config,
        student_model=get_student_model(config),
        llm_provider=LlamaOllamaProvider(model=config.llm_fallback_model),
        session=session,
        calibrator=ConfidenceCalibrator(),
    )

    result = runner.run(
        input_data={"text": invoice_text},
        prompt_fn=lambda d: f"Extract vendor, amount, due date from:\n{d['text']}",
    )
    print(result.output, result.handler, result.stage)
```

This uses `LlamaOllamaProvider`, which calls a local Ollama server — no
API key, no per-call cost. Install Ollama (https://ollama.com/download)
and run `ollama pull llama3.2` once beforehand. If you'd rather use a
hosted model, swap in `AnthropicProvider` or write a new provider
following the same `LLMProvider` interface.

## 3. Let shadow mode run

Do nothing for a while. Every request goes to the LLM as normal (so
production behavior is unchanged), and the student model silently learns
alongside it. Check progress any time:

```bash
python scripts/run_shadow_mode.py
```

## 4. Schedule retraining

Add a cron job (or Airflow DAG, or scheduled GitHub Action) matching the
`retrain_schedule` you set in the config:

```
0 2 * * *  cd /path/to/repo && python -m cascade.training.scheduler
```

This trains the student, refits calibration, evaluates accuracy, and
checks promotion/demotion — for every configured task in one run.

## 5. Watch it graduate

Once `min_samples_before_promotion` and `min_stable_cycles` are both
satisfied, the task auto-promotes to Assisted, then eventually
Autonomous. You'll see the LLM call volume for that task drop
correspondingly. No code changes needed on your end — the same
`runner.run(...)` call now serves most responses locally.

## Choosing `student_model_type`

| Your task looks like... | Use |
|---|---|
| Fixed set of output categories/labels | `classifier` |
| Extracting a few structured fields from text | `classifier` (one per field) or a custom hybrid |
| Free-text generation, summarization, drafting | `generative` (requires implementing the LoRA fine-tune loop — see `cascade/student_models/generative_student.py`) |
| Task-specific rules exist that should always apply first | `hybrid` (implement a `StudentModel` subclass combining rules + a fallback classifier) |

## Common mistakes to avoid

- **Setting thresholds too low to "see results faster."** A low
  `assisted_mode` threshold means the student serves answers it's
  genuinely unsure about. Start conservative (0.85+) and only lower it
  if calibration data shows the model is more reliable than that.
- **Skipping the novelty check to save an embedding call.** This is what
  catches inputs the student was never trained on — disabling it
  (`novelty_check_enabled: false`) is a legitimate performance
  optimization once you trust a task's stability, but not something to
  do from day one.
- **Not recording feedback.** If your system has any way to learn
  whether a served answer was actually correct (user reports it, a
  downstream check fails, human review), call `runner.record_feedback(
  request_log_id, was_correct)`. Without this, demotion has nothing to
  check against outside of shadow-mode agreement, which is a weaker signal.
