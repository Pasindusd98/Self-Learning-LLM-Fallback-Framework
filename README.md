# Cascade Framework

Confidence-gated model cascading for repetitive AI tasks: learn what a
third-party LLM does on a specific, recurring task, and stop paying for
it once a small local model can replicate it reliably. The LLM stays in
the loop as a fallback for whatever the local model isn't confident
about.

```
your system --> CascadeRunner --> [confident?] --> local student model
                                        |
                                    [not confident]
                                        |
                                        v
                                   third-party LLM (fallback)
```

## Why

Most AI-integrated systems call the same LLM, for the same category of
task, over and over — classify this ticket, extract these fields, route
this request. That traffic is mostly repetitive and mostly cheap to
predict once you've seen enough of it. This framework automates the
distillation loop: log what the LLM does, train a local model on it, and
gradually shift traffic to the local model as confidence in it grows —
never all at once, and never without a safety net.

## How it works (short version)

Every task goes through three stages, promoted automatically based on
measured accuracy, sample size, and stability — never on a single lucky
result:

1. **Shadow** — LLM handles everything; local model watches and learns silently.
2. **Assisted** — local model answers when confident; LLM only for the uncertain cases.
3. **Autonomous** — local model handles almost everything; LLM is a rare safety net.

A task can also be **demoted** automatically if its accuracy drifts below
a safety floor — this isn't a one-way ratchet.

Full details: [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
git clone <this-repo>
cd cascade-framework
pip install -e .
cp .env.example .env

# optional but recommended: install Ollama for a real local fallback LLM
# https://ollama.com/download, then:
ollama pull llama3.2

# run the full end-to-end demo (uses a mock LLM, no Ollama needed for this)
python examples/ticket_classification/integrate.py
```

You should see output walking through shadow mode, training, stage
promotion, and live serving of new requests — the whole lifecycle in one
run, in about a second.

Run the tests:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Adding your own task

See [`docs/adding_a_new_task.md`](docs/adding_a_new_task.md) for the full
walkthrough. Short version: write one YAML config, wrap your existing
LLM call with `CascadeRunner.run(...)`, let it collect data, and schedule
`python -m cascade.training.scheduler` to run periodically (cron/Airflow).

```python
from cascade.config import TaskConfig
from cascade.interceptor.proxy import CascadeRunner
from cascade.llm_providers.llama_ollama_provider import LlamaOllamaProvider
from cascade.router.calibration import ConfidenceCalibrator
from cascade.storage.db import session_scope
from cascade.student_models.registry import get_student_model

config = TaskConfig.from_yaml("configs/tasks/your_task.yaml")

with session_scope() as session:
    runner = CascadeRunner(
        config=config,
        student_model=get_student_model(config),
        llm_provider=LlamaOllamaProvider(model=config.llm_fallback_model),
        session=session,
        calibrator=ConfidenceCalibrator(),
    )
    result = runner.run(
        input_data={"text": some_input},
        prompt_fn=lambda d: f"Your prompt using {d['text']}",
    )
```

Prefer a hosted LLM (Anthropic, OpenAI, Together, etc.) instead of local
Ollama? Swap in `cascade.llm_providers.anthropic_provider.AnthropicProvider`
or implement a new provider following the same `LLMProvider` interface.

## Repo layout

```
cascade/
├── interceptor/     # drop-in proxy your system calls instead of the LLM directly
├── router/           # confidence combination, calibration, novelty detection
├── student_models/   # classifier + generative student model implementations
├── stages/            # shadow/assisted/autonomous lifecycle + promotion rules
├── training/           # data loading, retraining, scheduled retrain entrypoint
├── storage/             # DB models, session management, vector store
├── llm_providers/         # fallback LLM abstraction (Anthropic implemented)
└── monitoring/              # metrics + drift detection for dashboards/alerting

configs/tasks/*.yaml   # one file per task -- the whole plug-and-play surface
examples/               # worked end-to-end example (ticket classification)
dashboard/               # optional Streamlit monitoring UI
docs/                     # architecture, task onboarding, confidence scoring deep-dive
tests/                     # unit tests for router, stages, student models
```

## Instructions to follow when extending this

- **Start narrow.** Get one task fully through Shadow → Assisted →
  Autonomous before generalizing to a second. The example task
  (`ticket_classification`) is a working template for this.
- **Don't lower thresholds to see results faster.** Thresholds exist to
  make wrong answers rare, not to hit a promotion milestone. Let
  calibration data justify a lower threshold, not impatience.
- **Wire in `record_feedback()` wherever you can.** Shadow-mode agreement
  is a reasonable bootstrap signal, but real feedback (user reports,
  downstream validation, human review) is what keeps calibration honest
  over time — especially important once a task reaches Autonomous mode.
- **Keep the demotion guard enabled.** It's the main protection against
  silent degradation after a task is promoted. Only disable it
  deliberately, and monitor manually if you do.
- **Treat `configs/tasks/*.yaml` as the audit trail.** Every threshold
  and rule that governs a task's behavior lives there — useful given any
  governance/compliance requirements around automated decision systems.
- **Generative student models need real work before use.**
  `generative_student.py` is a skeleton (the LoRA fine-tuning loop is
  left unimplemented) — classification/extraction tasks are ready to
  use as-is; generation tasks need that loop filled in for your chosen
  base model first.

## License

MIT — see `LICENSE`.
