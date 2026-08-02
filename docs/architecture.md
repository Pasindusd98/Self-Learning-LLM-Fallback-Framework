# Architecture

## The core idea

Most "AI-integrated systems" call a third-party LLM repeatedly for the
same category of task — classify this ticket, extract these fields,
route this request. Most of that traffic is repetitive. This framework
watches what the LLM does on a given task, trains a small local model to
replicate it, and only calls the LLM when the local model isn't
confident enough to be trusted.

```
┌─────────────────────────────────────────────────────────┐
│                 YOUR SYSTEM (host app)                   │
│           calls CascadeRunner.run(...) instead           │
│           of calling the LLM directly                    │
└───────────────────────┬───────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  CascadeRunner        │  cascade/interceptor/proxy.py
              │  (the interceptor)     │
              └──────────┬────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                                 ▼
┌───────────────┐              ┌────────────────────┐
│  RequestLog     │              │  ConfidenceRouter    │
│  (storage/)     │              │  (router/)            │
└───────┬────────┘              └──────────┬──────────┘
        │                                  │
        ▼                          ┌───────┴────────┐
┌────────────────┐                 ▼                ▼
│ training/       │      ┌─────────────────┐  ┌─────────────┐
│ trainer.py       │─────▶│  StudentModel    │  │  LLMProvider  │
│ scheduler.py      │     │  (student_models/)│  │  (fallback)   │
└────────────────┘        └─────────────────┘  └─────────────┘
        │
        ▼
┌────────────────┐
│  StageManager    │  cascade/stages/
│  (promotion /     │
│   demotion)        │
└────────────────┘
```

## Lifecycle: the three stages

Every task starts in **Shadow** mode and only moves forward when the
data justifies it.

| Stage | What happens | Risk |
|---|---|---|
| **Shadow** | Every request goes to the LLM. The student model predicts silently alongside it, purely for logging/evaluation. | Zero — production behavior is unchanged. |
| **Assisted** | Student predicts first. If confidence ≥ `assisted_mode` threshold, serve its output. Otherwise, call the LLM. | Low — only confident predictions bypass the LLM. |
| **Autonomous** | Same as Assisted but with a higher confidence bar (`autonomous_mode`). LLM becomes a rare safety net. | Requires sustained proven accuracy to reach. |

Promotion requires **all** of:
1. Rolling accuracy ≥ the stage's threshold
2. At least `min_samples_before_promotion` logged examples
3. `min_stable_cycles` consecutive retrain cycles above threshold (guards
   against promoting on a lucky streak)

**Demotion** is checked every cycle too, independent of promotion: if
rolling-window accuracy falls below `demotion_accuracy_floor`, the task
drops one stage immediately. This is the drift safety net — a task can
be promoted and later demoted if the world changes underneath it (e.g.
users start asking about a new product category the student never saw).

## Confidence gating in detail

Raw model confidence (a softmax probability, for example) is usually
overconfident. `ConfidenceRouter` (cascade/router/confidence.py) combines
two signals into one number:

1. **Calibrated confidence** — the student's raw confidence is passed
   through an isotonic regression fit on (confidence, was_it_actually_correct)
   pairs, so "0.9 confidence" actually means "correct about 90% of the time"
   rather than whatever the raw model happens to output.
2. **Novelty score** — how similar the current input is (via sentence
   embeddings) to anything the student has seen before. An input that's
   never been seen pulls the combined confidence down even if the model
   itself claims to be sure — this catches "confidently wrong on
   unfamiliar input," the main real-world failure mode.

Once you have a few hundred labeled routing outcomes, you can swap the
weighted combination for `router/meta_model.py`, a small learned model
(gradient-boosted trees) that fits the actual relationship between these
signals and correctness — the same idea RouteLLM uses.

## Student model types

Not every task needs a fine-tuned LLM as its student:

- **Classification / routing / tagging / extraction** → `ClassifierStudent`
  (TF-IDF + calibrated linear classifier). Fast to train, cheap to run,
  calibration is handled natively by scikit-learn's `CalibratedClassifierCV`.
- **Open-ended generation** (drafting, summarizing) → `GenerativeStudent`
  (a small pretrained LLM like Phi-3-mini or Llama 3.2 3B, LoRA fine-tuned
  on logged prompt→output pairs). This is left as a skeleton in the repo —
  fill in the LoRA training loop for your chosen base model once your
  first classifier-based task is working end to end.
- **Highly structured/rule-friendly tasks** → implement a custom
  `StudentModel` subclass combining rules with a fallback classifier
  ("hybrid" in the config).

## Data flow: distillation loop

```
1. Request comes in -> CascadeRunner.run()
2. Student predicts (if trained) -> logged
3. Router decides: serve student, or call LLM?
4. LLM's answer (when called) becomes ground truth for training
5. training/scheduler.py runs periodically:
   a. Pull all (input, LLM_output) pairs since last training
   b. Fit/refit the student model on them
   c. Auto-label shadow-mode agreements for calibration data
   d. Refit the confidence calibrator
   e. Evaluate held-out accuracy
   f. StageManager checks promotion/demotion rules
6. Repeat -- the student gets better specifically on cases it
   used to escalate, since those are exactly what gets logged.
```

## Governance notes

Every request is logged with its input, which handler served it, the
stage at request time, and (once known) whether it was correct. This
gives you a full audit trail for questions like "why was this task
promoted to Autonomous" or "how often did the student get this category
wrong before we caught it" — worth keeping in mind given AI governance
work tends to require exactly this kind of traceability.
