# Confidence scoring

This is the part of the system that everything else depends on being
right. If confidence scores are wrong, the whole cascade either escalates
too often (no savings) or too rarely (silent failures reach users). This
doc explains the mechanics in more depth than `architecture.md`.

## Why raw model confidence isn't enough

A classifier's `predict_proba()` output is not a real probability out of
the box. Models trained with standard loss functions are frequently
**overconfident** — a model might output 0.95 "confidence" on inputs
it's only actually correct on 70% of the time. This is a well-documented
phenomenon (see: Guo et al., "On Calibration of Modern Neural Networks").

Using raw confidence directly as your escalation gate means your
threshold doesn't mean what you think it means.

## The fix: calibration

`cascade/router/calibration.py` fits an **isotonic regression** mapping
from raw confidence → actual empirical accuracy, using logged
`(raw_confidence, was_correct)` pairs. Isotonic regression is used
instead of a simpler method (like a single temperature scalar) because it
makes no assumption about the shape of the miscalibration curve — it just
needs raw confidence to be monotonically related to correctness, which is
a much weaker assumption.

For the `ClassifierStudent`, this is actually handled two ways:
- `CalibratedClassifierCV` (from scikit-learn) calibrates the classifier
  itself at training time, via cross-validated Platt scaling.
- `ConfidenceCalibrator` calibrates again downstream at the router level,
  using held-out logged outcomes — useful because it captures real-world
  drift the training-time calibration can't see yet.

**Before enough data exists to fit calibration** (fewer than 20 labeled
pairs), `ConfidenceCalibrator.calibrate()` applies a flat 0.8x haircut to
raw confidence rather than trusting it outright. This is a deliberate
conservative default — better to escalate a bit more than necessary early
on than to trust an unvalidated confidence score.

## Where labeled (confidence, correctness) pairs come from

Two sources, in order of when they become available:

1. **Shadow-mode agreement** (automatic, available from day one): during
   shadow mode, both the student's prediction and the LLM's real answer
   are logged for the same request. `auto_label_shadow_agreement()`
   compares them — if they match, `was_correct = True`. This gives you
   calibration data before you've promoted anything, at zero extra cost.

   Caveat: exact-match agreement works well for classification/extraction
   tasks. For open-ended generation, replace this with a similarity
   threshold (e.g. embedding cosine similarity above some cutoff) or a
   human review step — exact string match will produce false negatives
   on paraphrased-but-correct answers.

2. **Explicit feedback** (higher quality, requires integration effort):
   call `runner.record_feedback(request_log_id, was_correct)` whenever
   your system learns the true outcome — a user flags an issue, a
   downstream validation step fails, a human reviewer checks a sample.
   This is strictly more reliable than shadow-mode agreement and should
   be wired in wherever feasible.

## The novelty signal

Confidence alone can't catch a specific failure mode: a student model
that's *confidently wrong* on an input unlike anything it was trained on.
Calibration corrects confidence on average, but a single unusual input
can still fool a well-calibrated model.

`NoveltyDetector` embeds the input (via `sentence-transformers`,
`all-MiniLM-L6-v2` by default) and compares it against everything logged
for the task so far, using cosine similarity. The novelty score is
`1 - max_similarity` — high when nothing similar has been seen.

`ConfidenceRouter` blends this into the routing decision:

```python
combined = calibrated * (1 - novelty_weight) + (1 - novelty) * novelty_weight
```

Default `novelty_weight` is 0.3 — novelty pulls the combined score down
but doesn't dominate it. Raise this weight for tasks where "have I truly
seen this exact pattern before" matters a lot (e.g. fraud-adjacent
decisions); lower it for tasks with high natural input variety where
novelty is expected and not itself a risk signal (e.g. general customer
message classification).

## Upgrading to a learned router

Once you have a few hundred labeled routing outcomes — `(raw_confidence,
novelty_score) -> was_correct` — the fixed weighted combination in
`ConfidenceRouter` can be replaced by `router/meta_model.py`'s
`MetaRouter`, a small gradient-boosted classifier that learns the actual
relationship between these signals and correctness for your specific
task, rather than assuming a fixed linear blend. This mirrors what
RouteLLM does for LLM-tier routing.

Don't reach for this on day one — a learned router on too little data
will overfit and be *less* reliable than the simple weighted combination.
200+ labeled examples is a reasonable minimum before switching.

## Practical thresholds to start with

| Stage | Suggested starting threshold | Rationale |
|---|---|---|
| Assisted | 0.85 | Conservative enough that a wrong student answer is rare, but starts capturing real savings |
| Autonomous | 0.95 | Reserved for tasks that have proven out extensively in Assisted mode first |

Treat these as starting points, not fixed rules — tune per task based on
what a wrong answer actually costs you. A task where an incorrect answer
is mildly annoying (e.g. a suggested tag) can tolerate a lower threshold
than one where it's costly (e.g. an amount extracted for an invoice
that gets paid automatically).
