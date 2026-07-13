# ADR-0001: Experiment Tracking Tooling (W&B / MLflow / TensorBoard) vs. Status Quo

**Status:** Proposed (analysis for consideration — not yet a team decision)
**Date:** 2026-07-01
**Deciders:** Gideon Potok (proposal authored by Claude/Sonnet 5 based on repo archaeology; see [architecture/ablation-pipeline.md](../architecture/ablation-pipeline.md) and [runbooks/ship-a-new-ablation.md](../runbooks/ship-a-new-ablation.md))

## Context

Today, "experiment tracking" for this project is: a docstring per ablation
script, a `WEIGHTS_PREFIX`-scoped `_summary.json` per ablation, GitHub
Actions log lines for mid-run visibility, a root `README.md` frozen on
Ablation H, and git branch names as the comparison axis. This has real,
observed costs:

1. **No cross-ablation comparison.** To compare H through P's scores and
   hyperparameters, you must open ten branches and read ten docstrings /
   JSON files by hand. There is no single table.
2. **No mid-run visibility.** The only signal during a 6-hour worker run is
   periodic eval lines in an ephemeral Actions log. There is no persisted
   loss curve, score-over-episodes curve, or epsilon-decay curve for any
   ablation — once a run's log expires, that history is gone.
3. **Silent config bugs go undetected until a full run completes.**
   Ablation K's shaping flags never reached the JS server's parsed request
   (see the architecture doc's [Failure Modes](../architecture/ablation-pipeline.md#failure-modes)).
   A live per-worker metric stream showing the shaping term flatlining might
   have surfaced this in the first few episodes instead of after a full
   10-worker run.
4. **Smoke-test output is indistinguishable from production output** by
   filename alone (Ablation I's committed "final" weights are 5-episode
   smoke output) — nothing tags a run with its own provenance.
5. **CI config is hand-duplicated across three files per ablation**
   (`run_worker.py`, `collect.py`, `train.yml`) — orthogonal to tracking
   tooling, but the same underlying habit (no single source of truth for
   "what ablation is this run") shows up in both problems.

The question: would adopting Weights & Biases, MLflow, or TensorBoard fix
enough of this to be worth the added dependency, given the project's actual
shape — a single researcher, CPU-only training via GitHub Actions matrix
jobs, ~10 ablations shipped in one week at a pace of several per afternoon?

## Decision

**Add lightweight, file-based tracking now; do not add a hosted service
yet.** Concretely: TensorBoard for the mid-run visibility gap (problem 2/3),
plus either a lightweight MLflow file-store or — if that's more dependency
than wanted — a single hand-rolled `results.jsonl` appended to by
`collect.py`, for the cross-ablation comparison gap (problem 1/4). Defer
Weights & Biases entirely unless this becomes a multi-person effort or grows
into real hyperparameter sweeps.

## Alternatives Considered

### TensorBoard
- **Pros:** near-zero setup (`torch.utils.tensorboard.SummaryWriter`, a few lines in the worker's eval loop); no server, no account, no secrets in CI; event files can be uploaded as a GitHub Actions artifact next to the weights, at no extra infra cost; directly fixes problem 2 and would very likely have shortened the K/K2 bug's time-to-detection (problem 3) — a shaping-reward curve going flat is visually obvious in a way that a final scalar score isn't.
- **Cons:** no built-in cross-run comparison table (problem 1) without manually pointing `tensorboard --logdir` at all ten runs' event files side by side; no structured hyperparameter/config store; doesn't address problem 4/5 at all.
- **Fit:** high leverage for its cost. This is the easiest "yes."

### MLflow (local file store, no server)
- **Pros:** `mlflow.log_params(config)` / `mlflow.log_metrics(...)` / `mlflow.log_artifact(path)` calls in `collect.py` and `replay.py` would give exactly the missing cross-ablation table (problem 1) — sortable by score, filterable by hyperparameter, with the winning `.pt` and PNG/GIF attached as run artifacts. Runs can be tagged `smoke` vs. `full` at creation time, directly closing problem 4. No hosted account needed — a local `mlruns/` directory (or one uploaded as a CI artifact and pulled down for `mlflow ui` locally) is enough for a solo researcher.
- **Cons:** another dependency and another thing to wire into `run_worker.py`/`collect.py`; if the tracking store isn't persisted somewhere durable (e.g. only ever local to a given CI runner), it needs its own "download and merge" step, which is a new variant of the exact artifact-plumbing fragility that already burned three fix commits in this repo's history. Needs care to not become moving-part #4.
- **Fit:** highest leverage for the *stated* problem (comparing ablations, this repo's central pain), but only if the artifact/download wiring is done more carefully than the existing weights pipeline was.

### Weights & Biases
- **Pros:** richest dashboards, automatic system-metric capture, best-in-class run comparison and hyperparameter-sweep support, real collaboration features if this ever becomes a multi-person project.
- **Cons:** requires an account and an API key as a GitHub Actions secret — a new external dependency and a new failure surface (auth misconfiguration, rate limits, service outages) in a pipeline that is *already* the most fragile part of this project (see the artifact-path bug saga). Sends training metadata to a third-party cloud service, which is unnecessary exposure for what is currently solo, exploratory research. Sweep support is not needed yet — the "ablations" here are hand-designed single/few-variable changes, not automated hyperparameter search over a space large enough to need W&B's sweep scheduler.
- **Fit:** over-provisioned for the current scale. Revisit if the project grows to automated sweeps or multiple collaborators.

### Status Quo (docstrings + JSON summaries + README + git branches)
- **Pros:** zero added dependencies; already fully understood by the person running it; the docstring convention is genuinely good as an ideation record (see the architecture doc).
- **Cons:** this is the thing generating the actual, documented pain in this ADR's Context section. The frozen README and the ten-branches-to-compare problem are symptoms of *not* having tracked, structured, cross-run data — they are not a viable long-term substitute for it, especially at the observed shipping cadence (multiple ablations per afternoon means the manual-comparison cost compounds daily).
- **Fit:** acceptable only if the project's pace slows down significantly or stops; not recommended to continue as-is at the current cadence.

## Consequences

### Positive
- TensorBoard gives visibility into training dynamics that has never existed in this project, for near-zero cost, and would likely have caught the K/K2 bug class faster.
- A lightweight MLflow store (or even a hand-rolled `results.jsonl`) replaces "read ten branches by hand" with one sortable table, directly serving the comparison need that motivated this document.
- Neither choice requires secrets in CI, keeping the current no-external-account posture.

### Negative
- Any tracking layer that spans CI runs needs its own artifact-persistence story; done carelessly, it becomes a fourth hand-synced moving part alongside `run_worker.py`/`collect.py`/`train.yml` — the exact failure pattern already documented. This should be built *after* (or as part of) parameterizing `train.yml`, not bolted on top of the current hand-edited-three-files pattern.
- Adds a Python dependency (`tensorboard`, and/or `mlflow`) to a repo that currently has exactly two (`torch`, `numpy`).

### Neutral
- W&B is not rejected outright — it's deferred pending a concrete trigger (multi-person collaboration, or a real hyperparameter sweep) rather than judged unsuitable forever.

## Follow-up Actions

- [ ] Add `SummaryWriter` logging to the worker eval loop (score, per-preset breakdown, epsilon) and upload the event file as part of each `worker-<id>` CI artifact.
- [ ] Decide between a lightweight MLflow file store and a hand-rolled `results.jsonl` appended to by `collect.py`; either should record: ablation id, git branch/commit, full hyperparameter config, per-preset scores, `smoke` vs. `full` tag, and links to the produced PNG/GIF.
- [ ] Do this alongside — not instead of — parameterizing `train.yml` per the architecture doc's [Future Considerations](../architecture/ablation-pipeline.md#future-considerations); tracking tooling built on top of the current hand-edited CI config inherits its fragility.
