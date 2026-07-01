# Ablation Delivery Pipeline — Architecture

**Status:** Accepted (describes practice as observed on `main` and the `ablate-h`→`ablate-p` branch lineage)
**Author:** Documentation pass by Claude (Sonnet 5), reviewed against repo state on 2026-07-01
**Date:** 2026-07-01
**Reviewers:** Gideon Potok

## Context

This repo runs reinforcement-learning ablations for a stratospheric balloon
station-keeping agent (DQN / QR-DQN / Option-Critic, over a Node.js physics
environment). Ablations are shipped at high velocity — ten of them (H through
P) landed between 2026-06-24 and 2026-07-01, several within the same afternoon
(K, L, K2, M, N, O all landed 2026-06-30 between 13:04 and 18:26). This
document exists because that velocity outran the repo's own documentation:
the root `README.md` still describes only Ablation H, and — as detailed in
[Key Decisions](#key-decisions) below — it doesn't even exist on the git
lineage that most later ablations (J–P) descend from. This doc captures how
delivery actually works today, not an aspirational rewrite of it.

## Requirements

### Must Have
- Train 10 workers in parallel per ablation without paying for dedicated compute (GitHub Actions free-tier runners, CPU only).
- Reproduce which exact code + hyperparameters produced a given `.pt` checkpoint.
- Turn a checkpoint into a visual, human-checkable artifact (PNG trajectory panel, GIF flight animation) before trusting it.

### Nice to Have
- Compare scores/hyperparameters across ablations without opening every branch.
- Catch config regressions (e.g. a hyperparameter silently not taking effect) before a 10-worker, multi-hour run burns compute on it.

### Non-Goals
- GPU training (repo installs the CPU-only `torch` wheel in CI).
- A hosted metrics dashboard (not in place today — see [ADR-0001](../adr/0001-experiment-tracking-tooling.md)).

### Not Part of This Pipeline (despite appearances)
- **`task_scheduler.py`** (repo root) — its name suggests it might orchestrate worker/ablation scheduling. It doesn't: it's a generic, self-contained priority-queue exercise (`Task` dataclass, heap-based `SchedulerService`) with no reference to balloons, weights, or CI anywhere in it. Don't look here for orchestration logic — that's `ablate_<letter>_train.py` + `run_worker.py` + `train.yml`.
- **`scratch/`** — contains only agent-harness session boilerplate, not experiment scratch data.

## Architecture Overview

```
 ideation              implementation            CI launch                 collection              visualization
┌────────────────┐   ┌─────────────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│ docstring in a │   │ ablate_<x>_train.py │   │ train.yml          │   │ collect job        │   │ replay.py (PNG)      │
│ new            │──▶│  + hand-edit         │──▶│  workflow_dispatch │──▶│  download-artifact │──▶│ make_gif.py (GIF)    │
│ ablate_<x>_    │   │    run_worker.py     │   │  matrix: 10 workers│   │  collect.py picks  │   │  against the winning │
│ train.py       │   │    collect.py        │   │  each ≤6h          │   │  max(best_score)   │   │  weights/dqn_ablate_ │
│ (hypothesis +  │   │  to the new letter   │   │                    │   │  uploads            │   │  <x>.pt              │
│ prior-ablation │   │                      │   │                    │   │  final-ablate-<x>  │   │                      │
│ diagnosis)     │   │                      │   │                    │   │                    │   │                      │
└────────────────┘   └─────────────────────┘   └───────────────────┘   └───────────────────┘   └──────────────────────┘
                              │                          │                       │
                              ▼                          ▼                       ▼
                       qr_agent.py (shared)      servers/balloon_env_    weights/  (local, mostly
                       replay_buffer.py (shared)  server{,_v2}.mjs + js/  gitignored except
                       balloon_env.py (shared)    (Node physics process)  final-ablate-h/)
```

Everything left of "CI launch" happens on a human/agent's laptop and gets
pushed as a branch. Everything from "CI launch" through "collection" happens
on GitHub-hosted runners, triggered manually. "Visualization" happens back on
a laptop against a downloaded checkpoint.

## Components

### Ideation layer — `ablate_<letter>_train.py` docstring
- **Purpose:** the only "lab notebook" that exists. Every training script opens with a structured docstring: a one-line name for the ablation, a "Question this answers" section, and a "Changes vs Ablation `<prior>`" diff list.
- **Example (Ablation H, `ablate_h_train.py:1-18`):**
  ```
  Ablation H — Extended ablate_a (plain DQN, v1 server, 3600 episodes):
  Question this answers:
    "Was ablate_a still improving at ep 2799 because it hadn't converged,
     or was it close to its ceiling? Does more training close the 4pp gap
     with the heuristic?"
  Changes vs Ablation A: [...]
  ```
- **Scaling:** rich per-ablation, but there is no cross-ablation index of these — to see the whole research narrative you must check out every branch and read every docstring in order.

### Shared library layer
- **`qr_agent.py`** — one `QRConfig`/`QRNetwork`/`QRAgent` used, unmodified, by every ablation from H through P. `n_quantiles=1` degrades it to plain DQN; `n_quantiles=51` gives full QR-DQN; boolean flags (`use_recurrent`, `use_options`) bolt on a GRU-64 recurrent core and Option-Critic heads. Three training methods (`train_batch`, `train_batch_seq`, `train_batch_options`) coexist in one class, selected by config rather than by subclassing.
- **`replay_buffer.py`** — Prioritized Experience Replay + n-step accumulator, shared unchanged.
- **`balloon_env.py`** — Gym-style Python wrapper that spawns `node servers/balloon_env_server{,_v2}.mjs` as a persistent subprocess and speaks NDJSON over stdin/stdout (`reset`, `step`, `heuristic_step`, `close`).

### Environment layer — `servers/*.mjs` + `js/`
- **`servers/balloon_env_server.mjs` (v1)** — frozen physics/reward engine, used by Ablation H as the stable baseline.
- **`servers/balloon_env_server_v2.mjs` (v2)** — actively evolving; carries every reward-shaping / state-expansion / time-feature change from Ablation I onward. `BalloonEnv(server_version=...)` selects between them.
- **`js/`** — added wholesale in one commit (`aa09d55`) because the servers `import` from it. Only about half the files are real runtime dependencies (`config.js`, `atmosphere.js`, `geo.js`, `wind.js`, `balloon.js`, `wind_observer.js`, `wind_ekf.js`, `wind_degrader.js`, `navigator.js`, and one helper from `rl_agent.js`). The rest (`app.js`, `dqn.js`, `qr_agent.js`, `iqn_agent.js`, `rl_trainer.js`, `cem_mpc.js`, `charts.js`, `map.js`, `wind_archive.js`, `forecast.js`) are vestiges of a prior in-browser dashboard tool and are not imported by the training pipeline at all.

### Orchestration layer
- **`ablate_<letter>_train.py`** — local entry point: curriculum table, `BASE_CONFIG`, a `multiprocessing` launcher that starts N worker processes.
- **`run_worker.py`** — single-worker CI entry point. Imports `worker_fn` from a specific `ablate_<letter>_train` module and holds a `WEIGHTS_PREFIX` constant — both hand-edited per ablation (see [Key Decisions](#key-decisions)).
- **`collect.py`** — reads every `weights/{WEIGHTS_PREFIX}_w*.json`, prints a score table, copies the best worker's `.pt` to `weights/{WEIGHTS_PREFIX}.pt`, writes `{WEIGHTS_PREFIX}_summary.json`.

### CI layer — `.github/workflows/`
- **`train.yml`** — `workflow_dispatch`-triggered. `train` job: 10-way matrix (`worker_id: [0..9]`), `timeout-minutes: 360`, `fail-fast: false`, uploads artifact `worker-<id>` (glob over zero-padded `.pt`/`.json`). `collect` job: runs after `train` succeeds or fails (not on cancel), downloads all `worker-*` artifacts into `weights/`, runs `collect.py`, uploads `final-ablate-<letter>` (90-day retention).
- **`smoke_test.yml`** — one worker, 5 episodes, uploads `smoke-weights` (3-day retention) — meant as a fast sanity check before committing 10 workers × 6h of compute to a broken config.

### Artifact + visualization layer
- GitHub Actions artifacts are downloaded manually (`gh run download` or the Actions UI — no script wraps this) into `weights/`.
- **`replay.py`** — loads a checkpoint, reconstructs `QRConfig` from the checkpoint's own saved config, forces greedy eval (`epsilon=0`), runs one episode per wind preset, renders a 5-panel static PNG (trajectory map, altitude vs. time, distance vs. time, action histogram, action sequence).
- **`make_gif.py`** (introduced for Ablation N, extended for M) — a registry-based tool: an `ENV_FLAGS` dict keyed by ablation id (`'k2'`, `'l'`, `'m'`, …) plus `AGENT_KWARGS` for architecture flags not recoverable from the checkpoint (e.g. Option-Critic/GRU flags aren't in `QRAgent.state_dict()`), driving `matplotlib.animation.FuncAnimation` + `PillowWriter`.

## Data Flow

1. Pick the prior ablation to diagnose or extend; write a new `ablate_<letter>_train.py` with the ideation docstring and the actual code delta.
2. Hand-edit three files to point at the new letter: `run_worker.py`'s import + `WEIGHTS_PREFIX`, `collect.py`'s `WEIGHTS_PREFIX`, and `.github/workflows/train.yml`'s workflow name + 4 path strings (upload glob ×1, download path, final-artifact name, final-artifact path).
3. Push a branch named `ablate-<letter>-<slug>`.
4. Trigger `smoke_test.yml` manually (1 worker, 5 episodes) as an end-to-end sanity check.
5. Trigger `train.yml` manually — 10 workers in parallel, each capped at 6h, each saving `weights/dqn_ablate_<letter>_w<NN>.pt` (+ `.json`) and uploading it as its own artifact.
6. The `collect` job downloads every worker artifact into `weights/`, runs `collect.py` (picks `max(best_score)`), and uploads `final-ablate-<letter>`.
7. A human or agent downloads `final-ablate-<letter>` into local `weights/`.
8. `replay.py` and/or `make_gif.py` run locally against the winning `.pt`, producing `replay_ablate_<letter>_<preset>.png` / `.gif`, which get committed to the repo root as the visual deliverable.

## Key Decisions

### Decision 1: One config-driven agent class instead of one class per algorithm variant
- **Options considered:** a `QRAgent` subclass per variant (DQN, QR-DQN, Option-Critic+GRU) vs. one class with config flags.
- **Chosen:** one class (`qr_agent.py`), config-toggled.
- **Rationale:** every ablation shares 90% of its training loop; a shared class lets Option-Critic (Ablation M) and Fourier features (Ablation L) compose instead of forking the whole file.
- **Trade-offs:** the class now carries three parallel training methods (`train_batch`, `train_batch_seq`, `train_batch_options`); a bug in shared code (e.g. the eval epsilon-greedy leak fixed in Ablation N) affects every ablation that imports it.
- **Revisit when:** the flag surface grows past what one config dataclass can express clearly.

### Decision 2: Two parallel env server files (v1 frozen, v2 evolving) instead of one flagged file
- **Options considered:** version the physics/reward engine in place vs. keep v1 as an untouched baseline and let v2 absorb every reward-shaping/state change.
- **Chosen:** two files, selected via `BalloonEnv(server_version=...)`.
- **Rationale:** keeps Ablation H's baseline reproducible even as v2 changes weekly.
- **Trade-offs:** shared physics code (~half of each file) must be mentally kept in sync between two ~300–600 line `.mjs` files; nothing enforces that they don't drift on the parts that are supposed to be identical.
- **Revisit when:** v2 stabilizes enough to become the new frozen baseline, or a third variant is needed.

### Decision 3: Per-ablation CI config via hand-edited hardcoded strings, not `workflow_dispatch` inputs
- **Options considered:** parameterize `train.yml`/`run_worker.py`/`collect.py` with an `ablation_id` input vs. hand-edit three files per ablation.
- **Chosen (so far):** hand-edit. `run_worker.py` still reads `from ablate_i_train import worker_fn` / `WEIGHTS_PREFIX = "dqn_ablate_i"` even on the `ablate-j-recovery-spawn` branch at time of writing.
- **Rationale:** fastest to stand up for Ablation H; nobody has circled back to parameterize it.
- **Trade-offs:** this is the direct cause of the bug history in [Failure Modes](#failure-modes) — three files must change in lockstep, silently, with no validation that they agree.
- **Revisit when:** now — see [ADR-0001](../adr/0001-experiment-tracking-tooling.md) and the [runbook](../runbooks/ship-a-new-ablation.md) for the concrete fix.

### Decision 4: One branch per ablation, chained onto whichever prior ablation is the relevant baseline (not always the previous letter)
- Ablations don't fork linearly from `main` each time. K2 re-runs K after a server bug fix; L branches from K; M branches from L; N branches from K2 (a sibling of M, not a descendant); O and P continue N's line while conceptually reusing M's option-critic ideas by hand re-implementation, not `git merge` (no merge commits exist anywhere in this history). See the [runbook's branching section](../runbooks/ship-a-new-ablation.md#branching-model) for the full picture and its consequences.

## Failure Modes

| Failure | Impact | Detection | Recovery |
|---------|--------|-----------|----------|
| CI upload glob didn't match zero-padded worker filenames (`w0` vs `w00`) | `collect.py` always failed — every worker artifact was effectively empty | `collect` job failure | `82a5ff0`: switched to wildcard glob `dqn_ablate_h_w*.pt` |
| A second, independent fix for the same class of bug landed on a stray branch (`worktree-greedy-plotting-swan`, commit `6cb35b1`), hardcoding `w0${{matrix.worker_id}}` — only correct for single-digit ids, never merged | Duplicated effort; a narrower, still-fragile fix stranded off main | Manual branch review | Superseded by the wildcard-glob fix on main; the stray branch was never cleaned up |
| `download-artifact`'s `path: .` stripped the `weights/` prefix `collect.py` expects | `collect.py` reported "no worker JSON files found" even after the glob fix | `collect` job failure | `174adf6`: changed `path: .` → `path: weights` |
| Python's `_env_flags()` sent `shaping_linear`/`shaping_D_max`, but the JS server's `handleReset()` never extracted those keys before using them | **Silent** — Ablation K's entire run used `τ=100km` (H's old default) instead of the intended `τ=500km`, with no error anywhere | Only caught by a human/agent diffing K's and K2's result quality after the fact | Re-ran as Ablation K2 once the missing `const shapingLinear = !!req.shaping_linear; …` extraction was added |
| Smoke-test output (`smoke_test.yml`, 1 worker, 5 episodes) writes to the same filenames (`weights/dqn_ablate_i.pt`, `_summary.json`) as a real 10-worker/3600-episode run | Ablation I's committed "final" weights are actually a 5-episode smoke run (`n_workers: 1`, `best_episode: 4`, `best_score: 6.9%`) — anyone replaying them sees a near-random policy and could mistake it for a trained result | Only visible by opening the summary JSON and noticing `n_workers: 1` | Not yet fixed — no distinct namespace exists for smoke vs. production output |
| `EnterWorktree`-style branch creation collided with a pre-existing branch name: `ablate-j-recovery-spawn` never received the actual "Ablation J" commit (`0123b86`); that commit only landed on the confusingly similar `worktree-ablate-j-recovery-spawn` | The branch named for Ablation J doesn't contain Ablation J; anyone continuing "ablation J" work on the plain-named branch is missing the actual commit | `git merge-base --is-ancestor 0123b86 ablate-j-recovery-spawn` → no | Not yet reconciled — flagged here so it isn't silently perpetuated |
| `README.md` was added in a side commit (`b8416bf`, on `ablate-i-linear-shaping`) that is **not an ancestor of `main`** (`main` is at `174adf6`) | `README.md` does not exist at all on the branch lineage that J, K, L, K2, M, N, O, P all descend from — the project's only onboarding doc is invisible from most of the active work | `git merge-base --is-ancestor b8416bf ablate-p-clipped-option-critic` → no | This `docs/` tree is added on top of `main` (`174adf6`) for the same reason — see the note in the [runbook](../runbooks/ship-a-new-ablation.md) about merging docs forward |

## Security Considerations

Not applicable in the traditional sense — no user-facing surface, no secrets beyond the implicit GitHub Actions token. Worth noting: nothing in `train.yml` pins the `torch`/`numpy` versions installed in CI (`pip install torch numpy --index-url ...`), so a CI run today is not guaranteed to reproduce the exact dependency versions used in an earlier ablation's run.

## Operational Concerns

- **Monitoring:** per-worker eval lines printed to the GitHub Actions log (score, per-preset breakdown, epsilon) — this is the *only* mid-run visibility; there is no persisted metric history, so once a run's log expires (or the run is superseded), that visibility is gone. See [ADR-0001](../adr/0001-experiment-tracking-tooling.md).
- **Alerts:** none — a broken run is discovered by reading the Actions UI, not by any notification.
- **Deployment:** manual `workflow_dispatch` trigger only; nothing auto-triggers on push (deliberate — a 10-worker × 6h matrix is expensive to fire accidentally).
- **Rollback:** informal — "rolling back" means checking out a prior ablation's branch; there's no formal revert procedure.

## Future Considerations

- Parameterize `train.yml` with a `workflow_dispatch` input (`ablation_id`) instead of hand-editing three files per ablation — this single change would have prevented the entire artifact-path bug saga.
- Add a lightweight schema check between Python's `_env_flags()` and the JS server's `handleReset()` parsing (even just an assertion that every key sent is a key read) to catch the K/K2-style silent flag drop at run time instead of after the fact.
- Give smoke-test output its own filename prefix or subdirectory so it can never be mistaken for a production result.
- Prune or relocate the dead half of `js/` (`app.js`, `dqn.js`, `iqn_agent.js`, `rl_trainer.js`, `cem_mpc.js`, `charts.js`, `map.js`, `wind_archive.js`, `forecast.js`) — none of it is imported by the training pipeline.
- Consolidate `replay.py` and `make_gif.py` behind one shared per-ablation registry (generalizing `make_gif.py`'s `ENV_FLAGS`/`AGENT_KWARGS` pattern to drive both PNG and GIF output), rather than maintaining two tools plus one-off scripts like the observed `make_gif_j.py`.
- Consider real experiment-tracking tooling — see [ADR-0001: Experiment Tracking Tooling](../adr/0001-experiment-tracking-tooling.md).
