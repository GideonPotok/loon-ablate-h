# Ablation H — Extended DQN Training for Stratospheric Balloon Station-Keeping

**Question this answers:** Was Ablation A still improving at episode 2799 because it hadn't converged, or was it close to its ceiling? Does more training close the ~4 pp gap with the heuristic baseline?

Ablation H is a direct extension of Ablation A: same plain DQN configuration (n_quantiles=1, v1 server, PER + n-step=3), with the curriculum extended from 2800 to 3600 episodes per worker — an extra 48h tier doubled to 800 episodes and a new 72h tier added.

---

## Background

The environment simulates a stratospheric balloon maintaining position over a fixed ground station (TWR50: fraction of 5-minute navigation intervals spent within 50 km of station). The physics engine runs as a persistent Node.js subprocess (`servers/balloon_env_server.mjs`), communicating with Python over NDJSON on stdin/stdout. Three wind presets test different operating conditions: `tropical`, `strong-shear`, and `calm`.

The agent chooses one of 17 discrete target altitudes every 5 minutes. The JS side applies a bang-bang chase command toward that altitude.

**State vector (20-dim):** distance/bearing to station, altitude, vertical velocity, ballast, current wind u/v, and forecast wind at 4 altitudes above.

---

## Changes vs Ablation A

| Parameter | Ablation A | Ablation H |
|-----------|-----------|-----------|
| 48h tier episodes | 400 | 800 |
| 72h tier | — | 400 eps (new) |
| Total eps/worker | 2800 | 3600 |
| n_quantiles | 1 | 1 |
| Server | v1 | v1 |
| Everything else | — | identical |

---

## Curriculum

```
2h×200  →  6h×1000  →  12h×600  →  24h×600  →  48h×800  →  72h×400
```
Total: 3600 episodes per worker. Evaluation uses 72h episodes (aligned with the new final tier).

---

## Eval metric

```
score = 0.5 × mean_TWR50 + 0.5 × worst_preset_TWR50
```

Evaluated every 300 episodes across all 3 presets × 3 runs = 9 rollouts. Ablation A baseline: **49.1%** at episode 2799.

---

## Setup

**Requirements:** Python 3.13, Node.js 22

```bash
pip install torch numpy
```

Node.js must be on `PATH` (the env spawns `node servers/balloon_env_server.mjs` per env instance).

---

## Training

### Locally

```bash
python ablate_h_train.py
```

Spawns 10 worker processes. Each worker saves its best checkpoint to `weights/dqn_ablate_h_w<id>.pt` and a companion `.json`. The global winner is written to `weights/dqn_ablate_h.pt` plus `weights/dqn_ablate_h_summary.json`. Training log: `/tmp/train_ablate_h.log`.

### Via GitHub Actions (recommended)

Trigger the **Train Ablation H** workflow from the Actions tab. It runs 10 workers in a matrix (each capped at 6 hours), then a `collect` job downloads all artifacts, picks the winner, and uploads `final-ablate-h` (retained 90 days).

```
.github/workflows/train.yml      # 10-worker matrix + collect job
.github/workflows/smoke_test.yml # quick sanity check (5 episodes)
```

To run the smoke test locally:

```bash
python run_worker.py --worker-id 0 --max-episodes 5
python collect.py
```

### Collecting results after a matrix run

After downloading all `worker-*` artifacts into `weights/`:

```bash
python collect.py
```

Prints a per-worker score table, copies the winner's `.pt` to `weights/dqn_ablate_h.pt`, and writes the summary JSON.

---

## Visualization

```bash
python replay.py                                          # all 3 presets, 72h, default weight
python replay.py --preset tropical
python replay.py --weight weights/dqn_ablate_h_w03.pt --duration 43200 --seed 7
```

Produces `replay_<preset>.png` with 5 panels: lat/lon trajectory map, altitude over time, distance from station, action distribution histogram, and action sequence.

Sample outputs (trained weights from `weights/final-ablate-h/`):

| tropical | strong-shear | calm |
|----------|-------------|------|
| ![tropical](replay_tropical.png) | ![shear](replay_strong_shear.png) | ![calm](replay_calm.png) |

---

## Repository layout

```
ablate_h_train.py        # training script (local multi-process launcher)
run_worker.py            # single-worker entry point (used by CI)
collect.py               # post-run artifact aggregation + winner selection
replay.py                # eval + visualization
balloon_env.py           # Gym-style Python wrapper around the JS physics server
qr_agent.py              # QR-DQN agent (PyTorch); n_q=1 degrades to plain DQN
replay_buffer.py         # Prioritized Experience Replay + n-step accumulator
servers/
  balloon_env_server.mjs      # v1 physics engine (NDJSON over stdio)
  balloon_env_server_v2.mjs   # v2 (in-development; not used in Ablation H)
js/                      # shared JS modules (wind, balloon physics, navigator, etc.)
weights/                 # checkpoints (gitignored except final-ablate-h/)
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| State dim | 20 |
| Network | 20 → 128 → 64 → 17 |
| n_quantiles | 1 (plain DQN) |
| Optimizer | Adam, lr=1e-4 |
| gamma | 0.97 |
| epsilon | 1.0 → 0.03 (decay 0.9988) |
| Target update | every 15 episodes |
| Replay capacity | 100,000 |
| Batch size | 64 |
| n-step | 3 |
| PER alpha/beta0 | 0.6 / 0.4 |
| Train batches/step | 2 |
| Workers | 10 |
| Eval every | 300 eps |
| Eval duration | 72h |
