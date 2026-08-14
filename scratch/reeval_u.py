"""Clean 10-seed re-eval of Ablation U's winner + navigator-heuristic baseline.

Mirrors the R/S/T probe protocol (seeds 42+i*1_000_003, 72h episodes,
composite 0.5*mean + 0.5*worst over preset means) in U's own env: realism
bundle + navigation mode (spawn near station A, target B 100 km away in a
per-episode random direction).

The baseline U's docstring pre-registers is the JS navigator heuristic on the
SAME episodes: heuristic_step targets ep.targetLat/Lon, which navigation mode
sets to B, and seed+flags fully determine wind, spawn, and target — so agent
and heuristic runs pair per-episode. Reports paired per-episode delta and
t-stat (n=30), like the S-vs-R readout.

Usage:  python scratch/reeval_u.py [--seeds 10] [--duration-h 72]
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from replay import ABLATION_ENV_FLAGS, load_agent, run_episode  # noqa: E402

WEIGHTS = REPO / 'weights' / 'dqn_ablate_u.pt'
OUT     = REPO / 'weights' / 'dqn_ablate_u_reeval.json'
PRESETS = ['tropical', 'strong-shear', 'calm']

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, default=10)
parser.add_argument('--duration-h', type=float, default=72)
args = parser.parse_args()

duration_s = args.duration_h * 3600
flags = ABLATION_ENV_FLAGS['u']
agent = load_agent(WEIGHTS, None)   # plain MLP; state_dim 24 comes from the ckpt

results = {'agent': {}, 'heuristic': {}}
for policy in ('agent', 'heuristic'):
    for preset in PRESETS:
        scores = []
        for i in range(args.seeds):
            seed = 42 + i * 1_000_003
            traj = run_episode(agent, preset, duration_s, seed,
                               server_version='v2', flags=flags,
                               heuristic=(policy == 'heuristic'))
            scores.append(traj['twr50'])
            print(f'{policy:9s} {preset:13s} seed {seed:>9d}  '
                  f'TWR50 {traj["twr50"]*100:5.1f}%', flush=True)
        m = statistics.mean(scores)
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f'{policy:9s} {preset:13s} mean {m*100:5.1f}% ± {sd*100:4.1f}pp '
              f'({args.seeds} seeds)', flush=True)
        results[policy][preset] = scores

for policy in ('agent', 'heuristic'):
    means = {p: statistics.mean(results[policy][p]) for p in PRESETS}
    composite = 0.5 * statistics.mean(list(means.values())) + 0.5 * min(means.values())
    results[f'{policy}_composite'] = composite
    print(f'{policy:9s} composite (0.5*mean + 0.5*worst) = {composite*100:.1f}%',
          flush=True)

# Paired per-episode delta (agent - heuristic) across all preset x seed cells.
deltas = [a - h
          for p in PRESETS
          for a, h in zip(results['agent'][p], results['heuristic'][p])]
n = len(deltas)
mean_d = statistics.mean(deltas)
sd_d = statistics.stdev(deltas) if n > 1 else 0.0
t = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else float('inf')
results['paired'] = {'n': n, 'mean_delta': mean_d, 'sd_delta': sd_d, 't': t}
print(f'paired agent-heuristic: delta {mean_d*100:+.1f}pp ± {sd_d*100:.1f}pp, '
      f't={t:.2f}, n={n}', flush=True)

OUT.write_text(json.dumps(results, indent=2))
print(f'wrote {OUT}', flush=True)
print('REEVAL_DONE')
