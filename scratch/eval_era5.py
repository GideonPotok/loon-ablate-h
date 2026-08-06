#!/usr/bin/env python3
"""
eval_era5.py — how do the realism-era checkpoints do on real ERA5 wind?

Protocol, deliberately close to scratch/reeval_{r,s,t}.py so the numbers sit
next to the existing readout:

  * 30 episodes, 72 h each, seeds 42 + i·1,000,003.
  * Every policy sees the SAME 30 episodes. The server derives the ERA5 cell
    and start time from makeRng(seed + 313131), so a shared seed means a shared
    episode — which makes the comparison paired rather than merely averaged.
  * Each policy runs under its own training-time env flags (R's realism bundle,
    S's with the phase estimator), since those set the state layout.

Four policies:
  R           — the favored realism-era checkpoint (20-dim, no phase info)
  S           — the phase-estimator arm (24-dim); on ERA5 its demodulator is
                locked to the simulator's 8 h IGW frequency, which real wind
                does not have, so this doubles as a test of whether those
                features go from useless to harmful off-distribution
  heuristic   — the JS navigator. THE control. If it collapses too, the ceiling
                is the environment, not the policy.
  float       — hold mid-band all episode. The do-nothing floor.

Under the presets these same checkpoints score ~38%. ERA5 gives a median of
9.1 m/s of in-band shear against the tropical preset's 21.9 m/s step, and
genuinely opposing winds in only ~9.5% of cells, so a large drop is expected
and is not by itself evidence about the policies.

Usage:
    python scratch/eval_era5.py <era5_json_dir> [--episodes 30] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from replay import ABLATION_ENV_FLAGS, load_agent, run_episode  # noqa: E402

DURATION_S = 72 * 3600
SEED0, SEED_STRIDE = 42, 1_000_003

R_FLAGS = dict(ABLATION_ENV_FLAGS['r'])
S_FLAGS = {**R_FLAGS, 'use_estimated_phase_features': True}

# FLOAT_ACTION: mid-band hold. ACTION_DIM is 17, so 8 is the centre bin.
FLOAT_ACTION = 8


class FloatAgent:
    """Holds the mid-band altitude. Stands in for 'do nothing'."""
    def reset_hidden(self): pass
    def select_action(self, state, greedy=True): return FLOAT_ACTION


def summarize(scores: list[float]) -> dict:
    return {
        'mean':   statistics.mean(scores),
        'median': statistics.median(scores),
        'stdev':  statistics.stdev(scores) if len(scores) > 1 else 0.0,
        'min':    min(scores),
        'max':    max(scores),
        'n':      len(scores),
    }


def paired_delta(a: list[float], b: list[float]) -> dict:
    """Paired mean difference a − b with a one-sample t on the differences."""
    d = [x - y for x, y in zip(a, b)]
    m = statistics.mean(d)
    if len(d) < 2:
        return {'delta': m, 't': float('nan'), 'n': len(d)}
    sd = statistics.stdev(d)
    t = m / (sd / len(d) ** 0.5) if sd > 0 else float('inf')
    return {'delta': m, 't': t, 'n': len(d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('era5_dir')
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    seeds = [SEED0 + i * SEED_STRIDE for i in range(args.episodes)]

    policies = [
        ('R',         load_agent(REPO / 'weights' / 'dqn_ablate_r.pt'), R_FLAGS, False),
        ('S',         load_agent(REPO / 'weights' / 'dqn_ablate_s.pt'), S_FLAGS, False),
        ('heuristic', FloatAgent(),                                     R_FLAGS, True),
        ('float',     FloatAgent(),                                     R_FLAGS, False),
    ]

    print(f'ERA5 evaluation — {args.episodes} episodes x 72 h, {args.era5_dir}')
    print(f'seeds {seeds[0]} .. {seeds[-1]} (shared across policies -> paired)\n')

    results: dict = {'episodes': args.episodes, 'duration_h': 72,
                     'era5_dir': args.era5_dir, 'seeds': seeds, 'policies': {}}
    per_episode: dict[str, list[float]] = {}
    winds: list[dict] = []

    for name, agent, flags, heuristic in policies:
        scores, dists = [], []
        for i, seed in enumerate(seeds):
            traj = run_episode(agent, 'tropical', DURATION_S, seed,
                               server_version='v2', flags=flags,
                               wind_source='era5', era5_dir=args.era5_dir,
                               heuristic=heuristic)
            scores.append(traj['twr50'])
            dists.append(traj['dist_m'][-1])   # run_episode already reports km
            if name == 'R':
                winds.append(traj['wind'])
        per_episode[name] = scores
        s = summarize(scores)
        results['policies'][name] = {**s, 'twr50': scores,
                                     'final_dist_km': dists}
        print(f'  {name:10s} TWR50 mean {s["mean"]*100:5.2f}%  median {s["median"]*100:5.2f}%  '
              f'sd {s["stdev"]*100:4.1f}pp  range {s["min"]*100:.1f}-{s["max"]*100:.1f}%  '
              f'final dist {statistics.mean(dists):6.0f} km', flush=True)

    # Every episode identical across policies? If the cells differ the pairing
    # is a lie, so assert it rather than trust it.
    results['episode_cells'] = [
        {'seed': s, 'lat': w.get('lat'), 'lon360': w.get('lon360'),
         'start': w.get('startISO')} for s, w in zip(seeds, winds)
    ]

    print(f'\npaired deltas (same {args.episodes} episodes):')
    comparisons = [('R', 'heuristic'), ('R', 'float'), ('S', 'R'), ('heuristic', 'float')]
    results['paired'] = {}
    for a, b in comparisons:
        pd = paired_delta(per_episode[a], per_episode[b])
        results['paired'][f'{a}_vs_{b}'] = pd
        print(f'  {a:10s} − {b:10s}  {pd["delta"]*100:+6.2f} pp   t = {pd["t"]:+6.2f}')

    n_cells = len({(w.get('lat'), w.get('lon360')) for w in winds})
    print(f'\nepisode coverage: {n_cells}/{args.episodes} distinct grid cells')
    print(f'first episode: {winds[0].get("startISO")} @ '
          f'lat {winds[0].get("lat")}, lon {winds[0].get("lon360")}')

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f'\nwrote {args.out}')
    print('ERA5_EVAL_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
