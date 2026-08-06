#!/usr/bin/env python3
"""
eval_era5_gassand.py — how does R_Gassand do on real ERA5 wind?

The gassand twin of scratch/eval_era5.py. Same protocol, so the numbers sit next
to the v2 readout in weights/era5_eval_r_s_100ep.json:

  * N episodes (default 100), 72 h each, seeds 42 + i·1,000,003.
  * Every policy sees the SAME episodes. The server draws the ERA5 cell and
    start time from makeRng(seed + 313131) and the spawn from makeRng(seed),
    neither of which depends on the flags, so a shared seed is a shared episode
    and the comparison is paired rather than merely averaged. The script asserts
    this rather than trusting it.
  * 100 rather than 30: the TWR50 distribution is badly skewed, and on v2 thirty
    episodes could not separate R from the heuristic.

Policies:
  r_gassand   — the favored gassand checkpoint, trained under realism
  res_gassand — the resource-aware-reward arm (trades ~3 pp for ~18 kg of sand
                on the presets; real wind may change that trade)
  gassand     — the deterministic-trained demonstrator
  heuristic   — the JS navigator. THE control. If it collapses too, the ceiling
                is the environment and not the policy.
  float       — FLOAT every step (action 5, release nothing). The do-nothing
                floor. Note this is NOT v2's mid-band 8: gassand's action table
                is 5 sand drops, FLOAT, 5 helium vents.

All policies run under R's realism bundle, matching probe_gassand_transfer.py.
Worth knowing what that means on ERA5: wind_phase_jitter, wind_episode_noise and
wind_param_jitter modify the *synthetic* wind generator and are inert here,
because the archive supplies the wind. Only domain_rand still bites, on the
sensing stack. use_resource_reward is left off for everyone — it changes only
the reward, never the physics or the state, so it cannot move TWR50, and leaving
it off keeps every policy on a literally identical trajectory basis.

Resources are reported at episode end because finite consumables are the whole
point of this env: a policy that scores by dumping everything in the first six
hours is a different finding from one that paces itself.

Usage:
    python scratch/eval_era5_gassand.py "/Volumes/Gideon SDD/data/era5_json" \
        --episodes 100 --out weights/era5_eval_gassand_100ep.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from replay_gassand import load_agent, run_episode  # noqa: E402

DURATION_S = 72 * 3600
SEED0, SEED_STRIDE = 42, 1_000_003

REALISM_FLAGS = {
    'wind_phase_jitter':  True,
    'wind_episode_noise': True,
    'wind_param_jitter':  True,
    'domain_rand':        True,
}

# gassand's release ladder is [5 sand drops, FLOAT, 5 helium vents], so FLOAT is
# index 5 — SAND_RELEASE_KG.length, not v2's mid-band 8.
FLOAT_ACTION = 5

POLICY_WEIGHTS = {
    'r_gassand':   'weights/dqn_r_gassand_w00.pt',
    'res_gassand': 'weights/dqn_res_gassand_w00.pt',
    'gassand':     'weights/dqn_gassand_w00.pt',
    'heuristic':   None,     # agent=None -> env.heuristic_step()
    'float':       'FLOAT',  # sentinel, handled below
}


class FloatAgent:
    """Releases nothing, ever. The do-nothing floor."""
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
    return {'delta': m, 't': t, 'sd': sd, 'n': len(d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('era5_dir')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--policies', nargs='+', default=list(POLICY_WEIGHTS),
                    choices=list(POLICY_WEIGHTS))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    seeds = [SEED0 + i * SEED_STRIDE for i in range(args.episodes)]

    print(f'ERA5 gassand evaluation — {args.episodes} episodes x 72 h')
    print(f'archive: {args.era5_dir}')
    print(f'seeds {seeds[0]} .. {seeds[-1]} (shared across policies -> paired)\n')

    results: dict = {
        'episodes': args.episodes, 'duration_h': 72, 'era5_dir': args.era5_dir,
        'server_version': 'gassand', 'seeds': seeds,
        'flags': REALISM_FLAGS, 'float_action': FLOAT_ACTION, 'policies': {},
    }
    per_episode: dict[str, list[float]] = {}
    cells: dict[str, list[tuple]] = {}

    for name in args.policies:
        w = POLICY_WEIGHTS[name]
        if w not in (None, 'FLOAT') and not (REPO / w).exists():
            print(f'{name}: no checkpoint at {w} — skipping')
            continue
        if w == 'FLOAT':
            agent = FloatAgent()
        elif w is None:
            agent = None            # built-in JS navigator heuristic
        else:
            agent = load_agent(REPO / w)

        scores, he_left, sand_left, he_used, sand_used, dists = [], [], [], [], [], []
        mine: list[tuple] = []
        t0 = time.time()
        for i, seed in enumerate(seeds):
            traj = run_episode('tropical', DURATION_S, seed, agent=agent,
                               flags=REALISM_FLAGS, wind_source='era5',
                               era5_dir=args.era5_dir)
            scores.append(traj['twr50'])
            he_left.append(traj['he_left'])
            sand_left.append(traj['sand_left'])
            he_used.append(traj['he_used'])
            sand_used.append(traj['sand_used'])
            dists.append(float(traj['dist_km'][-1]))
            wmeta = traj['wind']
            mine.append((wmeta.get('lat'), wmeta.get('lon360'), wmeta.get('startUnix')))
            if (i + 1) % 20 == 0:
                print(f'    {name}: {i+1}/{len(seeds)} '
                      f'({time.time()-t0:.0f}s elapsed)', flush=True)

        per_episode[name] = scores
        cells[name] = mine
        s = summarize(scores)
        results['policies'][name] = {
            **s, 'weights': w, 'twr50': scores,
            'final_dist_km': dists,
            'he_left_kg': he_left, 'sand_left_kg': sand_left,
            'he_used_kg': he_used, 'sand_used_kg': sand_used,
            'mean_he_left_kg':    statistics.mean(he_left),
            'mean_sand_left_kg':  statistics.mean(sand_left),
            'mean_he_used_kg':    statistics.mean(he_used),
            'mean_sand_used_kg':  statistics.mean(sand_used),
        }
        print(f'  {name:12s} TWR50 mean {s["mean"]*100:5.2f}%  median {s["median"]*100:5.2f}%  '
              f'sd {s["stdev"]*100:4.1f}pp  range {s["min"]*100:.1f}-{s["max"]*100:.1f}%\n'
              f'  {"":12s} He left {statistics.mean(he_left):6.3f} kg (vented {statistics.mean(he_used):5.3f})  '
              f'sand left {statistics.mean(sand_left):5.2f} kg (dropped {statistics.mean(sand_used):5.2f})  '
              f'final dist {statistics.mean(dists):6.0f} km', flush=True)

    # The pairing is the whole design; if the cells differ it is a lie. Assert.
    names = list(cells)
    ref = cells[names[0]]
    for n in names[1:]:
        if cells[n] != ref:
            bad = [i for i, (a, b) in enumerate(zip(ref, cells[n])) if a != b]
            print(f'\n*** PAIRING BROKEN: {n} saw different episodes than '
                  f'{names[0]} at indices {bad[:5]} ***')
            return 1
    print(f'\npairing verified: all {len(names)} policies flew the identical '
          f'{len(ref)} ERA5 episodes')

    results['episode_cells'] = [
        {'seed': s, 'lat': c[0], 'lon360': c[1], 'startUnix': c[2]}
        for s, c in zip(seeds, ref)
    ]
    n_cells = len({(c[0], c[1]) for c in ref})
    print(f'episode coverage: {n_cells}/{args.episodes} distinct grid cells')

    print(f'\npaired deltas (same {args.episodes} episodes):')
    comparisons = [
        ('r_gassand', 'float'), ('r_gassand', 'heuristic'),
        ('heuristic', 'float'), ('res_gassand', 'r_gassand'),
        ('res_gassand', 'float'), ('gassand', 'float'),
        ('gassand', 'r_gassand'),
    ]
    results['paired'] = {}
    for a, b in comparisons:
        if a not in per_episode or b not in per_episode:
            continue
        pd = paired_delta(per_episode[a], per_episode[b])
        results['paired'][f'{a}_vs_{b}'] = pd
        print(f'  {a:12s} − {b:12s}  {pd["delta"]*100:+6.2f} pp   t = {pd["t"]:+6.2f}')

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f'\nwrote {args.out}')
    print('ERA5_GASSAND_EVAL_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
