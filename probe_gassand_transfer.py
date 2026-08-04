"""
Multi-seed transfer probe for the gassand env: the rigorous deterministic-vs-
realism compare that the single-seed media cannot give (one realism episode's
TWR50 swings ±35pp with the wind draw — seed 42 calm rendered 44% against
R_Gassand's ~21% in-run mean).

Evaluates fixed checkpoints (no retraining) in two environments:
  deterministic — no realism flags, the wind the demonstrator trained on
  realism       — R's 4 per-episode flags (wind_phase_jitter, wind_episode_noise,
                  wind_param_jitter, domain_rand), the wind R_Gassand trained on

Same protocol as probe_realism_transfer.py (the H–T lineage's clean probe):
3 presets × N seeds per mode, composite = 0.5·mean + 0.5·worst-preset,
degradation reported in pp. Two gassand-specific additions:

  * The realism port left the spawn RNG untouched (dedicated streams at
    seed+424243 / seed+848487), so seed i gives the identical spawn in both
    modes — the probe therefore also reports the PAIRED per-seed delta, which
    cancels spawn luck out of the transfer number.
  * End-of-episode reserves (helium/sand) are averaged per cell, since the
    resource economy is the point of this env and the input a resource-aware
    reward calibration needs.

Policies probed (any subset via --policies):
  heuristic — built-in wind-follower (no weights)
  gassand   — weights/dqn_gassand_w00.pt   (trained deterministic)
  r_gassand — weights/dqn_r_gassand_w00.pt (trained under realism)

Usage:
    python probe_gassand_transfer.py                        # all 3, 10 seeds
    python probe_gassand_transfer.py --policies r_gassand --seeds 5
    python probe_gassand_transfer.py --out probe_gassand_transfer.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from replay_gassand import PRESETS, load_agent, run_episode

POLICY_WEIGHTS = {
    'heuristic':   None,
    'gassand':     'weights/dqn_gassand_w00.pt',
    'r_gassand':   'weights/dqn_r_gassand_w00.pt',
    'res_gassand': 'weights/dqn_res_gassand_w00.pt',   # resource-aware reward arm
}

REALISM_FLAGS = {
    'wind_phase_jitter':  True,
    'wind_episode_noise': True,
    'wind_param_jitter':  True,
    'domain_rand':        True,
}

MODES = {'deterministic': None, 'realism': REALISM_FLAGS}


def composite_score(per_preset_means: dict[str, float]) -> float:
    vals = list(per_preset_means.values())
    return 0.5 * (sum(vals) / len(vals)) + 0.5 * min(vals)


def probe_policy(name: str, agent, seeds: int, duration_s: float) -> dict:
    """mode -> preset -> {'twr50': [...], 'he_left': [...], 'sand_left': [...]}"""
    out = {}
    for mode, flags in MODES.items():
        out[mode] = {}
        for preset in PRESETS:
            cell = {'twr50': [], 'he_left': [], 'sand_left': []}
            for i in range(seeds):
                seed = 42 + i * 1_000_003
                traj = run_episode(preset, duration_s, seed, agent=agent, flags=flags)
                cell['twr50'].append(traj['twr50'])
                cell['he_left'].append(traj['he_left'])
                cell['sand_left'].append(traj['sand_left'])
            out[mode][preset] = cell
            m  = statistics.mean(cell['twr50'])
            sd = statistics.stdev(cell['twr50']) if seeds > 1 else 0.0
            print(f'{name:10s} {mode:13s} {preset:13s} '
                  f'TWR50 {m*100:5.1f}% ± {sd*100:4.1f}  '
                  f'He {statistics.mean(cell["he_left"]):5.2f}kg  '
                  f'sand {statistics.mean(cell["sand_left"]):5.2f}kg  '
                  f'({seeds} seeds)', flush=True)
    return out


def summarize(name: str, results: dict, seeds: int) -> dict:
    summary = {'modes': {}}
    for mode in MODES:
        means = {p: statistics.mean(results[mode][p]['twr50']) for p in PRESETS}
        summary['modes'][mode] = {
            'per_preset_twr50': means,
            'score': composite_score(means),
            'mean_he_left':   {p: statistics.mean(results[mode][p]['he_left'])   for p in PRESETS},
            'mean_sand_left': {p: statistics.mean(results[mode][p]['sand_left']) for p in PRESETS},
            'raw_twr50': {p: results[mode][p]['twr50'] for p in PRESETS},
        }
    det = summary['modes']['deterministic']['score']
    rea = summary['modes']['realism']['score']
    summary['degradation_pp'] = (det - rea) * 100

    # Paired per-seed delta (same seed = same spawn in both modes), pooled
    # across presets: mean ± sd of (deterministic − realism) per episode.
    paired = []
    for p in PRESETS:
        d = results['deterministic'][p]['twr50']
        r = results['realism'][p]['twr50']
        paired.extend(di - ri for di, ri in zip(d, r))
    summary['paired_delta_pp'] = {
        'mean': statistics.mean(paired) * 100,
        'sd':   (statistics.stdev(paired) if len(paired) > 1 else 0.0) * 100,
        'n':    len(paired),
    }

    print(f'\n{name}: composite deterministic {det*100:.1f}%  realism {rea*100:.1f}%  '
          f'degradation {summary["degradation_pp"]:+.1f}pp  '
          f'(paired Δ {summary["paired_delta_pp"]["mean"]:+.1f} ± '
          f'{summary["paired_delta_pp"]["sd"]:.1f}pp, n={len(paired)})\n', flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policies', nargs='+', default=list(POLICY_WEIGHTS),
                    choices=list(POLICY_WEIGHTS))
    ap.add_argument('--seeds', type=int, default=10, help='episode seeds per cell')
    ap.add_argument('--duration-h', type=float, default=72.0)
    ap.add_argument('--out', default=None, help='optional JSON output path')
    args = ap.parse_args()

    duration_s = args.duration_h * 3600
    report = {'seeds': args.seeds, 'duration_h': args.duration_h, 'policies': {}}
    for name in args.policies:
        w = POLICY_WEIGHTS[name]
        if w and not Path(w).exists():
            print(f'{name}: no checkpoint at {w} — skipping')
            continue
        agent = load_agent(Path(w)) if w else None
        results = probe_policy(name, agent, args.seeds, duration_s)
        report['policies'][name] = {'weights': w, **summarize(name, results, args.seeds)}

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
