"""
Transfer probe: how brittle are the trained clock-reader policies when the
wind clock breaks?

Evaluates an existing trained checkpoint (no retraining) in two environments:
  legacy   — the deterministic wind the policy was trained on
  realism  — per-episode IGW/PW phase jitter + episode-seeded background noise
             + amplitude jitter + domain-randomized forecast degradation
             (wind_phase_jitter / wind_episode_noise / wind_param_jitter /
             domain_rand flags in servers/balloon_env_server_v2.mjs)

The state layout is unchanged (for feature-bearing ablations the Fourier time
features are still emitted — they just no longer predict the wind), so the
degradation delta isolates how much of the policy's skill was reading the
simulator's clock rather than the wind itself. This is the cross-environment
bridge between the H–Q lineage and the realism-era ablations (R, S, T, …).

Usage:
    python probe_realism_transfer.py --ablation n --weights weights/dqn_ablate_n.pt
    python probe_realism_transfer.py --ablation q --weights weights/dqn_ablate_q.pt \
        --seeds 10 --duration-h 72
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from replay import ABLATION_AGENT_KWARGS, load_agent, run_episode

PRESETS = ['tropical', 'strong-shear', 'calm']

# Legacy (training-time) env flags per ablation. N uses L's flags — same env,
# only optimizer/curriculum differed. Q trained with the same flag set.
LEGACY_FLAGS = {
    'l': {
        'use_reward_fix': True, 'use_shaping': True, 'use_expanded_state': False,
        'use_time_features': True, 'shaping_beta': 0.5, 'shaping_gamma': 0.97,
        'terminal_twr_bonus': 50.0, 'shaping_linear': False, 'shaping_D_max': 500_000.0,
    },
}
LEGACY_FLAGS['n'] = dict(LEGACY_FLAGS['l'])
LEGACY_FLAGS['m'] = dict(LEGACY_FLAGS['l'])
LEGACY_FLAGS['q'] = dict(LEGACY_FLAGS['l'])

REALISM_FLAGS = {
    'wind_phase_jitter':  True,
    'wind_episode_noise': True,
    'wind_param_jitter':  True,
    'domain_rand':        True,
}


def composite_score(per_preset_means: dict[str, float]) -> float:
    vals = list(per_preset_means.values())
    return 0.5 * (sum(vals) / len(vals)) + 0.5 * min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', default='n', choices=sorted(LEGACY_FLAGS))
    ap.add_argument('--weights', default=None,
                    help='checkpoint path (default: weights/dqn_ablate_<letter>.pt)')
    ap.add_argument('--seeds', type=int, default=10, help='episode seeds per preset')
    ap.add_argument('--duration-h', type=float, default=72.0)
    ap.add_argument('--out', default=None, help='optional JSON output path')
    args = ap.parse_args()

    weights = Path(args.weights or f'weights/dqn_ablate_{args.ablation}.pt')
    agent = load_agent(weights, ABLATION_AGENT_KWARGS.get(args.ablation))
    duration_s = args.duration_h * 3600

    results = {}   # mode -> preset -> [twr50 per seed]
    for mode, extra in (('legacy', {}), ('realism', REALISM_FLAGS)):
        flags = {**LEGACY_FLAGS[args.ablation], **extra}
        results[mode] = {}
        for preset in PRESETS:
            scores = []
            for i in range(args.seeds):
                seed = 42 + i * 1_000_003
                traj = run_episode(agent, preset, duration_s, seed,
                                   server_version='v2', flags=flags)
                scores.append(traj['twr50'])
            results[mode][preset] = scores
            mean = statistics.mean(scores)
            sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f'{mode:8s} {preset:13s} TWR50 {mean*100:5.1f}% ± {sd*100:4.1f} '
                  f'({args.seeds} seeds)', flush=True)

    print()
    summary = {'ablation': args.ablation, 'weights': str(weights),
               'seeds': args.seeds, 'duration_h': args.duration_h, 'modes': {}}
    for mode in results:
        means = {p: statistics.mean(results[mode][p]) for p in PRESETS}
        score = composite_score(means)
        summary['modes'][mode] = {'per_preset_twr50': means, 'score': score,
                                  'raw': results[mode]}
        print(f'{mode:8s} score = 0.5·mean + 0.5·worst = {score*100:.1f}%')
    delta = summary['modes']['legacy']['score'] - summary['modes']['realism']['score']
    summary['degradation_pp'] = delta * 100
    print(f'\ndegradation (legacy − realism): {delta*100:+.1f} pp')

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
