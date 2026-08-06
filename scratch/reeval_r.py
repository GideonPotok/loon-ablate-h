"""Clean 10-seed re-eval of Ablation R's winner checkpoint.

Mirrors probe_realism_transfer.py's protocol (same seeds, duration, scoring)
but for the R checkpoint, which trained in the realism env with time features
off. Evaluates both directions of the bridge:
  realism — R's own training env (phase jitter + episode noise + param jitter
            + domain rand)
  legacy  — the deterministic H–Q wind, realism flags off (state stays 20-dim)
"""
import json
import statistics
import sys
from pathlib import Path

WORKTREE = '/Users/gideonpotok/repos/loon-ablate-h/.claude/worktrees/ablate-q-per-step'
sys.path.insert(0, WORKTREE)

from replay import ABLATION_ENV_FLAGS, load_agent, run_episode

WEIGHTS = Path('/Users/gideonpotok/repos/loon-ablate-h/weights/dqn_ablate_r.pt')
OUT = Path('/Users/gideonpotok/.claude/jobs/8bdc873d/tmp/r_reeval.json')
PRESETS = ['tropical', 'strong-shear', 'calm']
SEEDS = 10
DURATION_S = 72 * 3600

agent = load_agent(WEIGHTS, None)   # plain MLP, no arch overrides

realism = dict(ABLATION_ENV_FLAGS['r'])
legacy = {**realism, 'wind_phase_jitter': False, 'wind_episode_noise': False,
          'wind_param_jitter': False, 'domain_rand': False}

results = {}
for mode, flags in (('realism', realism), ('legacy', legacy)):
    results[mode] = {}
    for preset in PRESETS:
        scores = []
        for i in range(SEEDS):
            seed = 42 + i * 1_000_003
            traj = run_episode(agent, preset, DURATION_S, seed,
                               server_version='v2', flags=flags)
            scores.append(traj['twr50'])
        m, sd = statistics.mean(scores), statistics.stdev(scores)
        print(f'{mode:8s} {preset:13s} TWR50 {m*100:5.1f}% ± {sd*100:4.1f}pp '
              f'({SEEDS} seeds)', flush=True)
        results[mode][preset] = scores

for mode in list(results):
    means = {p: statistics.mean(results[mode][p]) for p in PRESETS}
    score = 0.5 * statistics.mean(list(means.values())) + 0.5 * min(means.values())
    results[f'{mode}_composite'] = score
    print(f'{mode:8s} composite (0.5*mean + 0.5*worst) = {score*100:.1f}%', flush=True)

OUT.write_text(json.dumps(results, indent=2))
print('REEVAL_DONE')
