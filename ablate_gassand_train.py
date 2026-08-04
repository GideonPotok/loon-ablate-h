"""
ablate_gassand_train.py — Train a QR-DQN station-keeper on the helium/sand
gas-balloon env (server_version='gassand').

WHY THIS EXISTS
  The lineage's favoured policies (N, R, …) cannot run on the gassand env: they
  were trained with a 20/24-dim state and a 17-way target-altitude action head,
  whereas gassand emits a 21-dim state (two finite-reserve gauges) and a 11-way
  one-way RELEASE ladder (drop sand → rise, vent helium → sink). So a learned
  gassand policy has to be trained from scratch. This is that trainer — a
  single-process demonstrator, NOT a lineage ablation (different env → its scores
  are not comparable to H–T; compare only to the built-in heuristic baseline).

RECIPE
  Reuses N/R's feedforward recipe verbatim (γ=0.99, target_update 25, lr 1e-4,
  batch 64, n-step 3, PER, [128,64], n_quantiles 1, 2 grad-steps/env-step,
  recovery spawn) — only the env, state_dim (21), and action head (11) change.
  Reward is the gassand server's built-in station-keeping shape; it does NOT yet
  penalise release, so the finite reserves matter only through survival (running
  a reserve dry costs altitude control). A resource-aware reward is the deferred
  next step (see docs/architecture/gassand-env.md).

  Curriculum + episode budget are trimmed vs R (3600 eps) so this finishes on a
  laptop CPU in a couple of hours. Best-by-eval checkpoint is written to
  weights/dqn_gassand_w00.pt after every eval, so a partial run is still usable:
      python replay_gassand.py --weight weights/dqn_gassand_w00.pt --tag learned

Usage:
    python ablate_gassand_train.py                 # full trimmed run
    python ablate_gassand_train.py --episodes 300  # cap episodes (smoke test)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from qr_agent import QRAgent, QRConfig
from replay_buffer import PrioritizedReplayBuffer, NStepAccumulator
from balloon_env import BalloonEnv

# ── Trimmed curriculum (vs R's 3600 eps) ───────────────────────────────────────
CURRICULUM = [
    {'episodes':  80, 'duration_s': 3600 *  2, 'label':  '2h'},
    {'episodes': 220, 'duration_s': 3600 *  6, 'label':  '6h'},
    {'episodes': 220, 'duration_s': 3600 * 12, 'label': '12h'},
    {'episodes': 240, 'duration_s': 3600 * 24, 'label': '24h'},
    {'episodes': 140, 'duration_s': 3600 * 48, 'label': '48h'},
    {'episodes': 100, 'duration_s': 3600 * 72, 'label': '72h'},
]
TOTAL_EPS       = sum(t['episodes'] for t in CURRICULUM)   # 1000
PRESETS         = ['tropical', 'strong-shear', 'calm']
EVAL_EVERY      = 100
EVAL_RUNS       = 2
EVAL_DURATION_S = 3600 * 72
SERVER_VERSION  = 'gassand'

# Recovery spawn (carried from ablation J / R).
RECOVERY_SPAWN_PROB    = 0.30
RECOVERY_SPAWN_MIN_KM  = 150.0
RECOVERY_SPAWN_MAX_KM  = 500.0
RECOVERY_SPAWN_MIN_DUR = 3600 * 24

BASE_CONFIG = QRConfig(
    state_dim          = 21,      # gassand: 2 reserve gauges instead of 1 ballast gauge
    hidden_sizes       = [128, 64],
    action_count       = 11,      # gassand release ladder
    n_quantiles        = 1,
    huber_kappa        = 1.0,
    learning_rate      = 1e-4,
    optimizer          = 'adam',
    gamma              = 0.99,
    epsilon_start      = 1.0,
    epsilon_end        = 0.03,
    epsilon_decay      = 0.9988,
    target_update_freq = 25,
    replay_capacity    = 100_000,
    batch_size         = 64,
    n_step             = 3,
    per_alpha          = 0.6,
    per_beta0          = 0.4,
    per_beta_anneal    = 1e-4,
    cvar_alpha         = 1.0,
    train_batches_per_step = 2,
    device             = 'cpu',
    use_reward_fix     = False,
    use_shaping        = False,
    use_expanded_state = False,
    use_recurrent      = False,
    use_options        = False,
)

WEIGHTS_DIR    = Path(__file__).parent / 'weights'
WEIGHTS_PREFIX = 'dqn_gassand'
LOG_PATH       = Path('/tmp/train_gassand.log')


def _tier_at(ep: int) -> dict:
    cum = 0
    for tier in CURRICULUM:
        cum += tier['episodes']
        if ep < cum:
            return tier
    return CURRICULUM[-1]


def _eval(agent: QRAgent, ep: int, seed: int, n_runs: int, duration_s: float) -> dict:
    per_preset, all_scores = {}, []
    worst_preset, worst_twr = None, float('inf')
    for pi, preset in enumerate(PRESETS):
        scores = []
        for r in range(n_runs):
            eval_seed = seed + 1_000_000 + ep * 1000 + pi * 17 + r
            env = BalloonEnv(preset=preset, duration_s=duration_s, seed=eval_seed,
                             server_version=SERVER_VERSION)
            state, done, twr50 = env.reset(), False, 0.0
            while not done:
                action = agent.select_action(state, greedy=True)
                state, _, done, info = env.step(action)
                twr50 = info.get('twr50', twr50)
            scores.append(twr50)
            env.close()
        mean_p = float(np.mean(scores))
        per_preset[preset] = mean_p
        all_scores.extend(scores)
        if mean_p < worst_twr:
            worst_twr, worst_preset = mean_p, preset
    mean_twr50 = float(np.mean(all_scores))
    return {'score': 0.5 * mean_twr50 + 0.5 * worst_twr, 'mean': mean_twr50,
            'worst': worst_twr, 'worst_preset': worst_preset, 'per_preset': per_preset}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=0, help='cap episodes (0 = full curriculum)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, 'w', buffering=1)
    def tee(line: str):
        print(line, flush=True); log.write(line + '\n')

    seed   = args.seed
    config = replace(BASE_CONFIG, seed=seed)
    n_eps  = min(TOTAL_EPS, args.episodes) if args.episodes > 0 else TOTAL_EPS

    agent   = QRAgent(config)
    per_buf = PrioritizedReplayBuffer(config.replay_capacity, config.per_alpha,
                                      config.per_beta0, seed=seed + 1)
    n_acc   = NStepAccumulator(config.n_step, config.gamma, per_buf)
    rng     = np.random.default_rng(seed * 31 + 7919)
    n_params = sum(p.numel() for p in agent.policy_net.parameters())

    tee('═' * 78)
    tee('GASSAND demonstrator — QR-DQN station-keeper on the helium/sand env')
    tee('═' * 78)
    tee(f'Network:    {config.state_dim} → {" → ".join(map(str, config.hidden_sizes))} '
        f'→ {config.action_count}   ({n_params:,} params)')
    tee('Curriculum: ' + '  →  '.join(f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM)
        + f'   total {n_eps} eps')
    tee(f'Recipe:     γ={config.gamma} target_update={config.target_update_freq} '
        f'lr={config.learning_rate} batch={config.batch_size} n_step={config.n_step} PER')
    tee(f'Server:     {SERVER_VERSION}  (built-in station-keeping reward; NO resource cost yet)')
    tee(f'Checkpoint: {WEIGHTS_DIR / (WEIGHTS_PREFIX + "_w00.pt")}  (best-by-eval, rewritten each eval)')
    tee('─' * 78)

    best_score, best_episode = -float('inf'), -1
    start_ts = time.time()

    for ep in range(n_eps):
        tier    = _tier_at(ep)
        preset  = PRESETS[ep % len(PRESETS)]
        ep_seed = int(rng.integers(1_000_000_000))
        spawn_km = None
        if tier['duration_s'] >= RECOVERY_SPAWN_MIN_DUR and rng.random() < RECOVERY_SPAWN_PROB:
            spawn_km = float(rng.uniform(RECOVERY_SPAWN_MIN_KM, RECOVERY_SPAWN_MAX_KM))

        env = BalloonEnv(preset=preset, duration_s=tier['duration_s'], seed=ep_seed,
                         server_version=SERVER_VERSION)
        state = env.reset(spawn_offset_km=spawn_km)
        n_acc.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            n_acc.push(state, action, reward, next_state, done)
            n_acc.flush_to_buffer(next_state, episode_done=done)
            for _ in range(config.train_batches_per_step):
                agent.train_batch(per_buf)
            state = next_state
        env.close()
        agent.decay_epsilon()

        if (ep + 1) % EVAL_EVERY == 0 or ep == n_eps - 1:
            ev = _eval(agent, ep, seed, EVAL_RUNS, EVAL_DURATION_S)
            is_best = ev['score'] > best_score
            if is_best:
                best_score, best_episode = ev['score'], ep
                torch.save(agent.state_dict(), WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_w00.pt')
                (WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_w00.json').write_text(json.dumps({
                    'best_score': best_score, 'best_episode': best_episode,
                    'best_per_preset': ev['per_preset'], 'server_version': SERVER_VERSION}))
            m, s = divmod(int(time.time() - start_ts), 60)
            pp = ev['per_preset']
            tee(f'  {m:3d}m{s:02d}s  ep {ep:4d} [{tier["label"]:>3s}]  score {ev["score"]*100:5.1f}%'
                f'  mean {ev["mean"]*100:5.1f}%  worst({ev["worst_preset"]:<12}) {ev["worst"]*100:5.1f}%'
                f'  trop {pp["tropical"]*100:5.1f}%  shear {pp["strong-shear"]*100:5.1f}%'
                f'  calm {pp["calm"]*100:5.1f}%  ε {agent.epsilon:.3f}' + ('  ★' if is_best else ''))

    m, s = divmod(int(time.time() - start_ts), 60)
    tee('─' * 78)
    tee(f'DONE  {m}m{s:02d}s   best ep {best_episode}   best score {best_score*100:.1f}%')
    tee(f'Checkpoint: {WEIGHTS_DIR / (WEIGHTS_PREFIX + "_w00.pt")}')
    log.close()


if __name__ == '__main__':
    main()
