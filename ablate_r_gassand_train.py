"""
ablate_r_gassand_train.py — R_Gassand: R's realism bundle on the gassand physics.

R (server_version='v2') answered "how much skill survives when the wind clock is
broken and the oracle time-features are removed". R_Gassand asks the same
question for the helium/sand gas-balloon control problem: train + eval the
gassand QR-DQN under the SAME four per-episode realism flags R used
  wind_phase_jitter · wind_episode_noise · wind_param_jitter · domain_rand
(ported into the gassand server's handleReset, mirroring the v2 server).

Relationship to the other trainers:
  * ablate_gassand_train.py — DETERMINISTIC gassand wind (N-like env), plain
    station-keeping reward. The floor/baseline for the gassand family.
  * this (R_Gassand)        — same recipe + reward, but the R realism bundle in
    train AND eval. The env-invariance / transfer arm for gassand.
  Reward stays the gassand server's plain station-keeping shape (Option A): the
  realism bundle is R's one information-structure change; reward shaping is
  orthogonal and deliberately NOT ported (keep one change at a time). A
  resource-aware reward remains the separate deferred step.

NOT comparable to R/H–T: different physics, state (21-d) and action head (11).
Compare only WITHIN the gassand family (vs the deterministic demonstrator), and
— once both exist — via a transfer probe (deterministic-trained net evaluated
under realism, and vice-versa), the gassand analogue of probe_realism_transfer.

Usage:
    python ablate_r_gassand_train.py                 # full trimmed run
    python ablate_r_gassand_train.py --episodes 300  # cap (smoke test)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from qr_agent import QRAgent
from replay_buffer import PrioritizedReplayBuffer, NStepAccumulator
from balloon_env import BalloonEnv

# Single-source the recipe from the deterministic gassand trainer — only the
# realism flags, checkpoint prefix and log path differ.
from ablate_gassand_train import (
    CURRICULUM, PRESETS, EVAL_EVERY, EVAL_RUNS, EVAL_DURATION_S, SERVER_VERSION,
    RECOVERY_SPAWN_PROB, RECOVERY_SPAWN_MIN_KM, RECOVERY_SPAWN_MAX_KM,
    RECOVERY_SPAWN_MIN_DUR, BASE_CONFIG, _tier_at,
)

TOTAL_EPS      = sum(t['episodes'] for t in CURRICULUM)
WEIGHTS_DIR    = Path(__file__).parent / 'weights'
WEIGHTS_PREFIX = 'dqn_r_gassand'
LOG_PATH       = Path('/tmp/train_r_gassand.log')


def _realism_flags() -> dict:
    """The four R realism flags — applied in BOTH training and eval."""
    return {
        'wind_phase_jitter':  True,   # φ_igw, φ_pw ~ U[0,2π)
        'wind_episode_noise': True,   # episode seed → background-noise hash
        'wind_param_jitter':  True,   # IGW/PW amplitude × logU[0.7,1.4]
        'domain_rand':        True,   # degrader σ-scale logU[0.5,2.0] + lag U[0,6h]
    }


def _eval(agent: QRAgent, ep: int, seed: int, n_runs: int, duration_s: float) -> dict:
    per_preset, all_scores = {}, []
    worst_preset, worst_twr = None, float('inf')
    flags = _realism_flags()
    for pi, preset in enumerate(PRESETS):
        scores = []
        for r in range(n_runs):
            eval_seed = seed + 1_000_000 + ep * 1000 + pi * 17 + r
            env = BalloonEnv(preset=preset, duration_s=duration_s, seed=eval_seed,
                             server_version=SERVER_VERSION, flags=flags)
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
    flags  = _realism_flags()

    agent   = QRAgent(config)
    per_buf = PrioritizedReplayBuffer(config.replay_capacity, config.per_alpha,
                                      config.per_beta0, seed=seed + 1)
    n_acc   = NStepAccumulator(config.n_step, config.gamma, per_buf)
    rng     = np.random.default_rng(seed * 31 + 7919)
    n_params = sum(p.numel() for p in agent.policy_net.parameters())

    tee('═' * 78)
    tee('R_GASSAND — gassand QR-DQN under R\'s realism bundle (train + eval)')
    tee('═' * 78)
    tee(f'Network:    {config.state_dim} → {" → ".join(map(str, config.hidden_sizes))} '
        f'→ {config.action_count}   ({n_params:,} params)')
    tee('Curriculum: ' + '  →  '.join(f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM)
        + f'   total {n_eps} eps')
    tee(f'Recipe:     γ={config.gamma} target_update={config.target_update_freq} '
        f'lr={config.learning_rate} batch={config.batch_size} n_step={config.n_step} PER (N/R recipe)')
    tee(f'Realism:    phase jitter + episode noise + amp jitter + domain-rand (ALL ON, train+eval)')
    tee(f'Server:     {SERVER_VERSION}  (plain station-keeping reward; NO shaping, NO resource cost)')
    tee(f'Checkpoint: {WEIGHTS_DIR / (WEIGHTS_PREFIX + "_w00.pt")}  (best-by-eval)')
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
                         server_version=SERVER_VERSION, flags=flags)
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
                    'best_per_preset': ev['per_preset'], 'server_version': SERVER_VERSION,
                    'realism': True}))
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
