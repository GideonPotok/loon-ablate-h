"""
ablate_res_gassand_train.py — Res_Gassand: the resource-aware reward on top of
R_Gassand's realism bundle.

R_Gassand showed the plain station-keeping reward gives no conservation
incentive: every gassand policy so far (heuristic, deterministic-learned,
R_Gassand) ends its 72 h episode with the sand reserve at zero. Res_Gassand
trains the same net, same recipe, same realism, but with the gassand server's
flag-gated resource-aware reward switched on:

    reward = plain station-keeping
           − sand_cost_per_kg·sand_released − helium_cost_per_kg·He_released
           − depletion_penalty (one-time, per reserve run dry)
           − floor_penalty per step pinned at ALT_MIN
           + terminal_reserve_bonus · mean(He gauge, sand gauge) at episode end

Relationship to the other gassand trainers:
  * ablate_gassand_train.py   — deterministic wind, plain reward (the floor)
  * ablate_r_gassand_train.py — realism bundle, plain reward (R_Gassand)
  * this (Res_Gassand)        — realism bundle + resource-aware reward; the
    one-change-at-a-time step after R_Gassand (reward is the single change).

Checkpoint selection: best-by-eval RETURN composite (0.5·mean + 0.5·worst
preset), not TWR — the return is the trained objective and already folds in
conservation. TWR50 and end-of-episode reserves are logged alongside so the
checkpoint stays comparable to the plain-reward policies via
probe_gassand_transfer.py (which scores TWR + reserves, reward-independent).

Usage:
    python ablate_res_gassand_train.py                 # full trimmed run
    python ablate_res_gassand_train.py --episodes 300  # cap (smoke test)
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
# flags (realism + resource reward), checkpoint prefix and log path differ.
from ablate_gassand_train import (
    CURRICULUM, PRESETS, EVAL_EVERY, EVAL_RUNS, EVAL_DURATION_S, SERVER_VERSION,
    RECOVERY_SPAWN_PROB, RECOVERY_SPAWN_MIN_KM, RECOVERY_SPAWN_MAX_KM,
    RECOVERY_SPAWN_MIN_DUR, BASE_CONFIG, _tier_at,
)

TOTAL_EPS      = sum(t['episodes'] for t in CURRICULUM)
WEIGHTS_DIR    = Path(__file__).parent / 'weights'
WEIGHTS_PREFIX = 'dqn_res_gassand'
LOG_PATH       = Path('/tmp/train_res_gassand.log')

# Resource-reward coefficients, calibrated against the multi-seed probe
# (probe_gassand_transfer.json): under realism the plain-reward policies earn
# roughly 60–200 base return per 72 h episode while spending all 20 kg sand
# (cost 40) and ~3 kg He (cost ~80) — so full-waste costs about one typical
# episode's base return, while the frugal pattern R_Gassand showed in its first
# 30 h (sand 64%, He 90%) costs ~20. FLOAT-forever nets ~terminal bonus only;
# strategic spending that buys radius time still dominates it (1 kg sand = 2
# reward ≈ two in-radius steps).
RESOURCE_REWARD_FLAGS = {
    'use_resource_reward':    True,
    'sand_cost_per_kg':       2.0,
    'helium_cost_per_kg':     25.0,
    'terminal_reserve_bonus': 25.0,
    'depletion_penalty':      25.0,
    'floor_penalty':          0.1,
}


def _flags() -> dict:
    """R's four realism flags + the resource-aware reward — train AND eval."""
    return {
        'wind_phase_jitter':  True,
        'wind_episode_noise': True,
        'wind_param_jitter':  True,
        'domain_rand':        True,
        **RESOURCE_REWARD_FLAGS,
    }


def _eval(agent: QRAgent, ep: int, seed: int, n_runs: int, duration_s: float) -> dict:
    per_ret, per_twr, per_he, per_sand = {}, {}, {}, {}
    flags = _flags()
    for pi, preset in enumerate(PRESETS):
        rets, twrs, hes, sands = [], [], [], []
        for r in range(n_runs):
            eval_seed = seed + 1_000_000 + ep * 1000 + pi * 17 + r
            env = BalloonEnv(preset=preset, duration_s=duration_s, seed=eval_seed,
                             server_version=SERVER_VERSION, flags=flags)
            state, done = env.reset(), False
            ret, twr50, info = 0.0, 0.0, {}
            while not done:
                action = agent.select_action(state, greedy=True)
                state, reward, done, info = env.step(action)
                ret   += reward
                twr50  = info.get('twr50', twr50)
            rets.append(ret)
            twrs.append(twr50)
            hes.append(info.get('helium_kg', 0.0))
            sands.append(info.get('sand_kg', 0.0))
            env.close()
        per_ret[preset]  = float(np.mean(rets))
        per_twr[preset]  = float(np.mean(twrs))
        per_he[preset]   = float(np.mean(hes))
        per_sand[preset] = float(np.mean(sands))
    mean_ret  = float(np.mean(list(per_ret.values())))
    worst_p   = min(per_ret, key=per_ret.get)
    score     = 0.5 * mean_ret + 0.5 * per_ret[worst_p]
    return {'score': score, 'mean_return': mean_ret, 'worst_preset': worst_p,
            'per_return': per_ret, 'per_twr': per_twr,
            'per_he': per_he, 'per_sand': per_sand,
            'mean_twr': float(np.mean(list(per_twr.values()))),
            'mean_he': float(np.mean(list(per_he.values()))),
            'mean_sand': float(np.mean(list(per_sand.values())))}


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
    flags  = _flags()

    agent   = QRAgent(config)
    per_buf = PrioritizedReplayBuffer(config.replay_capacity, config.per_alpha,
                                      config.per_beta0, seed=seed + 1)
    n_acc   = NStepAccumulator(config.n_step, config.gamma, per_buf)
    rng     = np.random.default_rng(seed * 31 + 7919)
    n_params = sum(p.numel() for p in agent.policy_net.parameters())

    tee('═' * 78)
    tee('RES_GASSAND — gassand QR-DQN, realism bundle + resource-aware reward')
    tee('═' * 78)
    tee(f'Network:    {config.state_dim} → {" → ".join(map(str, config.hidden_sizes))} '
        f'→ {config.action_count}   ({n_params:,} params)')
    tee('Curriculum: ' + '  →  '.join(f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM)
        + f'   total {n_eps} eps')
    tee(f'Recipe:     γ={config.gamma} target_update={config.target_update_freq} '
        f'lr={config.learning_rate} batch={config.batch_size} n_step={config.n_step} PER (N/R recipe)')
    tee('Realism:    phase jitter + episode noise + amp jitter + domain-rand (ALL ON, train+eval)')
    tee('Reward:     resource-aware — ' + ', '.join(
        f'{k.replace("_", " ")}={v}' for k, v in RESOURCE_REWARD_FLAGS.items() if k != 'use_resource_reward'))
    tee(f'Selection:  best-by-eval RETURN composite (TWR + reserves logged alongside)')
    tee(f'Checkpoint: {WEIGHTS_DIR / (WEIGHTS_PREFIX + "_w00.pt")}')
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
                    'best_per_return': ev['per_return'], 'best_per_twr': ev['per_twr'],
                    'best_mean_he': ev['mean_he'], 'best_mean_sand': ev['mean_sand'],
                    'server_version': SERVER_VERSION, 'realism': True,
                    'resource_reward': RESOURCE_REWARD_FLAGS}))
            m, s = divmod(int(time.time() - start_ts), 60)
            tee(f'  {m:3d}m{s:02d}s  ep {ep:4d} [{tier["label"]:>3s}]  '
                f'ret {ev["score"]:7.1f}  twr {ev["mean_twr"]*100:5.1f}%  '
                f'He {ev["mean_he"]:5.2f}kg  sand {ev["mean_sand"]:5.2f}kg  '
                f'worst({ev["worst_preset"]:<12}) {ev["per_return"][ev["worst_preset"]]:7.1f}  '
                f'ε {agent.epsilon:.3f}' + ('  ★' if is_best else ''))

    m, s = divmod(int(time.time() - start_ts), 60)
    tee('─' * 78)
    tee(f'DONE  {m}m{s:02d}s   best ep {best_episode}   best return score {best_score:.1f}')
    tee(f'Checkpoint: {WEIGHTS_DIR / (WEIGHTS_PREFIX + "_w00.pt")}')
    log.close()


if __name__ == '__main__':
    main()
