#!/usr/bin/env python3
"""
demo_gassand.py — Exercise the helium/sand gas-balloon env variant.

Runs the zero-pressure gas-balloon model (server_version='gassand') through the
normal BalloonEnv IPC bridge and prints altitude / vertical-velocity / reserve
traces, demonstrating the three things the model is built to capture:

  1. FLOAT (release nothing) roughly holds altitude.
  2. Release amount → drift speed: a tiny release drifts slowly, a large one
     climbs/dives fast — in BOTH directions (drop sand = up, vent helium = down).
  3. Reserves are FINITE and irreversible: keep venting and the gas runs out.

Action ladder (11 discrete), low index = up:
    0..4  drop sand   (fast up  → slow up)
    5     FLOAT
    6..10 vent helium (slow down → fast down)

Run:  python demo_gassand.py
"""
from __future__ import annotations

from balloon_env import BalloonEnv

# Action indices (mirror the ACTIONS table in balloon_env_server_gassand.mjs).
# 0..4 drop sand (fast→slow up), 5 FLOAT, 6..10 vent helium (slow→fast down).
FLOAT      = 5
A_HE_FAST  = 10                      # largest helium vent (fastest descent)

SAND_LEVELS   = [4, 3, 2, 1, 0]      # slow → fast (drop sand, rise)
HELIUM_LEVELS = [6, 7, 8, 9, 10]     # slow → fast (vent helium, sink)
LABELS = {4: "sand 0.005kg", 3: "sand 0.02kg", 2: "sand 0.08kg", 1: "sand 0.30kg", 0: "sand 1.0kg",
          6: "He 0.0008kg", 7: "He 0.0032kg", 8: "He 0.0128kg", 9: "He 0.0481kg", 10: "He 0.1604kg"}


def release_then_settle(env, action, settle_steps=18):
    """
    Apply `action` once (recording the peak vertical speed the env reports for
    that decision interval), then FLOAT to a new steady altitude.
    Returns (peak_vv, alt_before, alt_settled).
    """
    _, _, _, info0 = env.step(action)
    peak = info0["vv_peak_m_s"]
    a_before = info0["alt_m"]
    last = info0
    for _ in range(settle_steps):
        _, _, done, last = env.step(FLOAT)
        if done:
            break
    return peak, a_before, last["alt_m"]


def main():
    print("=== gas-balloon (helium/sand) env — end-to-end demo ===\n")

    # ── 1. FLOAT holds ───────────────────────────────────────────────────────
    env = BalloonEnv(preset="calm", duration_s=6 * 3600, seed=7, server_version="gassand")
    print(f"state dim = {env.observation_space_shape[0]}   n_actions = {env.n_actions}")
    s = env.reset()
    print(f"initial state[3]=alt_frac {s[3]:.3f}  state[4]=vv {s[4]*2.5:+.3f}  "
          f"He-gauge {s[5]:.3f}  sand-gauge {s[6]:.3f}")

    alts = []
    for _ in range(24):                      # 24 × 5 min = 2 h of FLOAT
        s, _, _, info = env.step(FLOAT)
        alts.append(info["alt_m"])
    print(f"\n[1] FLOAT 2 h: alt {alts[0]:.0f} → {alts[-1]:.0f} m  "
          f"(span {max(alts) - min(alts):.0f} m)  He {info['helium_kg']:.3f}  sand {info['sand_kg']:.2f}")
    env.close()

    # ── 2. Release amount → velocity, both directions ────────────────────────
    print("\n[2] release amount → peak vertical speed  (each from a fresh, settled float)")
    print("      DROP SAND → up                       VENT HELIUM → down")
    for su, hd in zip(SAND_LEVELS, HELIUM_LEVELS):
        env = BalloonEnv(preset="calm", duration_s=6 * 3600, seed=7, server_version="gassand")
        env.reset()
        for _ in range(12):                  # settle to float first
            env.step(FLOAT)
        pk_s, a0_s, a1_s = release_then_settle(env, su)
        env.close()

        env = BalloonEnv(preset="calm", duration_s=6 * 3600, seed=7, server_version="gassand")
        env.reset()
        for _ in range(12):
            env.step(FLOAT)
        pk_h, a0_h, a1_h = release_then_settle(env, hd)
        env.close()

        print(f"   {LABELS[su]:>11}: {pk_s:+.3f} m/s (float {a0_s:.0f}→{a1_s:.0f}m)   "
              f"{LABELS[hd]:>11}: {pk_h:+.3f} m/s (float {a0_h:.0f}→{a1_h:.0f}m)")

    # ── 3. Finite reserves: vent helium until it runs out ────────────────────
    env = BalloonEnv(preset="calm", duration_s=48 * 3600, seed=7, server_version="gassand")
    env.reset()
    steps = 0
    info = None
    while True:
        s, _, done, info = env.step(A_HE_FAST)   # keep venting the largest amount
        steps += 1
        if info["helium_kg"] <= 1e-6 or done or steps > 2000:
            break
    print(f"\n[3] vent He (0.16 kg/step) until empty: {steps} steps, "
          f"He left {info['helium_kg']:.4f} kg, vented {info['helium_vented_kg']:.2f} kg, "
          f"alt {info['alt_m']:.0f} m")
    env.close()

    print("\nDone. Physics: js/balloon_gassand.js   Server: servers/balloon_env_server_gassand.mjs")


if __name__ == "__main__":
    main()
