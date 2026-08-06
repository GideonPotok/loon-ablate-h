# Gas-Balloon (Helium/Sand) Env Variant

A variant of the sim env in which altitude is controlled the way a **classic
zero-pressure gas balloon** is flown, instead of the reversible air-ballast pump
used by v1/v2:

- **Drop SAND** → shed mass → **rise**
- **Vent HELIUM** → lose lift → **sink**

Both reserves are **finite and irreversible** — once sand is dropped or gas is
vented it is gone; there is no pump to take either back. The **amount** released
per decision sets the size of the buoyancy imbalance, and drag turns that into a
vertical speed: a *tiny* release drifts *slowly*, a *large* one climbs/dives
*fast* (`v ∝ √amount`).

This is the model requested for the team's real platform. The physics is
deliberately decoupled from reward: by default the reward is the plain v1
station-keeping shape, with the resource economy exposed in `info`. A
**resource-aware reward** is available behind the `use_resource_reward` flag
(see below) — with the flag off, the reward path is byte-identical to the
physics-only baseline.

## Files

| File | Role |
|------|------|
| `js/balloon_gassand.js` | Physics: state, release actuator, buoyancy, sub-stepped integration, equilibrium/ceiling helpers. |
| `js/config.js` → `DEFAULT_GASSAND` / `runtime.gassand` | Platform params (envelope, masses, reserves, release ladder). |
| `servers/balloon_env_server_gassand.mjs` | NDJSON env server: 11-action release ladder, 21-dim state, reuses the v1 wind/sensing stack. |
| `balloon_env.py` → `server_version='gassand'` | Python bridge; sets `n_actions=11`, state dim `21`. |
| `demo_gassand.py` | End-to-end demo of the three core behaviours. |

## The lift model (two regimes from one `min`)

```
displaced_air_mass = min( helium_kg · (M_air/M_he),   ρ_air(alt) · V_env )
net_buoyancy       = (displaced_air_mass − (dry + sand + helium)) · g
```

- **Bubble regime** (envelope not full — low altitude / plentiful gas): the
  `min` picks `helium_kg · M_air/M_he`. Lift is proportional to helium and
  **altitude-independent**. This is the free-expansion regime of a zero-pressure
  balloon below its ceiling — venting gas here directly reduces lift → descent.
- **Superpressure regime** (envelope full — high altitude / less gas): the `min`
  picks `ρ_air · V_env`, which falls with altitude → a **stable float**, exactly
  like the fixed-volume Loon model in `balloon.js`.

The crossover (the **ceiling**) is the altitude where the gas just fills the
envelope. `M_air/M_he ≈ 7.236`, so each kg of helium displaces ~7.2 kg of air
and nets ~6.2 kg of lift after its own weight.

### Why launch must be *balanced*

There is a genuine physical tension: a **passively stable float** requires the
superpressure regime (envelope full), but in that regime venting a little helium
just sheds mass while the displaced volume stays capped — so the balloon would
**rise**, not sink. "Vent helium → down" only holds at/below the ceiling (bubble
regime). You cannot have both a firm passive float *and* small-vent-descends for
a single-gas balloon — that is real zero-pressure-balloon behaviour, not a bug.

So the default launch is **balanced**: `helium·(M_air/M_he − 1) = dry + sand`
(`HELIUM_INIT_KG = 19.24165 kg` for the default masses), which seats the balloon
right at its zero-pressure ceiling (~17.1 km). Consequences:

- **Do-nothing ≈ holds** (the ceiling seats it; the demo shows ±4 m over 2 h).
- **Drop sand → a new, higher, stable float** (rises into superpressure).
- **Vent helium → descends** toward a lower ceiling; keep venting and it sinks to
  the floor. Arresting a descent costs sand — the authentic gas-balloon economy.

The system is effectively a **double integrator with one-way thrusters and
finite fuel** — a clean, well-posed control problem.

### Numerical note

The vertical buoyancy/drag mode is stiff (natural period ~2–3 min). Integrating
it at the 60 s physics tick blows up into a ±MAX_VV oscillation — the same latent
instability lives in `balloon.js`, but v1/v2 hide it behind a bang-bang chase
that only reads 5-min-averaged altitude. Here the vertical velocity *is* the
controlled quantity, so `physicsStep` sub-steps the vertical integration at 5 s
(`VERT_SUBSTEP_S`) for stability. Horizontal advection is applied once per tick.

## Action space (11 discrete), ordered by altitude rate

| idx | action | effect |
|-----|--------|--------|
| 0–4 | drop sand `1.0 / 0.30 / 0.08 / 0.02 / 0.005` kg | fast up → slow up |
| 5   | **FLOAT** (release nothing) | hold |
| 6–10| vent helium `0.0008 / 0.0032 / 0.0128 / 0.0481 / 0.1604` kg | slow down → fast down |

Helium amounts are the paired sand amounts ÷ 6.236 (net lift per kg He), so an
"up" and its mirror "down" produce matched speeds (~±0.08 .. ±1.55 m/s). All
magnitudes live in `DEFAULT_GASSAND` and are tunable. Releasing a reserve you
have run out of is a no-op.

## State vector (21-dim)

Mirrors the v1 compact state but replaces the single ballast gauge with two
resource gauges:

```
[0]     dist / STATION_RADIUS_M
[1,2]   sin(bearing), cos(bearing)
[3]     (alt − altBandLow) / altBandRange           clamped [0,1]
[4]     vv / MAX_VV
[5]     helium_kg / HELIUM_CAPACITY_KG              ← lift-gas fuel gauge
[6]     sand_kg   / SAND_CAPACITY_KG                ← ballast fuel gauge
[7,8]   current-alt wind u,v / 20
[9..20] 4 × (u/20, v/20, σ/10) at 16625/17125/17625/18125 m
```

`info` additionally carries `vv_m_s`, `vv_peak_m_s` (peak vertical speed reached
during the decision interval — the honest "how fast did the release move it"
signal, since fast settles hide the peak at the 5-min cadence), `helium_kg`,
`sand_kg`, `helium_vented_kg`, `sand_dropped_kg`, and the per-step
`helium_released_kg` / `sand_released_kg`.

## Usage

```python
from balloon_env import BalloonEnv
env = BalloonEnv(preset="calm", duration_s=6*3600, seed=7, server_version="gassand")
s = env.reset()                     # (21,)
s, r, done, info = env.step(5)      # 5 = FLOAT; 0..4 sand-up, 6..10 helium-down
env.close()
```

`python demo_gassand.py` prints the three core behaviours (FLOAT holds; release
amount → peak speed both ways; finite depletion).

`replay_gassand.py` rolls out a policy (built-in wind-follower heuristic by
default, or a gassand-trained QR-DQN via `--weight`) and renders a PNG + GIF with
a **finite-reserve depletion panel** (helium & sand draining over time; the GIF
adds live draining fuel-gauge bars).

## Training on this env (deterministic vs realism)

Two QR-DQN trainers ship for the gassand env, both using N/R's feedforward recipe
(γ=0.99, target_update 25, lr 1e-4, batch 64, n-step 3, PER, [128,64], recovery
spawn) and the plain station-keeping reward above:

| script | wind | reward | role |
|--------|------|--------|------|
| `ablate_gassand_train.py`   | deterministic (no realism flags) | plain | N-like floor / demonstrator |
| `ablate_r_gassand_train.py` | R's 4 realism flags, train + eval | plain | **R_Gassand** — transfer/robustness arm |
| `ablate_res_gassand_train.py` | R's 4 realism flags, train + eval | resource-aware | **Res_Gassand** — conservation arm |

`server_version='gassand'` honours the same four per-episode realism flags as the
v2 server — `wind_phase_jitter`, `wind_episode_noise`, `wind_param_jitter`,
`domain_rand` — ported into `handleReset` (dedicated RNG streams: `seed+424243`
for wind mods, `seed+848487` for domain-rand; the spawn RNG is left untouched).
All default **off**, so a run with no flags is byte-for-byte the deterministic
baseline. R_Gassand deliberately keeps the plain reward (the realism bundle is
its single information-structure change; v2's reward shaping is *not* ported).
Because the physics, 21-dim state and 11-action head differ from v1/v2, gassand
scores are **not** comparable to H–T — compare within the gassand family
(deterministic demonstrator vs R_Gassand), and via a transfer probe once both
checkpoints exist.

## The multi-seed transfer probe

Single-episode gassand scores are close to meaningless under realism — one
realism episode's TWR50 swings ±35pp with the wind draw (the seed-42 calm render
scored 44% against a 19.1% ± 16.6 ten-seed mean), and each trainer's in-run
"best" is the max of a noisy eval sequence, i.e. winner's-cursed.
`probe_gassand_transfer.py` is the honest instrument (the gassand analogue of
`probe_realism_transfer.py`): fixed checkpoints, no retraining, evaluated in
both environments × 3 presets × 10 seeds, composite `0.5·mean + 0.5·worst`.
Because the realism port draws from dedicated RNG streams and leaves the spawn
RNG untouched, seed *i* gives the identical spawn in both modes, so the probe
also reports the spawn-luck-cancelling paired per-seed delta. End-of-episode
reserves are averaged per cell alongside TWR.

Results (10 seeds, 72 h, `probe_gassand_transfer.json`, 2026-07-17):

| policy | deterministic | realism | degradation | sand left (realism) |
|--------|--------------|---------|-------------|---------------------|
| heuristic | 4.5% | 2.7% | +1.8pp | 0.0 kg |
| `gassand` (det-trained) | 7.4% | 3.5% | +3.8pp | 0.0 kg |
| `r_gassand` (realism-trained) | 7.3% | 5.5% | +1.8pp | 0.0 kg |
| `res_gassand` (realism + resource reward) | 7.2% | 2.3% | +4.9pp | ~17.9 kg |

Among the plain-reward policies: the two learned ones tie on deterministic wind
(7.4 vs 7.3) — realism training cost nothing on the clean env — while under
realism the det-trained policy loses half its composite and R_Gassand keeps
three quarters. Every plain-reward cell ends with sand ≈ 0.00 kg: the plain
reward never penalises release, which motivates the resource-aware section
below. Res_Gassand (trained with that reward) inverts the economy: it keeps
~89% of its sand and ~99.9% of its helium, ties the others on deterministic
wind (its calm 30.4% is the best single cell of any policy — frugal metering
wins when wind is predictable), but under realism it won't spend what chasing
shifting wind costs and its composite drops to 2.3%. At the default prices the
conservation trade is ~0.15pp of realism composite per kg conserved; mapping
the rest of that curve (cheaper releases → more chasing) is a coefficient
sweep away.

## The resource-aware reward (flag-gated)

Switched on per reset with `use_resource_reward` (parsed in `handleReset`,
applied in `stepAction`; physics untouched; all coefficients per-reset tunable):

- **Per-release cost** — `−sand_cost_per_kg·sand_released −
  helium_cost_per_kg·He_released` each step (defaults 2.0 / 25.0). Helium is
  priced above its 6.236× lift equivalence because venting is doubly
  irreversible: it lowers the ceiling permanently *and* arresting the resulting
  descent costs sand.
- **Terminal reserve bonus** — `terminal_reserve_bonus · mean(He gauge, sand
  gauge)` at episode end (default 25.0).
- **Depletion / floor penalty** — one-time `depletion_penalty` (default 25.0)
  the first time either reserve runs dry, plus `floor_penalty` (default 0.1)
  per step pinned at `ALT_MIN` — an unrecoverable state in this model.

Calibration (from the probe's honest numbers): a plain-reward policy under
realism earns roughly 60–200 base return per 72 h episode while spending all
20 kg sand (cost 40) and ~3 kg He (cost ~80) — full waste costs about one
episode's base return. The frugal pattern R_Gassand showed in its first 30 h
(sand 64%, He 90%) costs ~20. FLOAT-forever nets only the terminal bonus, and
1 kg of sand (cost 2) buys ≈ two in-radius steps, so strategic spending still
dominates never-spending — the degenerate policies lose on both ends.

`ablate_res_gassand_train.py` (**Res_Gassand**) trains with this reward on top
of R_Gassand's realism bundle — the one-change-at-a-time step after R_Gassand.
It checkpoints best-by-eval **return** (the trained objective), logging TWR and
reserves alongside; cross-policy comparison stays with the probe, whose metrics
(TWR + reserves) are reward-independent. In the first full run (23 min local,
seed 42) the return-optimal checkpoint came early (ep 199) and sits at the
frugal end of the trade-off — the probe table above is its honest readout.
