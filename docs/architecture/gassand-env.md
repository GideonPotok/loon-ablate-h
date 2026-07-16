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

This is the model requested for the team's real platform. It is deliberately a
**physics-only** deliverable: the reward is still the plain v1 station-keeping
shape, and the resource economy is exposed in `info` so a resource-aware reward
can be layered on later without touching the dynamics.

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

## Baking in a resource-aware reward (next step)

The physics is intentionally decoupled from reward. To make resource use matter,
add terms in `computeReward` / `handleStep` of the gassand server (all inputs are
already in scope):

- **Per-release cost** — penalise `helium_released_kg` / `sand_released_kg` each
  step (helium is the scarcer, one-way-down resource, so weight it higher).
- **Terminal reserve bonus** — reward leftover `helium_kg` / `sand_kg` at episode
  end to encourage frugal station-keeping.
- **Depletion / floor penalty** — a large negative for running a reserve to zero
  or pinning at `ALT_MIN`, which in this model is an unrecoverable state.

Keep the physics (`js/balloon_gassand.js`) untouched when doing so — only the
reward assembly in the server needs to change.
