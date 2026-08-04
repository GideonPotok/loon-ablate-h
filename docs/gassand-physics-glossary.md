# Gassand physics glossary

Every mathematical symbol from the AP-Physics-B/C treatment of the helium/sand
zero-pressure balloon, with the value it takes in this sim (sources:
`js/config.js` `DEFAULT_GASSAND`, `js/balloon_gassand.js`, and the measured
49 m/kg settling probe).

| Symbol | Name / meaning | Units | Value in this sim |
|---|---|---|---|
| W | weight of the whole system, m·g | N | ≈ 139 × 9.81 ≈ 1366 at launch |
| F_B | buoyant force (Archimedes) | N | equals W at the float; code computes net `(displacedAirMass − totalMass)·G` |
| F_D | drag force, −½·C_d·ρ·A·\|v\|·v | N | sign opposes motion (`balloon_gassand.js:153`) |
| F_net | net vertical force fed into m·dv/dt | N | buoyancy + drag |
| ΔF | buoyancy imbalance right after a release | N | ≈ g·Δm_sand for sand; ≈ 6.236·g·Δm_He for helium (release ladders paired by this factor) |
| m, m_total | total mass: structure + sand + helium | kg | 100 + 20 + 19.24 ≈ 139.24 at launch (`DRY_MASS_KG` + reserves) |
| m_He | helium mass aboard (only depletes) | kg | 19.24165 at launch (`HELIUM_INIT_KG`) |
| Δm | mass shed by one release | kg | sand actions 0.005–1.0; helium actions 0.0008–0.1604 |
| g (code `G`) | gravitational acceleration | m/s² | 9.81 |
| v (code `vv_m_s`) | vertical velocity, + up | m/s | clamped to ±2.5 (`MAX_VV_M_S`) |
| v_terminal | drag-limited vertical speed, √(2·ΔF/(ρ·C_d·A)) | m/s | 0.11–1.56 across the action ladder; ∝ √Δm, so a 200× mass range gives only ~14× speed |
| M_air | molar mass of air | kg/mol | 0.0289647 (`M_AIR_KG_MOL`) |
| M_He | molar mass of helium | kg/mol | 0.0040026 (`M_HE_KG_MOL`) |
| M_air/M_He | air displaced per kg of helium (bubble regime) | — | 7.237 (`AIR_PER_HELIUM`); net lift per kg He = 6.236 |
| R | universal gas constant, in V = m_He·R·T/(M_He·P) | J/(mol·K) | 8.314 |
| T | ambient temperature (ideal-gas law contexts) | K | from `atmosphereAt(alt)` |
| P | ambient pressure | Pa | from `atmosphereAt(alt)` |
| ρ, ρ_air(h) | ambient air density at altitude h | kg/m³ | ≈ 0.139 at the 17.1 km float (= m_total/V_env) |
| ρ₀ | sea-level reference density | kg/m³ | ≈ 1.225 |
| H | density scale height in ρ(h) = ρ₀·e^(−h/H) | m | ≈ 6800 near these altitudes (fit to the sim atmosphere, not a config constant) |
| V, V_displaced | volume of air displaced by the gas bubble | m³ | min(bubble, envelope) — the two-regime `min` in `displacedAirMass` |
| V_env | envelope volume (fixed) | m³ | 1000 (`V_ENVELOPE_M3`) |
| r | balloon radius (sphere of volume V_env) | m | ∛(3·V_env/4π) ≈ 6.20 (`balloonRadius_m`) |
| A | cross-sectional area for drag, π·r² | m² | ≈ 121 |
| C_d | drag coefficient | — | 0.47, sphere (`DRAG_COEFFICIENT`) |
| h | altitude | m | sim band 15,000–22,000 (`ALT_MIN_M`/`ALT_MAX_M`) |
| h_park | park (float) altitude solving ρ(h)·V_env = m_total | m | 17,093 at launch config (`altBandLow_m`); 18,076 with all sand dropped |
| Δh | shift in park altitude per mass shed, ≈ H·(Δm/m) | m | ≈ 49 per kg of sand (measured); 983 for the whole 20 kg bag |
| k | effective spring constant of the float, m·g/H | N/m | 139·9.81/6800 ≈ 0.20 — matches the code comment "restoring force ~0.2 N/m" |
| ω | natural angular frequency of the float, √(g/H) | rad/s | ≈ 0.038 |
| T_osc | vertical oscillation period, 2π/ω (distinct from temperature T) | s | ≈ 166 (~2.8 min) — why the integrator sub-steps at 5 s (`VERT_SUBSTEP_S`, stability needs dt·ω < 2) |

Two symbols to keep un-confused: T is temperature in the ideal-gas law but a
period in the oscillation analysis (written T_osc here), and ρ always means
*ambient air* density — the helium's own density never appears because the
bubble-regime buoyancy m_He·(M_air/M_He)·g already folds it in.
