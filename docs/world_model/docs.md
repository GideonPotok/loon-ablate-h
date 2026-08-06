Great — let me cover both in depth.

---

## Part 1: Hydrogen vs Helium — What Has to Change in the BLE Balloon Model

The BLE's balloon physics are hardcoded for Loon's helium superpressure balloon. Adapting them for Valarus's hydrogen design touches six specific places in the code, and some of the changes are more subtle than just swapping a constant.

**The obvious change: molar mass**

In `constants.py`, line 28: `HE_MOLAR_MASS: float = 0.004002602` (helium, 4.003 g/mol). Hydrogen gas (H₂) has a molar mass of 0.002016 kg/mol — almost exactly half. This single constant propagates through the entire simulation via `balloon.py` line 417-419, where total flight system mass is computed:

```python
total_flight_system_mass = (
    constants.HE_MOLAR_MASS * state.mols_lift_gas +
    constants.DRY_AIR_MOLAR_MASS * state.mols_air + state.envelope_mass +
    state.payload_mass)
```

Halving the lift gas mass means the gas-mass term drops from ~27.3 kg (6830 mols × 0.004003) to ~13.8 kg. That's a ~13.5 kg reduction in total system mass (~161 kg → ~148 kg for BLE's default parameters), which increases buoyancy and changes the equilibrium float altitude. The buoyancy equation at line 422-427 computes `rho_air * envelope_volume - total_flight_system_mass` — a lighter gas means this quantity is more positive, so the balloon wants to float higher. In practice, hydrogen provides about 8% more gross lift per unit volume than helium at the same conditions.

**The less obvious change: initial mols of lift gas**

`BalloonState` defaults to `mols_lift_gas: float = 6830.0` — this was calibrated for Loon's helium balloon to achieve neutral buoyancy near 6000 Pa (~20 km altitude). If you just swap in H₂'s molar mass without adjusting this number, the balloon will float significantly higher because it's now overfilled — the same number of moles of a lighter gas means less weight pulling it down. Valarus's balloon will have a different target altitude, a different envelope volume, and different payload mass, so `mols_lift_gas` needs to be recalculated from the design parameters. The equation is straightforward from the superpressure equilibrium in `calculate_superpressure_and_volume()` (lines 552-609): at zero superpressure (neutral buoyancy), the unconstrained volume equals the envelope base volume, so:

```
mols_H2 = (P_target × V_base) / (R × T_internal)
```

For Valarus's specific envelope, you'd plug in their target pressure altitude, envelope volume, and expected internal temperature. The number will be *larger* than 6830 (you need more moles of the lighter gas to fill the same volume to the same pressure), but the total gas mass will still be lower.

**Thermal model: the gas matters more than the code currently admits**

The thermal model in `thermal.py` computes `d_balloon_temperature_dt` — the rate of change of internal gas temperature. It accounts for solar heating, earth IR, emission, and convection, but it treats the internal gas as thermally negligible: the thermal capacity term in line 229-230 is `_PE01_FILM_SPECIFIC_HEAT * balloon_mass`, which only counts the *envelope film mass*. This is a reasonable simplification for helium (monatomic, low heat capacity, small contribution relative to the massive film), but hydrogen is diatomic with a higher specific heat capacity per mole (Cp ≈ 28.8 J/(mol·K) for H₂ vs 20.8 for He). At 6830 moles, the gas heat capacity is ~197 kJ/K for H₂ vs ~142 kJ/K for He, while the film's is ~103 kJ/K (68.5 kg × 1500 J/(kg·K)). The gas contribution is actually *larger* than the film's and the BLE ignores it entirely. For a hydrogen balloon, you'd want to add:

```python
gas_heat_capacity = mols_lift_gas * CP_H2  # + mols_air * CP_AIR
total_heat_capacity = _PE01_FILM_SPECIFIC_HEAT * balloon_mass + gas_heat_capacity
```

This slows down thermal transients — the balloon temperature responds more sluggishly to sunrise/sunset cycles, which affects altitude stability. This matters for trajectory prediction because thermal lag drives slow altitude oscillations that are a real source of prediction error.

**Superpressure and envelope dynamics**

The `calculate_superpressure_and_volume` function uses the ideal gas law (PV = nRT), which is gas-agnostic — the same function works for hydrogen and helium. But the *envelope parameters* are Loon-specific: `envelope_volume_base = 1804 m³`, `envelope_volume_dv_pressure = 0.0199 m³/Pa`, `envelope_max_superpressure = 2380 Pa`. Valarus's hydrogen balloon will have a completely different envelope — Kristian's design with the "silo" launch system suggests a different form factor, and the payload cone with radar-absorbing foam changes the mass distribution. These are empirical parameters that Kristian would need to provide from his prototype testing.

One real physics difference: hydrogen's lower density means that for the same superpressure, the gas flows faster through the ACS valve. The valve flow equation at line 496-499 computes mass flow as proportional to `sqrt(superpressure * gas_density)`. With H₂'s lower molecular mass, the gas density inside the balloon is lower, so the mass flow rate for an "UP" command (venting gas through the valve) drops by a factor of √(M_H2/M_He) ≈ 0.71. This means altitude control response is asymmetric — ascending (venting) is slower with hydrogen, while descending (pumping air in) is gas-independent because you're pumping ambient air, not lift gas. This affects controllability and needs to be accounted for in the RL training.

**What Valarus's "hydrogen safety innovation" might mean for the physics**

The deck mentioned a launch platform enabling safe hydrogen use. If this involves a sealed launch tube (the "silo" concept), the ascent profile is fundamentally different from Loon's: instead of a gradual inflation during open-air ascent, the balloon might be released fully inflated at altitude, or undergo a rapid pressurization transition. The BLE's `stable_init.py` handles initial stabilization, but it assumes a standard open-atmosphere launch. A silo launch would skip the low-altitude regime entirely — which is actually simpler to model (no ground-level turbulence, no building interactions), but the transient dynamics right at release could be complex.

**Bottom line for the adaptation**

The minimal viable hydrogen port is: change one constant (`HE_MOLAR_MASS` → `H2_MOLAR_MASS = 0.002016`), recalculate `mols_lift_gas` for the target float altitude, and get Kristian's envelope parameters (`envelope_volume_base`, `dv_pressure`, `max_superpressure`, envelope mass, payload mass). That gets you a balloon that floats and can be controlled. The thermal model fix (adding gas heat capacity) is important for prediction accuracy but doesn't change the qualitative behavior. The ACS asymmetry is something the RL agent will learn to deal with but that you should be aware of when interpreting training curves.

---

## Part 2: Wind Correction Architectures for Limited Data

The core problem: Kristian has a handful of test flights (maybe single digits), and you need a trajectory prediction model that's better than raw GFS/ECMWF forecasts. The research literature and the BLE codebase both point to a progression of approaches ordered by how much data they need.

**Tier 0: Calibrated physics ensemble (zero flight data needed to start)**

Before any learning, you can already beat a single-forecast trajectory by running a Monte Carlo ensemble. The BLE does this conceptually — its `SimplexWindNoise` adds structured noise to the forecast to create different "possible realities." For real-world prediction, you'd use the forecast model's own ensemble members (GFS produces 31 ensemble members; ECMWF produces 51). Each member gives a different wind field; propagate the balloon through all of them; the spread of resulting trajectories is your uncertainty cone.

The "learning" here is just calibration: historically, are the ensemble spreads well-calibrated? (Usually no — they're underdispersive in the stratosphere.) You can learn a simple scaling factor from even 2-3 flights: if the actual balloon position falls outside the 90% ensemble cone 40% of the time, your cone is too narrow by a learnable factor. This is a one-parameter model and can be fit with minimal data.

**Tier 1: GP correction with flight observations (what the BLE already does, adapted)**

The `WindGP` in `wind_gp.py` is exactly the right starting architecture for early Valarus flights. It maintains a Gaussian process over the 4D space (x, y, pressure, time) with a Matérn kernel, modeling *deviations from the forecast*. As the balloon flies and measures actual wind, the GP integrates those observations and corrects the forecast at nearby points. The key parameters are already tuned: distance scale 357 km, pressure scale 326 Pa, time scale 34560 s (~9.6 hours), signal variance 3.6² m/s, noise variance 0.05 m/s.

For Valarus, you'd use this *during* a flight for online correction — as the balloon sends back telemetry, the GP updates its wind estimate and re-predicts the trajectory forward. The length scales might need adjustment for the specific geography and altitude range, but the BLE's values are reasonable starting points calibrated against real Loon data.

The limitation is that the GP only learns from *the current flight's* observations. It doesn't carry knowledge across flights. Each new launch starts with a blank GP. That's fine for real-time correction but doesn't help you make a better prediction before launch.

**Tier 2: Analog ensemble (cross-flight learning, minimal ML)**

The [JTECH 2016 paper on stratospheric balloon trajectory prediction](https://journals.ametsoc.org/view/journals/atot/33/8/jtech-d-15-0110_1.xml) and the [ResearchGate work on distilling analog ensembles into neural networks](https://www.researchgate.net/publication/343176007_Improving_Wind_Forecasts_in_the_Lower_Stratosphere_by_Distilling_an_Analog_Ensemble_Into_a_Deep_Neural_Network) describe an approach that works with limited flight data. The idea: for each new forecast, find the most similar historical forecasts (analogs), look at what the actual winds were for those historical cases, and use that to correct the current forecast. "Similar" is measured in the forecast feature space — same pressure pattern, same jet stream position, same season, etc.

This is data-efficient because each historical flight contributes multiple observations (one every 3 minutes for days), so even 5 flights give you thousands of data points. The analog matching is non-parametric, so it works well in the small-data regime without overfitting. [Windborne Systems](https://windbornesystems.com/products/forecasts) appears to use this general approach (combining balloon observations with ML correction) as the foundation of their WeatherMesh product, though they now have vastly more data.

**Tier 3: Small neural bias correction (the sweet spot for Valarus)**

Once you have 10-20 flights, you have enough data for a small neural network that learns systematic forecast biases. The architecture I'd recommend for Valarus's data regime:

*Input features:* forecast wind (u, v) at the query point, pressure altitude, time of day, day of year, geographic coordinates, vertical wind shear (du/dp, dv/dp computed from the forecast), and a few large-scale meteorological indices (QBO phase, jet stream latitude — available from public reanalysis).

*Output:* corrected wind (u, v) plus a learned uncertainty (aleatoric + epistemic).

*Architecture:* A residual MLP — 3-4 layers of 64-128 units, with a skip connection from the forecast input to the output. This means the network learns `correction = f(features)` and the output is `forecast + correction`. The skip connection is critical for small data: the network starts at zero correction (identity mapping) and only learns to deviate where the data supports it. This is much easier to train than a network that must learn the entire wind field from scratch.

For uncertainty, use a heteroscedastic Gaussian output (predict both mean and variance) plus MC dropout for epistemic uncertainty. This gives you calibrated error bars that grow when you're extrapolating beyond your training data — essential for the probability cone product.

Training would use a leave-one-flight-out protocol: train on N-1 flights, validate on the held-out flight, report the trajectory prediction error. With 10 flights this is expensive but necessary to avoid overfitting.

**Tier 4: Transfer learning from the BLE's VAE (leveraging your existing work)**

This is the most interesting option and the one that connects your ablation research directly to Milestone 1. The BLE's `Decoder` in `vae.py` has already learned a *generative model of plausible stratospheric wind fields*. It was trained on ERA5 reanalysis data (real historical wind fields, not simulated), compressed into a 64-dimensional latent space. The decoder produces physically-constrained outputs — incompressible flow fields via the stream function differentiation trick (lines 171-186 of `vae.py`).

The transfer learning approach: take the pretrained VAE decoder, freeze it, and learn a mapping from *forecast features → latent space correction*. Instead of training a neural network to directly predict wind corrections in the high-dimensional (u, v) × (lat, lng, pressure, time) space, you train a small network to predict a 64-dimensional vector that, when added to the latent encoding of the forecast, produces a corrected wind field. This is dramatically more data-efficient because the correction lives in a 64-dimensional space rather than the full grid space (21 × 21 × 10 × 9 × 2 = 79,380 values). The decoder's physical constraints (incompressibility) are automatically preserved.

The training loop would look like:
1. Encode the GFS forecast into the VAE's latent space (requires the encoder, or you can optimize the latent directly)
2. Propagate the balloon through the decoded wind field
3. Compare predicted trajectory to actual GPS trajectory
4. Backpropagate through the decoder (frozen) and into the latent correction network
5. The loss is trajectory error, not wind error — this means the model preferentially corrects winds that *matter for balloon position*, ignoring corrections at altitudes the balloon never visits

This is a form of "world model fine-tuning" and it's the approach I'd bet on for Valarus because it combines the BLE's learned prior (what stratospheric wind fields look like) with Valarus's specific flight data (where the forecasts are wrong for their operating region).

**Practical recommendation**

Start at Tier 0 (calibrated ensemble) immediately — it requires no flight data and gives Kristian a trajectory prediction tool to plan launches with. Layer on Tier 1 (GP correction) for real-time in-flight updates — the code is already written in the BLE. As flights accumulate, build the Tier 3 residual MLP — it's simple, interpretable, and the leave-one-out protocol gives honest error bars. Keep Tier 4 (VAE transfer learning) as the research play — it's the most novel, the most publishable, and the most defensible as IP, but it requires more engineering to set up and validate.

The probe protocol you built for the realism-era ablations (`probe_realism_transfer.py`) is the template for evaluating all of these. Instead of measuring "legacy vs realism" degradation, you'd measure "forecast-only vs forecast+correction" improvement, using the same composite score (0.5 × mean + 0.5 × worst-case across conditions). That evaluation framework is itself a deliverable — it's how Kristian demonstrates to customers that the prediction system works.

Sources:
- [Improving Wind Forecasts in the Lower Stratosphere by Distilling an Analog Ensemble Into a Deep Neural Network](https://www.researchgate.net/publication/343176007_Improving_Wind_Forecasts_in_the_Lower_Stratosphere_by_Distilling_an_Analog_Ensemble_Into_a_Deep_Neural_Network)
- [On the Prediction of Stratospheric Balloon Trajectories: Improving Winds with Mesoscale Simulations](https://journals.ametsoc.org/view/journals/atot/33/8/jtech-d-15-0110_1.xml)
- [Autonomous navigation of stratospheric balloons using reinforcement learning (Nature)](https://www.nature.com/articles/s41586-020-2939-8?proof=t)
- [WindBorne WeatherMesh AI Weather Forecast Model](https://windbornesystems.com/products/forecasts)
- [Advanced mathematical modeling and simulation of high altitude balloons (Nature Scientific Reports)](https://www.nature.com/articles/s41598-025-23571-1)
- [High-Altitude Gas Balloon Trajectory Prediction: A Monte Carlo Model](https://www.researchgate.net/publication/262990714_High-Altitude_Gas_Balloon_Trajectory_Prediction_A_Monte_Carlo_Model)
- [NCAR Technical Note: Aerostatic Lift of Helium and Hydrogen in the Atmosphere](https://opensky.ucar.edu/system/files/2024-08/technotes_56.pdf)
- [Lifting gas (Wikipedia)](https://en.wikipedia.org/wiki/Lifting_gas)