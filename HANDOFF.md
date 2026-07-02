# Handoff — 2026-07-02 (Ablation Q training / realism era begins)

State of play when this session paused, and exactly what to do on return.
Written from the `ablate-q-per-step-option-critic` worktree session that built
Ablations Q and R. Companion background: the Ablation Lineage table in
`README.md` (H→P) and `docs/architecture/ablation-pipeline.md`.

## Where things stand

1. **Ablation Q is training on CI** — option-critic with the cadence confound
   removed (M's exact architecture + L's optimization recipe: per-step
   training every 2 env steps ⇒ 8× M's gradient count, lr 1e-4,
   target-update 15, no clipping).
   - Run: <https://github.com/GideonPotok/loon-ablate-h/actions/runs/28603087701>
   - Branch `ablate-q-per-step-option-critic`, commit `adf40de`. As of this
     writing: 1/10 workers done, 9 running (~final stretch of a ~4.6h run).
2. **The realism era is implemented and shipped** — branch
   `ablate-r-realism-env` (commit `a31cfee`), draft PR
   [#3](https://github.com/GideonPotok/loon-ablate-h/pull/3) (based on the Q
   branch, not main — the lineage never merged back to main, so a main-based
   PR shows phantom conflicts; see the PR comment).
   - Flag-gated v2-server env changes, default off, flags-off path **proven
     bit-identical** (fixed-seed 3-preset × 100-step trajectory diff):
     `wind_phase_jitter`, `wind_episode_noise`, `wind_param_jitter`,
     `domain_rand`, plus `use_estimated_phase_features` (real-life-computable
     IGW phase estimator: demodulates GPS-drift wind residuals; <10° phase
     error by 6h, 1–4° by 24h; mutually exclusive with `use_time_features`).
   - `probe_realism_transfer.py` — evaluates any trained checkpoint in
     legacy vs realism env. **Headline result (10 seeds × 72h): N scores
     46.1% legacy → 16.0% realism (−30.1 pp; calm 98%→13%; stdev explodes
     to ±16–25 pp).** Much of N's skill was reading the sim's wind clock,
     which does not exist in reality.
   - `ablate_r_train.py` — the floor arm: N's exact recipe + all 4 realism
     flags (train AND eval), no time features (state 20-dim). CI trio is
     wired to `dqn_ablate_r`. Smoke-tested end-to-end.
3. **Approved experimental design (user decisions, do not relitigate):**
   bundle all realism flags into ONE fixed testbed; vary the *agent's
   information structure* one arm at a time; memory arm = **plain GRU**
   (`use_recurrent=True, use_options=False`). R/S/T scores are NOT
   comparable to H–Q — the transfer probe is the only cross-env bridge.

## What to do on return (in order)

1. **Collect Q's results** (run should be finished):
   `gh run download 28603087701 -n final-ablate-q -D weights/` then read
   `weights/dqn_ablate_q_summary.json`. Compare against L = 46.9% and
   M = 24.7% (10-seed clean re-eval numbers) and M's 9.9 pp seed stdev.
   Interpretation, per the Q plan: **Q ≈ L** ⇒ M's failure was the starved
   per-episode cadence, option-critic itself is fine; **Q ≈ M despite 8×
   the gradient steps** ⇒ real evidence against option-critic here.
   Add the Q row to README's Ablation Lineage (on main). Optionally
   regenerate replay GIFs (`replay.py` / `make_gif.py` have `'q'` entries
   on the Q branch).
2. **Run the transfer probe on Q's weights** (from the `ablate-r-realism-env`
   branch): `python probe_realism_transfer.py --ablation q --weights
   weights/dqn_ablate_q.pt`. This measures whether the GRU's hidden state
   also just encoded the clock, or something more robust — worth a row in
   the writeup either way.
3. **Dispatch Ablation R training** (only after Q's run is fully done —
   two 10-worker matrices contend for runner concurrency):
   `gh workflow run train.yml --ref ablate-r-realism-env`, then confirm
   with `gh run list --limit 3` that it started on commit `a31cfee` or
   later. Expect a cost similar to N's run (same recipe; the realism flags
   add negligible compute).
4. **Sanity-check R before building S/T**: R's best score should sit well
   above the random-policy floor (~7%) — the probe's 16% transfer score is
   a rough lower bound for what training-in-env should beat. If R lands at
   the floor, the env may be too hard; investigate before spending more CI.
5. **Create S and T as small deltas off `ablate-r-realism-env`:**
   - **S** (= estimator arm): copy `ablate_r_train.py` → `ablate_s_train.py`;
     set `use_estimated_phase_features: True` in `_env_flags()`,
     `state_dim = 24`; new prefix `dqn_ablate_s`; CI trio + replay/gif
     registry entries (copy the `'r'` pattern, estimator flag on).
   - **T** (= memory arm): R's env + config, plus `use_recurrent=True,
     use_options=False, gru_hidden=64` and Q's sequence-replay machinery
     (EpisodeSequenceBuffer, burn-in 16 + train 16, TRAIN_EVERY_STEPS=2 —
     lift from `ablate_q_train.py`). Check Q's completed run first for any
     recurrent-training lessons (wall-clock fit, stability).
   - House rules per ablation: docstring-first question, own branch, CI
     trio updated together + stale-prefix grep, smoke test, **delete smoke
     artifacts from weights/ before committing** (Ablation I pitfall),
     replay/gif entries.
6. **Read out R vs S vs T** (same env; matched eval protocol):
   S ≫ R ⇒ engineered phase estimation recovers the oracle's contribution.
   T ≈ S ⇒ memory learns phase inference (the GRU's rationale finally
   vindicated in the setting it was designed for). T ≈ R ⇒ memory fails to
   learn it and explicit estimation wins. All ≈ R ⇒ phase information
   wasn't the load-bearing ingredient.

## Gotchas / context that is easy to lose

- **Old ablation reproducibility is guaranteed**: all realism behavior is
  opt-in via reset flags; with flags off the server is bit-identical to
  pre-realism behavior. Don't "clean up" the `?? 0` / `?? 1` defaults in
  `js/wind.js` — the null-mods path relies on exact IEEE identities.
- **Estimator attribution matters**: observations are attributed to the
  step-START altitude/time where physics sampled the wind. Midpoint
  attribution re-introduces multi-m/s base-layer boundary pulses that swamp
  the demodulator on strong-shear (this was found and fixed the hard way).
- **Diurnal phase is intentionally NOT jittered** — the 24h solar tide is
  genuinely clock-locked in reality; only IGW/PW phases are randomized.
- **`use_time_features` and `use_estimated_phase_features` are mutually
  exclusive** — the server errors on reset if both are set.
- The Q-branch worktree used for this work lives at
  `.claude/worktrees/ablate-q-per-step` (branch now `ablate-r-realism-env`).
  The full R/S/T plan file: `~/.claude/plans/perfect-do-it-make-lucky-stardust.md`.
- GitHub shows workflow *names* from the default branch (runs of Q/R's
  `train.yml` display as "Train Ablation I") — check the run's branch/sha,
  not the displayed name.
- PR [#2](https://github.com/GideonPotok/loon-ablate-h/pull/2) (docs:
  noise/exploration audit + the then-unwired domain-rand gap) predates this
  work — the realism branch now wires domain-rand into the v2 server, so
  review that doc PR against the new reality before merging.
