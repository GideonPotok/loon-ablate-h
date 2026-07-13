# Runbook: Ship a New Ablation

**Type:** Delivery procedure (not an incident runbook — adapted from the runbook format because shipping an ablation is itself an operational procedure with a fixed sequence and known failure points)
**Estimated Time:** ~1–2 hours of hands-on work, plus up to 6h of unattended CI time for the training matrix
**Last Tested:** 2026-07-01 (Ablation P, commit `4242060`)
**Owner:** Gideon Potok

See [architecture/ablation-pipeline.md](../architecture/ablation-pipeline.md) for the system this runbook drives.

## When to Use

You have a hypothesis about why the current best ablation is underperforming
(e.g. "K's reward gradient goes flat past 500km," "M's option-critic head is
unstable") and want to test a specific, small code change against the
existing 3-preset (`tropical` / `strong-shear` / `calm`) eval harness.

## Prerequisites

- [ ] Python 3.13 and Node.js 22 available locally (`python --version`, `node --version`)
- [ ] `pip install torch numpy`
- [ ] A clear, one-sentence diagnosis of *why* the ablation you're branching from underperformed — this becomes the docstring
- [ ] `gh` CLI authenticated, or Actions-tab access, to trigger `workflow_dispatch` and download artifacts

## Steps

### 1. Ideate — write the docstring before the code

Every `ablate_<letter>_train.py` opens with the same three-part docstring.
Copy the shape from the most recent ablation (e.g. `ablate_h_train.py:1-18`):

```python
"""
Ablation <X> — <one-line name>:
  <what this is relative to the ablation it's diagnosing/extending>

Question this answers:
  "<the specific thing you don't know that this run will tell you>"

Changes vs Ablation <prior>:
  - <change 1>
  - <change 2>
  - Everything else identical
"""
```

**Right way:** name the *specific prior ablation* you're diagnosing, not just
"the previous letter." Ablations don't always branch from the immediately
preceding letter (see [Branching Model](#branching-model) below) — K2
diagnoses K, N diagnoses K2 (not L or M), O and P diagnose M's *recipe* while
living on N's *code*. State this explicitly in the docstring so the git
branch and the stated diagnosis don't drift apart.

### 2. Implement — extend the shared library, don't fork it

Add your change as a config flag or dataclass field in `qr_agent.py` /
`balloon_env.py` / the JS server, not as a copy-pasted new file.

**Right way (what M did):** Option-Critic + GRU were added as `use_recurrent`
/ `use_options` flags on the existing `QRConfig`, with a third training
method (`train_batch_options`) added alongside the existing two. Every
ablation before and after M still imports the same `qr_agent.py`.

**Wrong way to avoid:** copying `qr_agent.py` into a new file per ablation.
Nobody has done this yet for the Python agent — keep it that way. It *has*
happened on the JS side historically (`js/dqn.js`, `js/qr_agent.js`,
`js/iqn_agent.js` are three separate, redundant agent implementations from an
earlier tool, none of which the training pipeline uses today) — don't repeat
that pattern going forward.

If your change adds a new flag that Python sends to the JS env server (like
K's `shaping_linear`/`shaping_D_max`), **grep for where the JS side parses
incoming request keys and confirm your new key is actually extracted**, not
just referenced with a fallback default. K's flags were sent from Python,
read by the shaping-computation code, but never pulled out of the request in
`handleReset()` — the fallback (`ep.flags.shapingDMax || (2.0 * R)`) silently
substituted the *old* default, and the entire 10-worker run trained against
the wrong hyperparameter with no error. This is the single most expensive
mistake in the ablation history — it cost a full re-run (K2) to catch and fix.

### 3. Wire up the CI trio — and keep it in lockstep

Three files currently hardcode the ablation letter. All three must change
together:

```bash
# run_worker.py
from ablate_<letter>_train import worker_fn      # was: ablate_<prior>_train
WEIGHTS_PREFIX = "dqn_ablate_<letter>"            # was: dqn_ablate_<prior>

# collect.py
WEIGHTS_PREFIX = "dqn_ablate_<letter>"            # was: dqn_ablate_<prior>

# .github/workflows/train.yml
name: Train Ablation <LETTER>                     # was: Train Ablation <PRIOR>
#   upload glob:   weights/dqn_ablate_<letter>_w*.pt / .json
#   final artifact name: final-ablate-<letter>
#   final artifact path: weights/dqn_ablate_<letter>.pt / _summary.json
```

**Right way:** grep the whole repo for the prior letter's `WEIGHTS_PREFIX`
string before pushing, and confirm zero hits outside files you intentionally
left alone. This is exactly the class of bug that produced three separate
fix commits (`82a5ff0`, `6cb35b1`, `174adf6`) across two branches before it
was solved end-to-end — one of those fixes (`6cb35b1`) was itself incomplete
(hardcoded single-digit worker ids) and was never merged back.

**Better way, not yet built:** parameterize `train.yml` with a
`workflow_dispatch` input instead of hand-editing it. See
[ADR-0001](../adr/0001-experiment-tracking-tooling.md) and the architecture
doc's [Future Considerations](../architecture/ablation-pipeline.md#future-considerations).
If you have time before your next ablation, do this first — it removes this
entire step of the runbook.

### 4. Smoke test before spending 60 worker-hours

```bash
python run_worker.py --worker-id 0 --max-episodes 5
python collect.py
```

or trigger `smoke_test.yml` in the Actions tab. This should complete in
minutes and produce a `weights/dqn_ablate_<letter>.pt` with `n_workers: 1`
and a tiny `best_episode`.

**Critical: do not let this file survive as your "final" result.** Ablation
I's committed weights today (`weights/dqn_ablate_i.pt`,
`weights/dqn_ablate_i_summary.json`) are exactly this smoke-test output
(`n_workers: 1`, `best_episode: 4`, `best_score: 6.9%`) — because the smoke
test and the real run write to the identical filename, nothing forces you to
notice the swap. Before triggering the real matrix, either delete the smoke
output locally or rename it out of the way.

### 5. Launch the full matrix

Trigger **train.yml** from the Actions tab (`workflow_dispatch`). Ten workers
run in parallel, each capped at 6 hours. `fail-fast: false` means one
worker's crash doesn't cancel the other nine.

### 6. Collect

The `collect` job runs automatically after `train` (on success *or*
failure — not on cancel), downloads every `worker-*` artifact into
`weights/`, runs `collect.py`, and uploads `final-ablate-<letter>`.

If collecting locally instead:
```bash
gh run download <run-id> -n final-ablate-<letter> -D weights/
python collect.py
```

**Verify before trusting the winner:** open `weights/dqn_ablate_<letter>_summary.json`
and sanity-check `n_workers` (should be 10, or however many you launched) and
`best_episode` (should be near the curriculum's total episode count, not a
single digit).

### 7. Visualize

```bash
python replay.py --weight weights/dqn_ablate_<letter>.pt          # PNG, all 3 presets
python make_gif.py --ablation <letter>                            # GIF, if make_gif.py's
                                                                    # ENV_FLAGS registry has
                                                                    # an entry for <letter>
```

If your ablation changes the state vector, reward shaping, or architecture
flags (recurrent/options), **add an entry to `make_gif.py`'s `ENV_FLAGS` /
`AGENT_KWARGS` dicts** rather than writing a new one-off script. A prior
one-off (`make_gif_j.py`, observed only as an uncommitted file in a worktree)
had a copy-paste bug — its constant was still named `ABLATE_I_FLAGS` despite
the file being for ablation J — because it was cloned from the I-specific
script and never fully renamed. The registry pattern in `make_gif.py` exists
specifically to avoid this.

### 8. Commit the deliverables

Commit `replay_ablate_<letter>_<preset>.png` and `.gif` files at repo root
(matching the existing naming convention), plus the winning `.pt` if it's
small enough / valuable enough to keep tracked (see `weights/final-ablate-h/`
for the precedent — most other ablations do not keep tracked weights).

## Branching Model

**Convention:** `ablate-<letter>-<short-descriptive-slug>` (e.g.
`ablate-i-linear-shaping`, `ablate-l-fourier-features`), one branch per
ablation. This is generally followed, with the deviations below worth
knowing about before you create your next branch.

**Ablations don't fork linearly.** The actual lineage, as of 2026-07-01:

```
main (H) ── ablate-i-linear-shaping (I)
   │
   └── (fix-up commits) ── J ── K ── L ── K2 ──┬── M ── O ── P  (O, P actually
                                (K re-run       │              descend from N's
                                 after fixing    └── N          code, not M's —
                                 a silent JS                    see below)
                                 flag-parsing
                                 bug)
```

- **K2 reuses K's branch-naming slot** (`ablate-k-fixed`) instead of a
  `ablate-k2-...` name that would match everywhere else K2 appears (weights
  filenames, PNG/GIF names, `WEIGHTS_PREFIX='dqn_ablate_k2'`). If you go
  looking for "ablation K2" by branch name, look for `ablate-k-fixed`, not
  `ablate-k2-*`.
- **M and N are siblings**, both branching from K2 — N's own docstring
  reasons about L's Fourier-feature work, but N does not contain L's or M's
  commits as git ancestors (confirmed: `git merge-base` of M and N is K2).
  Any awareness of a sibling branch's ideas came from reading its code, not
  from merging it.
- **O and P build on N's code line, not M's**, even though O's docstring says
  it "stabilizes M's option-critic recipe." Getting M's Option-Critic + GRU
  logic onto N's line required hand re-implementing/porting it — there is
  **no `git merge` commit anywhere in this repo's history**. (Commit
  `5dd2090`, "Add ablation M... support to replay/gif tooling," is exactly
  this kind of manual port, applied to the viz tooling.)
- **Branch/worktree name collision:** `ablate-j-recovery-spawn` (the plain
  branch) never received the actual Ablation J commit (`0123b86`) — that
  commit exists only on the near-identically-named
  `worktree-ablate-j-recovery-spawn`. If you're picking up "ablation J" work,
  verify which of the two branches you're actually on
  (`git log --oneline -3`) before assuming.
- **`README.md` lives off `main`.** It was added on `ablate-i-linear-shaping`
  (commit `b8416bf`) — a branch that is *not* an ancestor of `main`. `main`
  itself sits at `174adf6`, and every ablation from J through P descends from
  `174adf6`, not `b8416bf`. Practically: the repo's only onboarding README
  does not exist on any branch where J–P's code lives. This `docs/` tree is
  added on top of `main` (`174adf6`) for the same reason the README isn't
  reachable from those branches — **if you want these docs visible from an
  in-flight ablation branch, merge or cherry-pick this commit onto it
  explicitly; it will not appear automatically.**
- **Auxiliary branches exist for cross-cutting fixes**, not tied to one
  ablation letter: `replay-fix-ij` (replay tooling for I/J), `viz-port`
  (porting visualization code). Several `worktree-<adjective>-<adjective>-<noun>`
  branches (e.g. `worktree-greedy-plotting-swan`) are auto-generated by the
  `EnterWorktree` tool when no explicit name is given — these get reused
  across unrelated ablations over time as worktree directories are recycled,
  so the directory/branch name tells you nothing about which ablation is
  currently checked out there. Always confirm with `git log --oneline -3`,
  not the directory name.

**Right way going forward:** when starting a new ablation, decide explicitly
which prior branch you're forking from (not necessarily the alphabetically
previous one), name the branch to match every other place that ablation's id
appears (weights prefix, PNG/GIF filenames), and if you want a sibling
ablation's changes, `git merge` or explicitly note that you're hand-porting
them (as O's docstring does) — don't let the git graph and the stated
diagnosis silently diverge.

## Rollback

If a launched matrix run turns out to be testing the wrong hypothesis (e.g.
you discover the silent-flag-drop bug mid-run, as happened between K and
K2): let the run finish or cancel it, do **not** overwrite the broken
result's weights — keep them under their own `WEIGHTS_PREFIX` for
comparison, then branch a fresh `-fixed` or `<letter>2` variant per the K→K2
precedent, with a docstring that explains what was wrong with the first
attempt.

## History

| Date | Ablation | What | Outcome |
|------|----------|------|---------|
| 2026-06-24 | H | Extended A's curriculum 2800→3600 eps | Baseline 49.1%→established; artifact-path bugs found and fixed same day |
| 2026-06-26 | I | Linear potential shaping, v2 server | Committed "final" weights are actually smoke-test output (see Step 4) |
| 2026-06-29 | J | Recovery-spawn training | Landed on `worktree-ablate-j-recovery-spawn`, not the plain `ablate-j-recovery-spawn` branch |
| 2026-06-30 13:04 | K | Exponential shaping, τ=500km | Silent JS flag-parsing bug — τ never actually changed from 100km |
| 2026-06-30 14:01 | L | Fourier time features, state 20→24 dim | — |
| 2026-06-30 14:28 | K2 | Re-run of K after fixing the flag-parsing bug | Confirms τ=500km's actual effect |
| 2026-06-30 16:11 | M | Option-Critic + GRU-64, on top of L | — |
| 2026-06-30 17:47 | N | γ 0.97→0.99, curriculum rebalance, eval epsilon-greedy leak fix | Sibling of M (both fork from K2) |
| 2026-06-30 18:26 | O | "Stabilize" M's option-critic recipe | Built on N's code, M's ideas ported by hand |
| 2026-07-01 09:21 | P | Isolate gradient clipping as the only variable vs M | — |
