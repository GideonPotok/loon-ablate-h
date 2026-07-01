# Docs Index

Process and architecture documentation for the ablation-delivery pipeline
(as opposed to the root `README.md`, which documents Ablation H's specific
hyperparameters and results).

- [Architecture: Ablation Delivery Pipeline](architecture/ablation-pipeline.md) — how ideation, implementation, CI, artifact collection, and visualization fit together end to end, plus the key design decisions and their trade-offs.
- [Runbook: Ship a New Ablation](runbooks/ship-a-new-ablation.md) — step-by-step procedure for delivering the next ablation, with the right way flagged at each step against real historical mistakes, plus the full branching model.
- [ADR-0001: Experiment Tracking Tooling](adr/0001-experiment-tracking-tooling.md) — whether Weights & Biases, MLflow, or TensorBoard would help vs. continuing the current docstring/JSON/branch-name process.

**Note on reach:** these docs live on `main` (commit `174adf6` at the time of
writing). Ablation branches J through P do not descend from the commit that
added the original `README.md` and won't automatically see this `docs/` tree
either — merge or cherry-pick it onto an in-flight branch explicitly if you
want it visible there. See the runbook's [Branching Model](runbooks/ship-a-new-ablation.md#branching-model) section for why.

## Where to Commit This `docs/` Tree

**Recommendation: `main`, then cherry-pick onto the current frontier branch.**

1. **Merge/push to `main` first.** `main` (commit `174adf6`) is this repo's
   GitHub default branch (`gh repo view` confirms `defaultBranchRef.name == "main"`)
   and is the one branch nobody is actively training on — every ablation
   happens on its own `ablate-<letter>-<slug>` branch, so `main` doesn't
   churn the way those do. Cross-cutting process docs belong on the branch
   that's stable and universally discoverable, not on any one ablation's
   branch (that's the exact mistake that made the original `README.md`
   invisible from J–P — see [Failure Modes](architecture/ablation-pipeline.md#failure-modes)).
2. **Then cherry-pick (or merge) that commit onto the frontier branch**,
   which as of 2026-07-01 is `ablate-p-clipped-option-critic` — the deepest
   point in the lineage (`main` → H fix-ups → J → K → L → K2 → N → O → P,
   confirmed via `git merge-base --is-ancestor`; P is 9 commits ahead of
   `main`). Whoever picks up the *next* ablation will almost certainly branch
   from P, so putting the docs there too means the next ablation's branch
   inherits them for free.
3. **Do not treat any single `ablate-*` branch as the only place these docs
   live.** Given the fork-not-merge pattern documented in the runbook (M/N
   are siblings, O/P reuse M's ideas without merging), a doc committed only
   to one ablation branch will not reach its siblings.

**Caveat — check before pushing:** this repo has several worktrees actively
in use right now (`git worktree list` shows five, on top of `main`,
`ablate-j-recovery-spawn`, `ablate-m-option-critic`, `worktree-greedy-plotting-swan`,
and `ablate-p-clipped-option-critic`), and at least one commit landed on
`ablate-j-recovery-spawn` in real time while this `docs/` tree was being
written. Re-run `git log --oneline -1 <branch>` immediately before merging
to confirm you're not clobbering concurrent work, and prefer `git merge`
or `git cherry-pick` over a force-push.

## A Note on "main2"

No branch, tag, remote ref, reflog entry, or dangling commit named `main2`
exists anywhere in this repo's history — checked via `git for-each-ref`,
`git reflog --all`, `git fsck --dangling`, and the GitHub API
(`gh api repos/GideonPotok/loon-ablate-h/branches`, which lists only `main`,
the nine `ablate-*` branches, and `worktree-greedy-plotting-swan`). The
GitHub default branch has always been `main` (`defaultBranchRef.name ==
"main"`). The sibling repos in this same research lineage
(`loon-ablate-f`, `loon-ablate-g`) and the upstream fork this project builds
on (`looney_loons` / `google_loon`, and further upstream
`google/balloon-learning-environment`) also have no `main2` — `loon-ablate-f`
and `-g` each have only `main`, and `google/balloon-learning-environment`
uses `master`, not `main`. If you're looking for a second trunk, the closest
real thing is the repo-per-early-ablation pattern itself: `loon-ablate-f` and
`loon-ablate-g` are each a *separate GitHub repo* with their own `main`,
predating the switch (starting at Ablation H) to keeping all later ablations
as branches within one repo. That history — multiple same-named `main`
branches across sibling repos — is the most likely source of a "wasn't there
another main" recollection; there is no second branch within *this* repo.
