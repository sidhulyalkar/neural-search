# Repository Cleanup Proposal — 2026-07-01

Read-only inventory. Nothing below has been deleted yet except the one item explicitly marked done.

## Done already (unambiguous, no approval needed)

- `data/graph/neural_search_graph.real_corpus.json.bak-pre-reanalysis` (53MB, untracked, my own transient safety backup from this session) — **removed**, the fix it was guarding against is verified (zero dangling edges, 114 tests passing).

## Safe deletes (tracked files, low risk, recommend proceeding)

| Path | Why |
|---|---|
| `manifest.json` (repo root) | Old MVP artifact list from the `demo_v05`/`real_v07` regime; superseded by `reports/eval/current_artifact_manifest.json`. |
| `benchmark_queries.yaml` (repo root) | Duplicate of `data/eval/benchmark_queries.yaml`, which is canonical (referenced by the ablation ladder). |
| `reports/eval/ablation_ladder_report.partial.json` / `.partial.md` | Incomplete/interrupted run artifacts, not a finished report. |

## Archive candidates (tracked, superseded by newer versions — recommend moving to `reports/archive/` rather than deleting, so history isn't lost)

- `reports/ablation_v08.md`, `reports/usefulness_benchmark_v08.md`, `reports/benchmark_baseline_v0_2.md` / `.json` — superseded by v09–v11 equivalents.
- `reports/strategy/2026-06-23_qrels_branch_merge_and_kg_rigor_plan.md`
- `reports/strategy/brainknow_comparison_publishable_plan.md`
- `reports/strategy/claude_whitepaper_frontend_execution_prompt.md`
- `reports/strategy/neural_search_project_roadmap.md`
- `reports/strategy/next_gen_kg_development_plan.md`

  All five predate and are superseded by the two 2026-07-01 strategy docs plus the new consolidated plan (`reports/strategy/2026-07-01_next_phase_growth_validation_plan.md`).

## Needs your explicit call (destructive, large, or ambiguous purpose)

1. **`brain/` directory + `brain.zip`** (~62MB combined) — contains a single `Brain.glb` 3D model file under `brain/source/`. This does not appear connected to any neuroscience-search backend code (no imports/references found anywhere in `neural_search/`, `apps/`, or `scripts/`). Looks like an unrelated 3D asset (possibly for a future visualization feature, or leftover from an unrelated experiment). **I won't delete this without you confirming it's not needed.**

2. **`.claude/worktrees/`** — 12 worktree directories, ~2.7GB total, including the stray `agent-a332321c26e449277` worktree with a duplicate `corpus_kg_linker.py` noted earlier this session. These are typically cleaned up automatically once an agent's work is merged/abandoned, but I don't have visibility into which are still in active use. **Recommend running `git worktree list` yourself and pruning finished ones** — I'd rather not guess which of 12 are safe to remove.

3. **`neural_search/release/` module + `tests/test_real_corpus_v07.py` + `corpus/convert_demo_seed.py`** (old release-checker regime) — still imported by one test file, not dead, but represents the `demo_v05`/`real_v07` artifact regime that `reports/eval/current_artifact_manifest.json` was built to replace (per the existing `reports/strategy/2026-07-01_platform_evaluation_and_adoption_plan.md`, Phase 0 already recommends reconciling this). **Recommend deferring removal until the release checker is updated to understand the current artifact regime** (that's a Phase 0 task in the next-phase plan, not a delete-it-now cleanup item) — removing it now would leave the repo with no release-readiness check at all.

4. **Paired audit instruction/summary files** (`affordance_audit_instructions.md` + `affordance_audit_summary_2026.md`, and three similar pairs for findings/paper-links/typed-fields) — the `_instructions.md` files are templates for work that's already been completed (per the paired `_summary_2026.md` files). Low-risk archive candidates, but keeping them costs little; flagging for your awareness rather than recommending action.

## What I'd like from you

A yes/no on: (a) proceed with the "safe deletes" list, (b) archive (not delete) the superseded reports/strategy docs listed above, (c) whether `brain/`/`brain.zip` can go, (d) whether to leave worktree cleanup to you directly.
