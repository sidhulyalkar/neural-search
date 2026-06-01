# Neural Search Handoff Kit

This kit is designed to be dropped next to the repo and handed to Claude Code and Codex as the next-level development packet.

## Contents

- `00_REPO_EVALUATION.md` - current state audit, test results, product/readiness assessment.
- `01_NEXT_STEPS_ROADMAP.md` - prioritized roadmap from cleanup to live corpus expansion to latent-state search.
- `prompts/CLAUDE_NEXT_LEVEL_MASTER_PROMPT.md` - full prompt for Claude Code as product architect/full-stack agent.
- `prompts/CODEX_BACKEND_CORE_PROMPT.md` - full prompt for Codex as backend/retrieval/data systems agent.
- `prompts/CODEX_TESTING_CI_PROMPT.md` - focused prompt to harden tests, CI, type checks, and quality gates.
- `plans/TESTING_AND_CI_PLAN.md` - exact test matrix and quality gates.
- `plans/DATASET_COMPILATION_PLAN.md` - DANDI/OpenNeuro/OpenAlex corpus expansion plan.
- `plans/RETRIEVAL_EMBEDDINGS_PLAN.md` - hybrid retrieval, embeddings, and evaluation improvements.
- `plans/FRONTEND_PRODUCT_PLAN.md` - UI/product polish plan for demos and usability.
- `templates/GITHUB_ISSUES.md` - ready-to-paste issues.
- `templates/PR_CHECKLIST.md` - release/PR guardrails.
- `configs/benchmark_queries_expansion.yaml` - starter benchmark query expansion.
- `scripts/bootstrap_validate.sh` - local environment bootstrap + validation.
- `scripts/quality_gate.sh` - backend/frontend gate for every PR.
- `patches/known_frontend_build_fix.diff` - minimal fix for the current TypeScript build blocker.

## Recommended usage

1. Give Claude `prompts/CLAUDE_NEXT_LEVEL_MASTER_PROMPT.md` plus `00_REPO_EVALUATION.md` and `01_NEXT_STEPS_ROADMAP.md`.
2. Give Codex `prompts/CODEX_BACKEND_CORE_PROMPT.md` plus the plans in `plans/`.
3. After both complete work, run `scripts/quality_gate.sh` from the repo root.
4. Only merge when backend tests, frontend build, lint, and benchmark reports pass.

The theme: convert the current dragon-egg MVP into a demoable research product with reliable ingestion, measurable retrieval quality, and a polished scientific search experience.
