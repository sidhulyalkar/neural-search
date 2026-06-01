# Neural Search: Cleanup Log

**Purpose:** Track repository cleanup actions and artifact management.

**Last Updated:** 2026-05-27

---

## Cleanup Session: 2026-05-27

### Starting State

- **Branch:** `claude/full-paper-and-experiment-upgrade`
- **Python:** 3.13.5
- **Tests:** 697 collected, core tests passing
- **Lint:** Minor line length warnings only

### Actions Taken

#### 1. Cache Cleanup (Local)

Removed local Python cache directories:
- `__pycache__/` directories throughout codebase
- `*.pyc` files
- `.pytest_cache/`
- `.ruff_cache/`
- `.mypy_cache/`

Note: These were already gitignored, cleanup only affects working tree.

#### 2. Generated Artifacts Cleanup (Local)

Removed from working tree (already gitignored):
- `dist/` - Python wheel and sdist artifacts
- `apps/web/node_modules/` - Node dependencies
- `apps/web/dist/` - Frontend build artifacts
- Root-level `*.zip` handoff packages

#### 3. Export Script Added

Created `scripts/export_clean_repo.sh`:
- Uses `git archive` for deterministic clean exports
- Verifies no generated artifacts in output
- Reports file size

### Verified Gitignore Coverage

The following patterns are properly excluded by `.gitignore`:
- `__pycache__/`, `*.pyc`, `*.pyo`
- `dist/`, `build/`, `*.egg-info/`
- `node_modules/`, `apps/web/node_modules/`, `apps/web/dist/`
- `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- `*.zip`
- `data/eval/results/`, `data/raw/`

### Documentation State

Current canonical docs in `docs/`:
- `CLAIM_LEDGER.md` - Tracks implementation status of whitepaper claims
- `CURRENT_SYSTEM_MAP.md` - Module overview
- `ARCHITECTURE_V05.md` - System architecture
- `WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md` - Paper-code alignment

Historical/reference docs (retained for context):
- Various TASK_* docs, ROADMAP docs, HANDOFF docs
- Older version instruction sets

**Recommendation:** Consider moving clearly superseded docs to `archive_docs/` in future cleanup pass.

---

## Artifact Management Guidelines

### What Should Be Gitignored

1. **Build artifacts:** `dist/`, `build/`, `*.egg-info/`
2. **Cache files:** `__pycache__/`, `.pytest_cache/`, etc.
3. **Node artifacts:** `node_modules/`, frontend `dist/`
4. **Generated data:** `data/eval/results/`, `data/raw/`
5. **Handoff packages:** Root-level `*.zip` files
6. **Environment files:** `.env`, secrets

### What Should Be Committed

1. **Source code:** All `*.py` files in `neural_search/`
2. **Tests:** All files in `tests/`
3. **Configuration:** `pyproject.toml`, `*.yaml` configs
4. **Documentation:** Markdown files in `docs/`
5. **Seed data:** Schema examples in `data/seed/`
6. **Benchmark definitions:** Query sets in `data/eval/`

### Clean Export Command

```bash
./scripts/export_clean_repo.sh neural-search-clean.zip
```

This produces a clean zip using `git archive` that excludes all gitignored files.
