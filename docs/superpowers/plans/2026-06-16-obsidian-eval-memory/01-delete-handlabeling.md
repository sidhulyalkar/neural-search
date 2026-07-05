# Task 01 — Delete Hand-Labeling Code

**Files affected:**
- Delete: `neural_search/labeling/` (entire package)
- Delete: `neural_search/evaluation/label_relevance.py`
- Delete: `neural_search/evaluation/relevance.py`
- Delete: `scripts/run_annotation_session.py`
- Delete: `scripts/eval/annotate_candidates.py`
- Delete: `tests/test_relevance.py`
- Modify: `neural_search/evaluation/__init__.py`
- Modify: `tests/test_search_quality.py`

---

- [ ] **Step 1: Delete the labeling package and annotation scripts**

```bash
rm -rf neural_search/labeling/
rm neural_search/evaluation/label_relevance.py
rm neural_search/evaluation/relevance.py
rm scripts/run_annotation_session.py
rm scripts/eval/annotate_candidates.py
rm tests/test_relevance.py
```

- [ ] **Step 2: Check what the evaluation __init__ exports and remove labeling references**

```bash
grep -n "relevance\|labeling\|RelevanceJudgment\|RelevanceLabelSet" neural_search/evaluation/__init__.py
```

Open `neural_search/evaluation/__init__.py` and remove any lines that import from
`neural_search.evaluation.relevance` or `neural_search.evaluation.label_relevance`.
These are the hand-labeling types that no longer exist.

- [ ] **Step 3: Fix test_search_quality.py**

`tests/test_search_quality.py` currently imports from `neural_search.evaluation.relevance`.
Those functions (`compute_hard_negative_violations`, `compute_human_precision`,
`load_relevance_labels`) will not exist until the new package is built in Task 07
(label_ensemble.py). For now, stub the import out so the test file is importable:

Replace the broken import block at the top of `tests/test_search_quality.py`:

```python
# OLD (delete these lines):
from neural_search.evaluation.relevance import (
    RelevanceJudgment,
    RelevanceLabelSet,
    compute_hard_negative_violations,
    compute_human_precision,
    load_relevance_labels,
)

# NEW (replace with):
# Hard-negative violation helpers will be re-imported from
# neural_search.eval.label_ensemble once Task 07 is complete.
# Skipping relevance-label-dependent tests until then.
import pytest
pytestmark = pytest.mark.skip(reason="Migrating to neural_search.eval — re-enable after Task 07")
```

- [ ] **Step 4: Verify no remaining imports of the deleted modules**

```bash
grep -r "neural_search.labeling\|evaluation.relevance\|evaluation.label_relevance\|annotate_candidates\|run_annotation_session" \
  --include="*.py" . | grep -v "__pycache__"
```

Expected: no output (or only comments).

- [ ] **Step 5: Run the test suite to confirm no import errors**

```bash
pytest tests/ -x --ignore=tests/test_search_quality.py -q 2>&1 | tail -20
```

Expected: all existing tests pass; no `ModuleNotFoundError` for deleted modules.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove hand-labeling evaluation code

Deletes neural_search/labeling/, evaluation/relevance.py,
evaluation/label_relevance.py, scripts/run_annotation_session.py,
scripts/eval/annotate_candidates.py, tests/test_relevance.py.
Stubs test_search_quality.py pending migration to neural_search.eval."
```
