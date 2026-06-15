"""Tests for ablate_graph_proximity.py."""
import subprocess
import sys

import pytest


def test_dry_run_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/ablate_graph_proximity.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "DRY RUN" in result.stdout


def test_syntax():
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/ablate_graph_proximity.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_graph_exits_nonzero(tmp_path, monkeypatch):
    from pathlib import Path
    repo_root = str(Path(__file__).parent.parent)
    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {repo_root!r}); "
         "from scripts.ablate_graph_proximity import main; sys.exit(main([]))"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "ERROR" in result.stdout


# Unit tests for NDCG calculation
def _import_ndcg():
    import importlib.util
    from pathlib import Path
    script_path = Path(__file__).parent.parent / "scripts" / "ablate_graph_proximity.py"
    spec = importlib.util.spec_from_file_location(
        "ablate_graph_proximity",
        str(script_path),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._ndcg, mod._dcg


def test_ndcg_perfect_ranking():
    _ndcg, _dcg = _import_ndcg()
    gains = [2.0, 0.0, 0.0, 0.0, 0.0]
    assert _ndcg(gains, gains) == pytest.approx(1.0)


def test_ndcg_zero_when_no_relevant():
    _ndcg, _dcg = _import_ndcg()
    gains = [0.0, 0.0, 0.0]
    # ideal is all zeros → idcg=0 → ndcg=0
    assert _ndcg(gains, gains) == pytest.approx(0.0)


def test_ndcg_partial_ranking():
    _ndcg, _dcg = _import_ndcg()
    perfect = [2.0, 0.0]
    worst = [0.0, 2.0]
    # Perfect ordering against its own ideal = 1.0 (relevant item at top = ideal)
    assert _ndcg(perfect, perfect) == pytest.approx(1.0)
    # Worst ordering against the same ideal pool is < 1.0 (relevant item at bottom)
    assert _ndcg(worst, perfect) < 1.0
    # Perfect always beats worst against same ideal
    assert _ndcg(perfect, perfect) > _ndcg(worst, perfect)


def test_dcg_ordering():
    _ndcg, _dcg = _import_ndcg()
    # DCG([2, 0]) = 2/log2(2) = 2.0; DCG([0, 2]) = 0 + 2/log2(3) < 2.0
    assert _dcg([2.0, 0.0]) > _dcg([0.0, 2.0])
