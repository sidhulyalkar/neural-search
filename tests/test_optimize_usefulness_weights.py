# tests/test_optimize_usefulness_weights.py
import json
from pathlib import Path


def test_optimizer_script_syntax():
    import ast
    src = Path("scripts/optimize_usefulness_weights.py").read_text()
    ast.parse(src)  # raises SyntaxError if broken


def test_optimizer_dry_run(tmp_path):
    """--dry-run should print current weights without writing anything."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "scripts/optimize_usefulness_weights.py", "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "INTENT_WEIGHT_PROFILES" in result.stdout or "dry run" in result.stdout.lower()


def test_optimizer_produces_valid_weights(tmp_path):
    """Optimizer must produce weights that sum to ~1.0 per intent."""
    import subprocess
    import sys
    out_file = tmp_path / "weights_out.json"
    result = subprocess.run(
        [sys.executable, "scripts/optimize_usefulness_weights.py",
         "--n-trials", "2", "--out", str(out_file)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert out_file.exists(), "Output file not created"
    weights = json.loads(out_file.read_text())
    for intent, dims in weights.items():
        total = sum(dims.values())
        assert abs(total - 1.0) < 0.01, f"Intent {intent} weights sum to {total}"
