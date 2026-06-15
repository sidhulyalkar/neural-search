# tests/test_evaluate_usefulness_correlation.py
import subprocess
import sys
from pathlib import Path


def test_script_has_no_syntax_errors():
    result = subprocess.run(
        [sys.executable, "-c",
         "import ast; ast.parse(open('scripts/evaluate_usefulness_correlation.py').read())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_dry_run_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_usefulness_correlation.py", "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY RUN" in result.stdout
