# tests/test_evaluate_usefulness_correlation.py
import subprocess
import sys


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
        cwd="/mnt/c/Users/sidso/Documents/neural-search",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY RUN" in result.stdout
