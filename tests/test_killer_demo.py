"""Tests for killer demo pipeline."""
import subprocess
import sys


def test_dry_run():
    r = subprocess.run(
        [sys.executable, "scripts/run_killer_demo.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout or "dry" in r.stdout.lower()


def test_syntax():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/run_killer_demo.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
