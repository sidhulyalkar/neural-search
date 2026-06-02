"""Tests for build and validate turbovec index scripts."""
import subprocess
import sys


def test_dry_run_build():
    r = subprocess.run(
        [sys.executable, "scripts/build_turbovec_index.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout


def test_dry_run_validate():
    r = subprocess.run(
        [sys.executable, "scripts/validate_turbovec_recall.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout


def test_syntax_build():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/build_turbovec_index.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_syntax_validate():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/validate_turbovec_recall.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
