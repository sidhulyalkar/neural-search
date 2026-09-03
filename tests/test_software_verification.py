"""Tests for deterministic software verification execution."""

from __future__ import annotations

import sys
from pathlib import Path

from neural_search.software.schema import VerificationLevel
from neural_search.software.verification import VerificationSpec, execute_verification


def test_verification_runner_captures_passing_evidence(tmp_path: Path) -> None:
    spec = VerificationSpec(
        verification_id="v1",
        hypothesis_id="h1",
        level=VerificationLevel.EXECUTABLE,
        command=[sys.executable, "-c", "print('oracle=0.5')"],
        workdir=tmp_path,
        expected_stdout_contains=["oracle=0.5"],
    )

    verification, result = execute_verification(spec)

    assert verification.passed is True
    assert result.returncode == 0
    assert len(result.output_sha256) == 64
    assert verification.observed["stdout_sha256"]


def test_verification_runner_marks_missing_expected_output_as_failure(tmp_path: Path) -> None:
    spec = VerificationSpec(
        verification_id="v2",
        hypothesis_id="h1",
        level=VerificationLevel.NUMERICAL_ORACLE,
        command=[sys.executable, "-c", "print('actual=0.2')"],
        workdir=tmp_path,
        expected_stdout_contains=["expected=0.4"],
    )

    verification, _ = execute_verification(spec)
    assert verification.passed is False
