"""Deterministic execution harness for audit verification.

This module intentionally does not accept arbitrary shell strings. Commands are argument
vectors executed with ``shell=False`` and a bounded timeout. Production deployments should
place this runner inside a container or job sandbox with additional filesystem/network limits.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from neural_search.software.schema import VerificationLevel, VerificationRun


class VerificationSpec(BaseModel):
    verification_id: str
    hypothesis_id: str
    level: VerificationLevel
    command: list[str]
    workdir: Path
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    env: dict[str, str] = Field(default_factory=dict)
    expected_returncode: int = 0
    expected_stdout_contains: list[str] = Field(default_factory=list)
    network_required: bool = False

    @field_validator("command")
    @classmethod
    def command_must_be_argv(cls, value: list[str]) -> list[str]:
        if not value or not all(part.strip() for part in value):
            raise ValueError("verification command must be a non-empty argv list")
        return value


class ExecutionResult(BaseModel):
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    output_sha256: str


def execute_verification(spec: VerificationSpec) -> tuple[VerificationRun, ExecutionResult]:
    """Execute one pre-reviewed verification specification and capture its evidence."""

    workdir = spec.workdir.resolve()
    if not workdir.is_dir():
        raise FileNotFoundError(f"verification workdir does not exist: {workdir}")

    env = os.environ.copy()
    env.update(spec.env)
    try:
        completed = subprocess.run(
            spec.command,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            check=False,
            shell=False,
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timed_out = True

    rendered = f"returncode={returncode}\nstdout={stdout}\nstderr={stderr}\ntimed_out={timed_out}\n"
    output_sha = hashlib.sha256(rendered.encode()).hexdigest()
    content_ok = all(fragment in stdout for fragment in spec.expected_stdout_contains)
    passed = not timed_out and returncode == spec.expected_returncode and content_ok
    result = ExecutionResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        output_sha256=output_sha,
    )
    verification = VerificationRun(
        verification_id=spec.verification_id,
        hypothesis_id=spec.hypothesis_id,
        level=spec.level,
        command=spec.command,
        environment={key: spec.env[key] for key in sorted(spec.env)},
        observed={
            "returncode": returncode,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            "timed_out": timed_out,
        },
        expected={
            "returncode": spec.expected_returncode,
            "stdout_contains": spec.expected_stdout_contains,
        },
        passed=passed,
        metadata={
            "workdir": str(workdir),
            "network_required": spec.network_required,
            "execution_output_sha256": output_sha,
        },
    )
    return verification, result
