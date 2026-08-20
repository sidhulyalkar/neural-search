import tomllib
from pathlib import Path


def test_execution_profile_extras_do_not_self_reference_package():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = payload["project"]["optional-dependencies"]

    for profile in ("researcher", "corpus-builder", "evaluator", "full-stack", "all"):
        requirements = extras[profile]
        assert requirements
        assert all(
            not requirement.lower().startswith("neural-search")
            for requirement in requirements
        ), f"{profile} must be directly resolvable without recursive self-dependencies"
