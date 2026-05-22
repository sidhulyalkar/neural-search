"""CLI for ontology validation."""

from __future__ import annotations

import argparse

from neural_search.ontology.loader import OntologyValidationError, validate_ontology


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ontology.validate")
    parser.add_argument("path")
    args = parser.parse_args(argv)

    try:
        ontology = validate_ontology(args.path)
    except OntologyValidationError as exc:
        print(f"Ontology invalid: {exc}")
        return 1

    print(
        f"Ontology valid: {len(ontology.tasks)} tasks, "
        f"{len(ontology.behavior_labels)} behavior labels, "
        f"{len(ontology.modality_names)} modalities, "
        f"{len(ontology.region_names)} brain regions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

