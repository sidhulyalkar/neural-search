"""CLI for matching analysis recipes to task ontology IDs."""

from __future__ import annotations

import argparse
import json

from neural_search.recipes import get_recipe, match_recipes_for_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.recipes.match")
    parser.add_argument("--task", action="append", default=[], help="Task ontology ID to match.")
    parser.add_argument("--recipe", default=None, help="Recipe ID to inspect.")
    args = parser.parse_args(argv)

    if args.recipe:
        recipe = get_recipe(args.recipe)
        if recipe is None:
            print(json.dumps({"error": f"Recipe not found: {args.recipe}"}, indent=2))
            return 1
        print(json.dumps(recipe, indent=2))
        return 0

    if not args.task:
        parser.error("--task or --recipe is required")

    matches = match_recipes_for_tasks(args.task)
    print(json.dumps({"tasks": args.task, "recipes": matches}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
