"""Entry point for python -m neural_search.evaluation."""

if __name__ == "__main__":
    import sys
    # Avoid import order warning by importing directly
    if "neural_search.evaluation.run_benchmark" not in sys.modules:
        from neural_search.evaluation.run_benchmark import main
    else:
        main = sys.modules["neural_search.evaluation.run_benchmark"].main
    raise SystemExit(main())
