"""Evaluate spectral-parameterization exponent recovery on synthetic data.

Generates synthetic 1/f-like signals with known aperiodic exponents (with
and without injected oscillatory peaks), fits them with the available
backends, and reports the recovered exponent error against a documented
tolerance. Exits 1 if any backend exceeds tolerance on the no-peak cases
(peaked cases are reported but not gating, since even real FOOOF/specparam
accuracy degrades with strong peak contamination).

Usage:
    python scripts/eval/evaluate_aperiodic_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.spectral.psd import restrict_freq_range, welch_psd  # noqa: E402
from neural_search.spectral.schemas import SpectralRunConfig  # noqa: E402
from neural_search.spectral.specparam_backend import (  # noqa: E402
    BackendUnavailableError,
    MockSpectralParamBackend,
    SpecparamBackend,
)
from neural_search.spectral.synthetic import (  # noqa: E402
    SyntheticPeak,
    SyntheticSpectrumSpec,
    synthetic_aperiodic_signal,
)

# Relative tolerance on the no-peak case: |recovered - true| / true
NO_PEAK_TOLERANCE = 0.20
# Looser tolerance once a realistically-scaled oscillatory peak is present
WITH_PEAK_TOLERANCE = 0.35

SAMPLE_RATE_HZ = 250.0
N_SAMPLES = 20_000
FREQ_RANGE_HZ = (2.0, 40.0)
TRUE_EXPONENTS = (1.0, 1.5, 2.0, 2.5)


def _fit_exponent(backend, freqs, power, run_config) -> float:
    fit = backend.fit(freqs, power, run_config)
    return fit.aperiodic_exponent


def main() -> int:
    run_config = SpectralRunConfig(run_id="run:eval:synthetic", backend="mock", freq_range_hz=FREQ_RANGE_HZ)
    backends = {"mock": MockSpectralParamBackend()}
    try:
        backends["specparam"] = SpecparamBackend()
    except BackendUnavailableError:
        print("specparam/fooof not installed; evaluating mock backend only.", file=sys.stderr)

    failures: list[str] = []
    rows: list[tuple[str, float, bool, float, float]] = []

    for backend_name, backend in backends.items():
        for true_exponent in TRUE_EXPONENTS:
            spec = SyntheticSpectrumSpec(exponent=true_exponent, offset=-2.0)
            signal = synthetic_aperiodic_signal(N_SAMPLES, SAMPLE_RATE_HZ, spec, seed=0)
            freqs, power = welch_psd(signal, SAMPLE_RATE_HZ, nperseg=500)
            freqs, power = restrict_freq_range(freqs, power, FREQ_RANGE_HZ)
            recovered = _fit_exponent(backend, freqs, power, run_config)
            error = abs(recovered - true_exponent) / true_exponent
            ok = error <= NO_PEAK_TOLERANCE
            rows.append((backend_name, true_exponent, False, recovered, error))
            if not ok:
                failures.append(
                    f"{backend_name} no-peak exponent={true_exponent}: recovered={recovered:.3f} "
                    f"error={error:.2%} > tolerance {NO_PEAK_TOLERANCE:.0%}"
                )

            spec_with_peak = SyntheticSpectrumSpec(
                exponent=true_exponent,
                offset=-2.0,
                peaks=(SyntheticPeak(center_frequency_hz=10.0, power=0.0008, bandwidth_hz=1.5),),
            )
            signal_with_peak = synthetic_aperiodic_signal(N_SAMPLES, SAMPLE_RATE_HZ, spec_with_peak, seed=0)
            freqs_p, power_p = welch_psd(signal_with_peak, SAMPLE_RATE_HZ, nperseg=500)
            freqs_p, power_p = restrict_freq_range(freqs_p, power_p, FREQ_RANGE_HZ)
            recovered_p = _fit_exponent(backend, freqs_p, power_p, run_config)
            error_p = abs(recovered_p - true_exponent) / true_exponent
            rows.append((backend_name, true_exponent, True, recovered_p, error_p))
            if error_p > WITH_PEAK_TOLERANCE:
                print(
                    f"[info, non-gating] {backend_name} with-peak exponent={true_exponent}: "
                    f"recovered={recovered_p:.3f} error={error_p:.2%} > {WITH_PEAK_TOLERANCE:.0%}",
                    file=sys.stderr,
                )

    print(f"{'backend':<10} {'true':>6} {'peak':>5} {'recovered':>10} {'error':>8}")
    for backend_name, true_exponent, has_peak, recovered, error in rows:
        print(f"{backend_name:<10} {true_exponent:>6.2f} {str(has_peak):>5} {recovered:>10.3f} {error:>8.2%}")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll no-peak exponent-recovery checks passed within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
