"""Small reproducibility checks for the public release code."""

from __future__ import annotations

import logging

import pandas as pd

logging.disable(logging.WARNING)
import app
logging.disable(logging.NOTSET)

import Resolution as R
import experimental_data


def assert_close(name: str, got: float, expected: float, tol: float) -> None:
    if abs(got - expected) > tol:
        raise AssertionError(f"{name}: got {got}, expected {expected} +/- {tol}")


def assert_percent(name: str, got: float, expected_percent: float, tol: float = 0.01) -> None:
    assert_close(name, 100.0 * got, expected_percent, tol)


def assert_raises(name: str, expected_error: type[Exception], func) -> None:
    try:
        func()
    except expected_error:
        return
    except Exception as exc:
        raise AssertionError(f"{name}: expected {expected_error.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"{name}: expected {expected_error.__name__}")


def check_certification_table() -> None:
    expected = [
        (1, 2, 1, 66.10, 66.73, 81.90, 2, 2),
        (2, 3, 2, 66.28, 67.32, 65.16, 2, 2),
        (3, 3, 2, 63.45, 64.97, 65.16, 3, 3),
        (4, 4, 3, 56.23, 57.04, 54.75, 3, 3),
        (5, 5, 3, 48.67, 49.64, 47.98, 3, 3),
        (6, 6, 3, 46.28, 47.09, 45.03, 3, 3),
        (7, 7, 3, 40.87, 41.59, 40.40, 3, 3),
        (8, 7, 3, 39.65, 40.86, 40.40, 4, 3),
    ]
    rows = R.build_manuscript_certification_table()
    if len(rows) != len(expected):
        raise AssertionError(f"Expected {len(expected)} certification rows, got {len(rows)}")
    for row, (m, n_inputs, resolution, trusted, untrusted, observed, cert_trusted, cert_intensity) in zip(rows, expected):
        if (row.max_photons, row.num_inputs, row.resolution) != (m, n_inputs, resolution):
            raise AssertionError("Certification row order or labels changed")
        assert_percent(f"trusted m={m}", row.trusted_bound, trusted)
        assert_percent(f"untrusted m={m}", row.untrusted_bound, untrusted)
        assert_percent(f"observed N={n_inputs}", row.experimental_guess or 0.0, observed)
        if row.certified_resolution_trusted() != cert_trusted:
            raise AssertionError(f"Trusted certification mismatch for m={m}")
        if row.certified_resolution_intensity() != cert_intensity:
            raise AssertionError(f"Intensity-bounded certification mismatch for m={m}")


def check_effective_efficiencies() -> None:
    expected = {2: 87.79, 3: 84.43, 4: 83.16, 5: 82.83, 6: 79.48, 7: 79.48}
    for num_inputs, expected_eta in expected.items():
        eta = R.solve_efficiency_for_target(
            experimental_data.INTENSITIES[:num_inputs],
            experimental_data.OBSERVED_GUESSING_PROBABILITIES[num_inputs],
            photon_cutoff=8,
            denominator=num_inputs,
        )
        assert_percent(f"eta N={num_inputs}", eta, expected_eta)


def check_certified_efficiencies() -> None:
    expected = {
        1: (86.89, 82.49),
        2: (82.54, 78.22),
        3: (84.20, 81.59),
        4: (82.71, 80.00),
        5: (82.20, 79.29),
        6: (77.07, 72.04),
        7: (77.42, 72.38),
        8: (78.47, 74.27),
    }
    rows = R.build_manuscript_certification_table()
    for row in rows:
        trusted, untrusted = expected[row.max_photons]
        if row.trusted_efficiency is None or row.untrusted_efficiency is None:
            raise AssertionError(f"Missing certified efficiencies for m={row.max_photons}")
        assert_percent(
            f"trusted certified eta m={row.max_photons}",
            row.trusted_efficiency,
            trusted,
        )
        assert_percent(
            f"intensity certified eta m={row.max_photons}",
            row.untrusted_efficiency,
            untrusted,
        )


def check_efficiency_limited_thresholds() -> None:
    table = R.efficiency_resolution_table(n_levels=(2, 3), resolutions=(3, 4), tol=5e-4)
    assert_close("threshold [0..2], R=3", table[2][3], 61.8, 0.1)
    assert_close("threshold [0..3], R=3", table[3][3], 50.0, 0.1)
    assert_close("threshold [0..3], R=4", table[3][4], 81.1, 0.1)


def check_generalized_tail() -> None:
    old_tail = R.poisson_tail(4, 3.2)
    new_tail = R.worst_case_poisson_tail(4, 3.2)
    assert_close("generalized tail reduces to Poisson tail", new_tail, old_tail, 1e-14)


def check_safety_margin_intensity_caps() -> None:
    cases = (
        R.CertificationConfig(num_inputs=6, max_photons=6, resolution=3, intensity_cap=1.03 * 6.9896),
        R.CertificationConfig(num_inputs=7, max_photons=7, resolution=3, intensity_cap=1.03 * 7.9741),
    )
    rows = R.build_certification_table(
        mus=experimental_data.INTENSITIES,
        cases=cases,
        observed_guesses=experimental_data.OBSERVED_GUESSING_PROBABILITIES,
    )
    expected = {
        6: (47.09, 72.04),
        7: (41.59, 72.38),
    }
    for row in rows:
        bound, eta = expected[row.max_photons]
        assert_percent(f"3% safety bound m={row.max_photons}", row.untrusted_bound, bound)
        assert_percent(f"3% safety eta m={row.max_photons}", row.untrusted_efficiency or 0.0, eta)
        if row.certified_resolution_intensity() != 3:
            raise AssertionError(f"Unexpected 3% safety certified resolution for m={row.max_photons}")


def check_app_custom_probabilities_not_overridden() -> None:
    cases = (R.CertificationConfig(num_inputs=2, max_photons=1, resolution=1, intensity_cap=1.0),)
    probabilities = ((1.0, 0.0), (1.0, 0.0))
    use_exact = app.should_use_exact_manuscript_values((0.0, 1.0), probabilities, cases, requested=True)
    if use_exact:
        raise AssertionError("Custom data incorrectly matched bundled manuscript data")
    observed = app.observed_guess_map(probabilities, cases, use_exact)
    assert_close("custom app P_guess", observed[2], 0.5, 1e-14)


def check_app_bundled_manuscript_values_available() -> None:
    mus = app.parse_intensities(app.default_intensities_frame())
    probabilities = app.parse_probabilities(app.default_probabilities_frame())
    cases = app.parse_cases(app.default_cases_frame())
    use_exact = app.should_use_exact_manuscript_values(mus, probabilities, cases, requested=True)
    if not use_exact:
        raise AssertionError("Bundled manuscript data was not recognized")
    observed = app.observed_guess_map(probabilities, cases, use_exact)
    assert_close("bundled exact P_guess N=2", observed[2], 0.8190, 1e-14)


def check_app_rejects_nonnumeric_cells() -> None:
    assert_raises(
        "nonnumeric probability cell",
        ValueError,
        lambda: app.parse_probabilities(pd.DataFrame({"p0": ["bad"], "p1": [1.0]})),
    )
    assert_raises(
        "nonnumeric intensity cell",
        ValueError,
        lambda: app.parse_intensities(pd.DataFrame({"mu": [0.0, "bad"]})),
    )


def check_app_rejects_partial_cases() -> None:
    assert_raises(
        "partial certification case",
        ValueError,
        lambda: app.parse_cases(pd.DataFrame({"N": [2], "m": [1], "R": [""], "I": [""]})),
    )
    blank_cases = app.parse_cases(pd.DataFrame({"N": [""], "m": [""], "R": [""], "I": [""]}))
    if blank_cases:
        raise AssertionError("Blank case rows should be ignored")


def check_app_rejects_resolution_above_subspace_dimension() -> None:
    case = R.CertificationConfig(num_inputs=2, max_photons=0, resolution=2, intensity_cap=1.0)
    errors = app.case_validation_errors(case, num_intensities=2, num_probability_rows=2)
    if not any("R cannot exceed m+1" in error for error in errors):
        raise AssertionError("Expected R > m+1 validation error")


def main() -> None:
    check_certification_table()
    check_effective_efficiencies()
    check_certified_efficiencies()
    check_efficiency_limited_thresholds()
    check_generalized_tail()
    check_safety_margin_intensity_caps()
    check_app_custom_probabilities_not_overridden()
    check_app_bundled_manuscript_values_available()
    check_app_rejects_nonnumeric_cells()
    check_app_rejects_partial_cases()
    check_app_rejects_resolution_above_subspace_dimension()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
