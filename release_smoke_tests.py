"""Small reproducibility checks for the public release code."""

from __future__ import annotations

import Resolution as R
import experimental_data


def assert_close(name: str, got: float, expected: float, tol: float) -> None:
    if abs(got - expected) > tol:
        raise AssertionError(f"{name}: got {got}, expected {expected} +/- {tol}")


def assert_percent(name: str, got: float, expected_percent: float, tol: float = 0.01) -> None:
    assert_close(name, 100.0 * got, expected_percent, tol)


def check_certification_table() -> None:
    expected = [
        (1, 2, 1, 66.10, 66.10, 81.90, True),
        (2, 3, 2, 66.28, 66.70, 65.16, False),
        (3, 3, 2, 63.45, 64.49, 65.16, True),
        (4, 4, 3, 56.23, 56.55, 54.75, False),
        (5, 5, 3, 48.67, 49.16, 47.98, False),
        (6, 6, 3, 46.28, 46.57, 45.03, False),
        (7, 7, 3, 40.87, 41.11, 40.40, False),
        (8, 7, 3, 39.65, 40.39, 40.40, True),
    ]
    rows = R.build_manuscript_certification_table()
    if len(rows) != len(expected):
        raise AssertionError(f"Expected {len(expected)} certification rows, got {len(rows)}")
    for row, (m, n_inputs, resolution, trusted, untrusted, observed, certified) in zip(rows, expected):
        if (row.max_photons, row.num_inputs, row.resolution) != (m, n_inputs, resolution):
            raise AssertionError("Certification row order or labels changed")
        assert_percent(f"trusted m={m}", row.trusted_bound, trusted)
        assert_percent(f"untrusted m={m}", row.untrusted_bound, untrusted)
        assert_percent(f"observed N={n_inputs}", row.experimental_guess or 0.0, observed)
        is_certified = (row.experimental_guess or 0.0) > row.trusted_bound
        if is_certified != certified:
            raise AssertionError(f"Certification flag mismatch for m={m}")


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


def check_efficiency_limited_thresholds() -> None:
    table = R.efficiency_resolution_table(n_levels=(2, 3), resolutions=(3, 4), tol=5e-4)
    assert_close("threshold [0..2], R=3", table[2][3], 61.8, 0.1)
    assert_close("threshold [0..3], R=3", table[3][3], 50.0, 0.1)
    assert_close("threshold [0..3], R=4", table[3][4], 81.1, 0.1)


def main() -> None:
    check_certification_table()
    check_effective_efficiencies()
    check_efficiency_limited_thresholds()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
