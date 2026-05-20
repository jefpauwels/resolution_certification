# Code for the genuine resolution manuscript

This folder contains the code accompanying the genuine resolution manuscript.
The single computational source file is `Resolution.py`; it contains all
routines needed to reproduce the numerical tables reported in the manuscript.

## Files

- `Resolution.py` - Computes the trusted
  calibration witnesses, the intensity-bounded witnesses, the experimental
  certification table, the effective efficiencies in the experimental summary,
  and the efficiency-limited intrinsic-resolution table.
- `experimental_data.py` - intensities, rounded click probabilities, observed
  guessing probabilities, and certification cases used for the manuscript.
- `release_smoke_tests.py` - small reproducibility checks for the numerical
  values reported in the manuscript.
- `requirements.txt` - Python dependencies.
- `LICENSE` - MIT license for the code in this folder.

Local development artifacts such as `.venv`, `.DS_Store`, and `__pycache__`
should not be included in an arXiv/source-code archive.

## Requirements

The Python code requires Python 3.9 or newer plus NumPy and SciPy.

```bash
cd "final codes"
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The `.venv` directory is intentionally not part of the release archive.

## Quick Reproducibility Check

Run:

```bash
cd "final codes"
./.venv/bin/python release_smoke_tests.py
```

Expected output:

```text
All smoke tests passed.
```

The smoke tests verify:

- the trusted and intensity-bounded bounds in Table `tab:certification`;
- the observed guessing probabilities used in Tables `tab:certification` and
  `tab:summary`;
- the effective efficiencies in Table `tab:summary`;
- a small subset of the efficiency-limited thresholds in Table `tab:eff-lim`.

## Main Python Entry Points

Reproduce the certification table:

```bash
./.venv/bin/python -c "import Resolution as R; print(R.format_certification_table(R.build_manuscript_certification_table()))"
```

Reproduce the effective efficiencies:

```bash
./.venv/bin/python -c "import Resolution as R, experimental_data as D; print([round(100*R.solve_efficiency_for_target(D.INTENSITIES[:n], D.OBSERVED_GUESSING_PROBABILITIES[n], photon_cutoff=8, denominator=n), 2) for n in range(2, 8)])"
```

Reproduce the efficiency-limited benchmark table. The full table uses many
linear programs and may take longer than the smoke tests:

```bash
./.venv/bin/python -c "import Resolution as R; print(R.format_efficiency_table(R.resolution_table(n_levels=range(2, 9), resolutions=range(3, 10), tol=5e-4)))"
```

Running `Resolution.py` directly prints a compact demonstration of the main
calculations:

```bash
./.venv/bin/python Resolution.py
```

## Using New Data

To analyse another experiment, copy `experimental_data.py`, replace the
intensities and probability/guessing tables, and call:

```python
import Resolution as R
import my_experimental_data as D

cases = R.certification_configs_from_records(D.CERTIFICATION_CASES)
rows = R.build_certification_table(
    mus=D.INTENSITIES,
    cases=cases,
    measurement_probabilities=D.CLICK_PROBABILITIES,
    observed_guesses=D.OBSERVED_GUESSING_PROBABILITIES,
)
print(R.format_certification_table(rows))
```

## License

The code in this folder is released under the MIT license; see `LICENSE`.
