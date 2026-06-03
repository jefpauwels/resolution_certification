# Code for the genuine resolution manuscript

Interactive app: https://resolutioncertification.streamlit.app

This folder contains the code accompanying the genuine resolution manuscript.
The single computational source file is `Resolution.py`; it contains all
routines needed to reproduce the numerical tables reported in the manuscript.

## Files

- `Resolution.py` - Computes the trusted
  calibration witnesses, the intensity-bounded witnesses, the experimental
  certification table, the effective efficiencies in the experimental summary,
  and the efficiency-limited intrinsic-resolution table.
- `experimental_data.py` - intensities, rounded click probabilities, observed
  guessing probabilities, and certification cases used for the manuscript. The
  bundled intensity-bounded cases use a 3% safety-corrected intensity cap by
  default.
- `release_smoke_tests.py` - small reproducibility checks for the numerical
  values reported in the manuscript.
- `requirements.txt` - Python dependencies.
- `LICENSE` - MIT license for the code in this folder.

Local development artifacts such as `.venv`, `.DS_Store`, and `__pycache__`
should not be included in an arXiv/source-code archive.

## Requirements

The Python code requires Python 3.9 or newer plus NumPy and SciPy. The Streamlit
app also uses Streamlit and pandas, all listed in `requirements.txt`.

```bash
cd resolution_certification
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The `.venv` directory is intentionally not part of the release archive.

## Quick Reproducibility Check

Run:

```bash
cd resolution_certification
./.venv/bin/python release_smoke_tests.py
```

Expected output:

```text
All smoke tests passed.
```

The smoke tests verify:

- the trusted and intensity-bounded bounds in Table `tab:certification`;
- the observed guessing probabilities used in Tables `tab:certification` and
  `tab:certified-efficiencies`;
- the certified effective efficiencies in Table `tab:certified-efficiencies`;
- a small subset of the efficiency-limited thresholds in Table `tab:eff-lim`.

## Main Python Entry Points

Reproduce the certification table:

```bash
./.venv/bin/python -c "import Resolution as R; print(R.format_certification_table(R.build_manuscript_certification_table()))"
```

Reproduce the certified effective efficiencies:

```bash
./.venv/bin/python -c "import Resolution as R; rows=R.build_manuscript_certification_table(); print([(r.max_photons, round(100*r.trusted_efficiency, 2), round(100*r.untrusted_efficiency, 2)) for r in rows])"
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
intensities, probability/guessing tables, and certification cases, and call:

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

For each certification case, set `intensity_cap` to the calibration bound you
want to assume. Use `None` to take the largest selected input intensity instead.

## License

The code in this folder is released under the MIT license; see `LICENSE`.
