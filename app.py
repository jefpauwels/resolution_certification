"""Streamlit interface for photon-number resolution certification."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd
import streamlit as st

import Resolution as R
import experimental_data as D


PROBABILITY_COLUMNS = [f"p({i})" for i in range(8)] + ["p(>=8)"]
CASE_COLUMNS = ["N", "m", "R", "I"]


def default_intensities_frame() -> pd.DataFrame:
    return pd.DataFrame({"mu": list(D.INTENSITIES)})


def default_probabilities_frame() -> pd.DataFrame:
    return pd.DataFrame(D.CLICK_PROBABILITIES, columns=PROBABILITY_COLUMNS)


def default_cases_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "N": num_inputs,
                "m": max_photons,
                "R": resolution,
                "I": "" if intensity_cap is None else intensity_cap,
            }
            for num_inputs, max_photons, resolution, intensity_cap in D.CERTIFICATION_CASES
        ],
        columns=CASE_COLUMNS,
    )


def read_csv_upload(uploaded_file, fallback: pd.DataFrame) -> pd.DataFrame:
    if uploaded_file is None:
        return fallback.copy()
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        return fallback.copy()


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce")


def parse_intensities(frame: pd.DataFrame) -> tuple[float, ...]:
    numeric = numeric_frame(frame)
    if "mu" in numeric.columns:
        values = numeric["mu"]
    else:
        values = numeric.iloc[:, 0]
    values = values.dropna()
    return tuple(float(value) for value in values)


def parse_probabilities(frame: pd.DataFrame) -> tuple[tuple[float, ...], ...]:
    numeric = numeric_frame(frame)
    if "mu" in numeric.columns and len(numeric.columns) > 1:
        numeric = numeric.drop(columns=["mu"])
    numeric = numeric.dropna(how="all")
    return tuple(tuple(float(value) for value in row) for row in numeric.to_numpy())


def parse_cases(frame: pd.DataFrame) -> tuple[R.CertificationConfig, ...]:
    numeric = numeric_frame(frame.rename(columns=str.strip))
    if not set(CASE_COLUMNS[:3]).issubset(numeric.columns):
        raise ValueError("Cases must contain columns N, m, and R.")

    cases: list[R.CertificationConfig] = []
    for _, row in numeric.dropna(how="all").iterrows():
        if pd.isna(row["N"]) or pd.isna(row["m"]) or pd.isna(row["R"]):
            continue
        intensity_cap = None
        if "I" in numeric.columns and not pd.isna(row["I"]):
            intensity_cap = float(row["I"])
        cases.append(
            R.CertificationConfig(
                num_inputs=int(row["N"]),
                max_photons=int(row["m"]),
                resolution=int(row["R"]),
                intensity_cap=intensity_cap,
            )
        )
    return tuple(cases)


def validate_inputs(
    mus: tuple[float, ...],
    probabilities: tuple[tuple[float, ...], ...],
    cases: Iterable[R.CertificationConfig],
) -> bool:
    ok = True
    if not mus:
        st.error("At least one intensity is required.")
        return False
    if any(mu < 0 for mu in mus):
        st.error("Intensities must be non-negative.")
        ok = False
    if any(mus[i] > mus[i + 1] for i in range(len(mus) - 1)):
        st.warning("Intensities are not sorted. The certification routines use the row order as given.")

    if not probabilities:
        st.error("At least one probability row is required.")
        return False
    row_lengths = {len(row) for row in probabilities}
    if len(row_lengths) != 1:
        st.error("All probability rows must have the same number of columns.")
        ok = False
    if any(value < 0 for row in probabilities for value in row):
        st.error("Probabilities must be non-negative.")
        ok = False

    for index, row in enumerate(probabilities, start=1):
        total = sum(row)
        if abs(total - 1.0) > 1e-3:
            st.warning(
                f"Probability row {index} sums to {total:.6f}. "
                "Resolution.py normalizes rows before computing observed P_guess."
            )

    for case in cases:
        if case.num_inputs <= 0 or case.max_photons < 0 or case.resolution <= 0:
            st.error(f"Invalid case: {asdict(case)}")
            ok = False
        if case.num_inputs > len(mus):
            st.error(f"Case N={case.num_inputs} uses more intensities than provided.")
            ok = False
        if case.num_inputs > len(probabilities):
            st.error(f"Case N={case.num_inputs} uses more probability rows than provided.")
            ok = False
        if case.resolution > case.num_inputs:
            st.error(f"Case R={case.resolution}, N={case.num_inputs}: R cannot exceed N.")
            ok = False
        if case.intensity_cap is not None and case.intensity_cap > case.max_photons + 1:
            st.warning(
                f"Case N={case.num_inputs}, m={case.max_photons}, R={case.resolution}: "
                "I is larger than m+1 and will be clipped by Resolution.py."
            )
    return ok


def observed_guess_map(
    probabilities: tuple[tuple[float, ...], ...],
    cases: tuple[R.CertificationConfig, ...],
    use_manuscript_values: bool,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for case in cases:
        if use_manuscript_values and case.num_inputs in D.OBSERVED_GUESSING_PROBABILITIES:
            values[case.num_inputs] = D.OBSERVED_GUESSING_PROBABILITIES[case.num_inputs]
        else:
            values[case.num_inputs] = R.experimental_guessing_probability(probabilities, case.num_inputs)
    return values


def rows_to_frame(rows: list[R.CertificationRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        p_guess = row.experimental_guess
        records.append(
            {
                "m": row.max_photons,
                "N": row.num_inputs,
                "R": row.resolution,
                "trusted bound [%]": 100.0 * row.trusted_bound,
                "intensity-bound [%]": 100.0 * row.untrusted_bound,
                "P_guess [%]": None if p_guess is None else 100.0 * p_guess,
                "trusted certified resolution": row.certified_resolution_trusted(),
                "intensity certified resolution": row.certified_resolution_intensity(),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    st.set_page_config(page_title="PNR Resolution Certification", layout="wide")
    st.title("PNR Resolution Certification")

    with st.sidebar:
        st.header("Options")
        use_manuscript_values = st.checkbox(
            "Use exact manuscript P_guess values when available",
            value=True,
            help="For the bundled example data this reproduces the manuscript table. "
            "Turn this off to compute P_guess directly from the probability table.",
        )
        st.caption("The app calls Resolution.py without modifying the numerical routines.")

    st.subheader("Input intensities")
    intensity_upload = st.file_uploader("Upload intensity CSV", type="csv", key="intensity_csv")
    intensities_frame = read_csv_upload(intensity_upload, default_intensities_frame())
    intensities_frame = st.data_editor(
        intensities_frame,
        num_rows="dynamic",
        use_container_width=True,
        key="intensities_table",
    )

    st.subheader("Click probabilities")
    probability_upload = st.file_uploader("Upload probability CSV", type="csv", key="probability_csv")
    probabilities_frame = read_csv_upload(probability_upload, default_probabilities_frame())
    probabilities_frame = st.data_editor(
        probabilities_frame,
        num_rows="dynamic",
        use_container_width=True,
        key="probabilities_table",
    )

    st.subheader("Certification cases")
    cases_upload = st.file_uploader("Upload cases CSV", type="csv", key="cases_csv")
    cases_frame = read_csv_upload(cases_upload, default_cases_frame())
    cases_frame = st.data_editor(
        cases_frame,
        num_rows="dynamic",
        use_container_width=True,
        key="cases_table",
    )

    if st.button("Compute witnesses", type="primary"):
        try:
            mus = parse_intensities(intensities_frame)
            probabilities = parse_probabilities(probabilities_frame)
            cases = parse_cases(cases_frame)
        except Exception as exc:
            st.error(f"Could not parse inputs: {exc}")
            return

        if not cases:
            st.error("At least one certification case is required.")
            return
        if not validate_inputs(mus, probabilities, cases):
            return

        guesses = observed_guess_map(probabilities, cases, use_manuscript_values)
        try:
            rows = R.build_certification_table(
                mus=mus,
                cases=cases,
                measurement_probabilities=probabilities,
                observed_guesses=guesses,
            )
        except Exception as exc:
            st.error(f"Could not compute witnesses: {exc}")
            return

        result = rows_to_frame(rows)
        st.subheader("Results")
        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True,
            column_config={
                "trusted bound [%]": st.column_config.NumberColumn(format="%.2f"),
                "intensity-bound [%]": st.column_config.NumberColumn(format="%.2f"),
                "P_guess [%]": st.column_config.NumberColumn(format="%.2f"),
            },
        )


if __name__ == "__main__":
    main()
