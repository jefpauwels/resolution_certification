"""Streamlit interface for photon-number resolution certification."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Iterable

import pandas as pd
import streamlit as st

import Resolution as R
import experimental_data as D


PROBABILITY_COLUMNS = [f"p({i})" for i in range(8)] + ["p(>=8)"]
DEFAULT_PROBABILITY_COLUMN_COUNT = len(PROBABILITY_COLUMNS)
CASE_COLUMNS = ["N", "m", "R", "I"]
MAX_CASES = 12
MAX_NUM_INPUTS = 8
MAX_MAX_PHOTONS = 8
MAX_RESOLUTION = 8
MAX_PROBABILITY_COLUMNS = 32
MATCH_TOLERANCE = 1e-12


def default_intensities_frame() -> pd.DataFrame:
    return pd.DataFrame({"mu": list(D.INTENSITIES)})


def default_probabilities_frame(num_columns: int = DEFAULT_PROBABILITY_COLUMN_COUNT) -> pd.DataFrame:
    frame = pd.DataFrame(D.CLICK_PROBABILITIES, columns=PROBABILITY_COLUMNS)
    return resize_probability_frame(frame, num_columns)


def resize_probability_frame(frame: pd.DataFrame, num_columns: int) -> pd.DataFrame:
    if num_columns < 1:
        raise ValueError("At least one probability column is required.")
    if num_columns == len(frame.columns):
        return frame.copy()
    if num_columns < len(frame.columns):
        resized = frame.iloc[:, :num_columns].copy()
        if num_columns < len(frame.columns):
            resized.iloc[:, -1] = frame.iloc[:, num_columns - 1 :].sum(axis=1)
        resized.columns = [f"p({i})" for i in range(num_columns - 1)] + [f"p(>={num_columns - 1})"]
        return resized

    resized = frame.copy()
    for index in range(len(frame.columns), num_columns):
        resized[f"p(extra {index - len(frame.columns) + 1})"] = 0.0
    return resized


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


def read_csv_upload(uploaded_file, fallback: pd.DataFrame, label: str) -> tuple[pd.DataFrame, bool]:
    if uploaded_file is None:
        return fallback.copy(), True
    try:
        return pd.read_csv(uploaded_file), True
    except Exception as exc:
        st.error(f"Could not read {label} CSV file: {exc}")
        return fallback.copy(), False


def is_blank_cell(value: object) -> bool:
    return pd.isna(value) or (isinstance(value, str) and value.strip() == "")


def numeric_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    records: list[dict[object, float]] = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        converted_row: dict[object, float] = {}
        for column in frame.columns:
            value = row[column]
            if is_blank_cell(value):
                converted_row[column] = math.nan
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label}: row {row_number}, column {column!r} must be numeric or blank."
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"{label}: row {row_number}, column {column!r} must be finite."
                )
            converted_row[column] = numeric_value
        records.append(converted_row)
    return pd.DataFrame(records, columns=frame.columns)


def require_integer(value: float, row_number: int, column: str) -> int:
    if not float(value).is_integer():
        raise ValueError(f"Cases: row {row_number}, column {column!r} must be an integer.")
    return int(value)


def parse_intensities(frame: pd.DataFrame) -> tuple[float, ...]:
    numeric = numeric_frame(frame, "Intensities")
    if "mu" in numeric.columns:
        values = numeric["mu"]
    else:
        values = numeric.iloc[:, 0]
    values = values.dropna()
    return tuple(float(value) for value in values)


def parse_probabilities(frame: pd.DataFrame) -> tuple[tuple[float, ...], ...]:
    numeric = numeric_frame(frame, "Probabilities")
    if "mu" in numeric.columns and len(numeric.columns) > 1:
        numeric = numeric.drop(columns=["mu"])
    numeric = numeric.dropna(how="all")
    numeric = numeric.fillna(0.0)
    return tuple(tuple(float(value) for value in row) for row in numeric.to_numpy())


def intensity_cap_for_case(mus: tuple[float, ...], case: R.CertificationConfig) -> float | None:
    if case.num_inputs > len(mus):
        return None
    return float(case.intensity_cap if case.intensity_cap is not None else max(mus[: case.num_inputs]))


def parse_cases(frame: pd.DataFrame) -> tuple[R.CertificationConfig, ...]:
    numeric = numeric_frame(frame.rename(columns=str.strip), "Cases")
    if not set(CASE_COLUMNS[:3]).issubset(numeric.columns):
        raise ValueError("Cases must contain columns N, m, and R.")

    cases: list[R.CertificationConfig] = []
    for row_number, (_, row) in enumerate(numeric.iterrows(), start=1):
        if row.dropna().empty:
            continue
        missing = [column for column in CASE_COLUMNS[:3] if pd.isna(row[column])]
        if missing:
            missing_columns = ", ".join(missing)
            raise ValueError(f"Cases: row {row_number} is missing required value(s): {missing_columns}.")
        intensity_cap = None
        if "I" in numeric.columns and not pd.isna(row["I"]):
            intensity_cap = float(row["I"])
        cases.append(
            R.CertificationConfig(
                num_inputs=require_integer(row["N"], row_number, "N"),
                max_photons=require_integer(row["m"], row_number, "m"),
                resolution=require_integer(row["R"], row_number, "R"),
                intensity_cap=intensity_cap,
            )
        )
    return tuple(cases)


def case_validation_errors(
    case: R.CertificationConfig,
    num_intensities: int,
    num_probability_rows: int,
) -> list[str]:
    errors: list[str] = []
    if case.num_inputs <= 0 or case.max_photons < 0 or case.resolution <= 0:
        errors.append(f"Invalid case: {asdict(case)}")
    if case.num_inputs > MAX_NUM_INPUTS:
        errors.append(f"Case N={case.num_inputs} exceeds the public app limit N <= {MAX_NUM_INPUTS}.")
    if case.max_photons > MAX_MAX_PHOTONS:
        errors.append(f"Case m={case.max_photons} exceeds the public app limit m <= {MAX_MAX_PHOTONS}.")
    if case.resolution > MAX_RESOLUTION:
        errors.append(f"Case R={case.resolution} exceeds the public app limit R <= {MAX_RESOLUTION}.")
    if case.num_inputs > num_intensities:
        errors.append(f"Case N={case.num_inputs} uses more intensities than provided.")
    if case.num_inputs > num_probability_rows:
        errors.append(f"Case N={case.num_inputs} uses more probability rows than provided.")
    if case.resolution > case.num_inputs:
        errors.append(f"Case R={case.resolution}, N={case.num_inputs}: R cannot exceed N.")
    if case.resolution > case.max_photons + 1:
        errors.append(
            f"Case R={case.resolution}, m={case.max_photons}: R cannot exceed m+1."
        )
    return errors


def validate_inputs(
    mus: tuple[float, ...],
    probabilities: tuple[tuple[float, ...], ...],
    cases: Iterable[R.CertificationConfig],
) -> bool:
    cases = tuple(cases)
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
    elif next(iter(row_lengths)) > MAX_PROBABILITY_COLUMNS:
        st.error(
            f"Probability tables are limited to {MAX_PROBABILITY_COLUMNS} columns "
            "in the public app."
        )
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

    if len(cases) > MAX_CASES:
        st.error(f"Certification tables are limited to {MAX_CASES} cases in the public app.")
        ok = False

    for case in cases:
        errors = case_validation_errors(case, len(mus), len(probabilities))
        for error in errors:
            st.error(error)
            ok = False
        if errors:
            continue
        cap = intensity_cap_for_case(mus, case)
        if cap is not None:
            if cap <= 0:
                st.error(
                    f"Case N={case.num_inputs}, m={case.max_photons}, R={case.resolution}: "
                    "the intensity bound I must be positive."
                )
                ok = False
            if cap > case.max_photons + 1 + 1e-12:
                st.info(
                    f"Case N={case.num_inputs}, m={case.max_photons}, R={case.resolution}: "
                    f"I = {cap:g} exceeds m+1 = {case.max_photons + 1}. "
                    "The app will use the generalized conservative tail correction."
                )
    return ok


def floats_close(left: float, right: float, tol: float = MATCH_TOLERANCE) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)


def sequences_close(left: Iterable[float], right: Iterable[float]) -> bool:
    left_tuple = tuple(left)
    right_tuple = tuple(right)
    return len(left_tuple) == len(right_tuple) and all(
        floats_close(a, b) for a, b in zip(left_tuple, right_tuple)
    )


def probabilities_match_manuscript(probabilities: tuple[tuple[float, ...], ...]) -> bool:
    if len(probabilities) != len(D.CLICK_PROBABILITIES):
        return False
    return all(
        sequences_close(row, manuscript_row)
        for row, manuscript_row in zip(probabilities, D.CLICK_PROBABILITIES)
    )


def cases_match_manuscript(cases: tuple[R.CertificationConfig, ...]) -> bool:
    manuscript_cases = R.MANUSCRIPT_CERTIFICATION_CASES
    if len(cases) != len(manuscript_cases):
        return False
    for case, manuscript_case in zip(cases, manuscript_cases):
        if (
            case.num_inputs != manuscript_case.num_inputs
            or case.max_photons != manuscript_case.max_photons
            or case.resolution != manuscript_case.resolution
        ):
            return False
        if case.intensity_cap is None or manuscript_case.intensity_cap is None:
            if case.intensity_cap != manuscript_case.intensity_cap:
                return False
        elif not floats_close(case.intensity_cap, manuscript_case.intensity_cap):
            return False
    return True


def inputs_match_manuscript(
    mus: tuple[float, ...],
    probabilities: tuple[tuple[float, ...], ...],
    cases: tuple[R.CertificationConfig, ...],
) -> bool:
    return (
        sequences_close(mus, D.INTENSITIES)
        and probabilities_match_manuscript(probabilities)
        and cases_match_manuscript(cases)
    )


def should_use_exact_manuscript_values(
    mus: tuple[float, ...],
    probabilities: tuple[tuple[float, ...], ...],
    cases: tuple[R.CertificationConfig, ...],
    requested: bool,
) -> bool:
    return requested and inputs_match_manuscript(mus, probabilities, cases)


def observed_guess_map(
    probabilities: tuple[tuple[float, ...], ...],
    cases: tuple[R.CertificationConfig, ...],
    use_exact_manuscript_values: bool,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for case in cases:
        if use_exact_manuscript_values and case.num_inputs in D.OBSERVED_GUESSING_PROBABILITIES:
            values[case.num_inputs] = D.OBSERVED_GUESSING_PROBABILITIES[case.num_inputs]
        else:
            values[case.num_inputs] = R.experimental_guessing_probability(probabilities, case.num_inputs)
    return values


def effective_intensity_cap(mus: tuple[float, ...], case: R.CertificationConfig) -> float:
    """Return the intensity cap used by Resolution.py for an intensity-bounded row."""

    cap = intensity_cap_for_case(mus, case)
    if cap is None:
        raise ValueError("Case uses more intensities than provided.")
    return cap


def rows_to_frame(
    rows: list[R.CertificationRow],
    cases: tuple[R.CertificationConfig, ...],
    mus: tuple[float, ...],
) -> pd.DataFrame:
    records = []
    for row, case in zip(rows, cases):
        p_guess = row.experimental_guess
        records.append(
            {
                "m": row.max_photons,
                "N": row.num_inputs,
                "R": row.resolution,
                "input I": None if case.intensity_cap is None else float(case.intensity_cap),
                "I used": effective_intensity_cap(mus, case),
                "trusted bound [%]": 100.0 * row.trusted_bound,
                "intensity-bound [%]": 100.0 * row.untrusted_bound,
                "P_guess [%]": None if p_guess is None else 100.0 * p_guess,
                "trusted eta_* [%]": (
                    None if row.trusted_efficiency is None else 100.0 * row.trusted_efficiency
                ),
                "intensity eta_* [%]": (
                    None if row.untrusted_efficiency is None else 100.0 * row.untrusted_efficiency
                ),
                "trusted certified resolution": row.certified_resolution_trusted(),
                "intensity certified resolution": row.certified_resolution_intensity(),
                "trusted status": certification_status(p_guess, row.trusted_bound),
                "intensity status": certification_status(p_guess, row.untrusted_bound),
            }
        )
    return pd.DataFrame(records)


def certification_status(observed: float | None, bound: float) -> str | None:
    if observed is None:
        return None
    return "Certified" if observed > bound else "Not certified"


def result_column_config() -> dict[str, object]:
    return {
        "input I": st.column_config.NumberColumn(
            format="%.4f",
            help=r"Optional certified intensity bound $I$ entered for the case.",
        ),
        "I used": st.column_config.NumberColumn(
            format="%.4f",
            help=r"Intensity bound $I$ used for the intensity-bounded witness.",
        ),
        "trusted bound [%]": st.column_config.NumberColumn(
            format="%.2f",
            help=r"Trusted-calibration bound on $P_{\rm guess}$.",
        ),
        "intensity-bound [%]": st.column_config.NumberColumn(
            format="%.2f",
            help=r"Intensity-bounded bound on $P_{\rm guess}$.",
        ),
        "P_guess [%]": st.column_config.NumberColumn(
            format="%.2f",
            help=r"Observed guessing probability $P_{\rm guess}$.",
        ),
        "trusted eta_* [%]": st.column_config.NumberColumn(
            format="%.2f",
            help=r"Certified efficiency $\eta_\ast$ from the trusted-calibration benchmark.",
        ),
        "intensity eta_* [%]": st.column_config.NumberColumn(
            format="%.2f",
            help=r"Certified efficiency $\eta_\ast$ from the intensity-bounded benchmark.",
        ),
    }


def metric_value(value: object, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def best_efficiency(frame: pd.DataFrame) -> float | None:
    values = pd.concat(
        [
            frame["trusted eta_* [%]"],
            frame["intensity eta_* [%]"],
        ],
        ignore_index=True,
    ).dropna()
    if values.empty:
        return None
    return float(values.max())


def render_metric_row(frame: pd.DataFrame) -> None:
    case_count, trusted_res, intensity_res, eta = st.columns(4)
    case_count.metric("Cases", f"{len(frame)}")
    trusted_res.metric("Max trusted resolution", metric_value(frame["trusted certified resolution"].max()))
    intensity_res.metric("Max intensity resolution", metric_value(frame["intensity certified resolution"].max()))
    eta.metric(r"Best $\eta_\ast$", metric_value(best_efficiency(frame), "%"))


def grouped_bar_data(frame: pd.DataFrame, value_columns: dict[str, str]) -> pd.DataFrame:
    data = frame[["m", *value_columns.keys()]].rename(columns=value_columns)
    return data.melt(id_vars="m", var_name="quantity", value_name="value").dropna()


def render_grouped_bar_chart(
    data: pd.DataFrame,
    *,
    y_title: str,
    y_domain: list[float] | None = None,
) -> None:
    y_encoding: dict[str, object] = {
        "field": "value",
        "type": "quantitative",
        "title": y_title,
        "stack": None,
    }
    if y_domain is not None:
        y_encoding["scale"] = {"domain": y_domain}

    st.vega_lite_chart(
        data,
        {
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {
                    "field": "m",
                    "type": "ordinal",
                    "title": "subspace cutoff m",
                    "sort": [int(value) for value in sorted(data["m"].unique().tolist())],
                },
                "xOffset": {"field": "quantity", "type": "nominal"},
                "y": y_encoding,
                "color": {
                    "field": "quantity",
                    "type": "nominal",
                    "title": None,
                    "scale": {
                        "range": ["#2E6F9E", "#D97706", "#4B5563"],
                    },
                },
                "tooltip": [
                    {"field": "m", "type": "ordinal", "title": "m"},
                    {"field": "quantity", "type": "nominal", "title": "quantity"},
                    {"field": "value", "type": "quantitative", "title": y_title, "format": ".2f"},
                ],
            },
        },
        width="stretch",
    )


def render_bar_plots(frame: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader(r"Guessing probability $P_{\rm guess}$")
        render_grouped_bar_chart(
            grouped_bar_data(
                frame,
                {
                    "P_guess [%]": "P_guess",
                    "trusted bound [%]": "trusted bound",
                    "intensity-bound [%]": "intensity bound",
                },
            ),
            y_title="probability [%]",
            y_domain=[0, 100],
        )
    with right:
        st.subheader(r"Certified efficiency $\eta_\ast$")
        render_grouped_bar_chart(
            grouped_bar_data(
                frame,
                {
                    "trusted eta_* [%]": "trusted eta_*",
                    "intensity eta_* [%]": "intensity eta_*",
                },
            ),
            y_title="eta_* [%]",
            y_domain=[0, 100],
        )


def render_dashboard(frame: pd.DataFrame) -> None:
    render_metric_row(frame)

    render_bar_plots(frame)

    st.subheader("Certification table")
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config=result_column_config(),
    )


def latex_results_table(frame: pd.DataFrame) -> str:
    columns = [
        ("m", "$m$", "{:d}"),
        ("N", "$N$", "{:d}"),
        ("R", "$R$", "{:d}"),
        ("I used", "$I$", "{:.4f}"),
        ("trusted bound [%]", "Trusted", "{:.2f}\\%"),
        ("intensity-bound [%]", "Intensity", "{:.2f}\\%"),
        ("P_guess [%]", "$P_{\\rm guess}$", "{:.2f}\\%"),
        ("trusted eta_* [%]", "$\\eta_*^{\\rm tr}$", "{:.2f}\\%"),
        ("intensity eta_* [%]", "$\\eta_*^{I}$", "{:.2f}\\%"),
    ]
    lines = [
        "\\begin{tabular}{ccccccccc}",
        "\\hline",
        " & ".join(label for _, label, _ in columns) + " \\\\",
        "\\hline",
    ]
    for _, row in frame.iterrows():
        values = []
        for column, _, template in columns:
            value = row[column]
            if pd.isna(value):
                values.append("-")
            elif template == "{:d}":
                values.append(template.format(int(value)))
            else:
                values.append(template.format(float(value)))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def initialize_session_state() -> None:
    if "result_frame" not in st.session_state:
        rows = R.build_manuscript_certification_table()
        st.session_state["result_frame"] = rows_to_frame(
            rows,
            R.MANUSCRIPT_CERTIFICATION_CASES,
            D.INTENSITIES,
        )


def case_records(cases: tuple[R.CertificationConfig, ...]) -> tuple[tuple[int, int, int, float | None], ...]:
    return tuple(
        (case.num_inputs, case.max_photons, case.resolution, case.intensity_cap)
        for case in cases
    )


@st.cache_data(show_spinner="Computing witnesses...")
def build_certification_rows_cached(
    mus: tuple[float, ...],
    records: tuple[tuple[int, int, int, float | None], ...],
    probabilities: tuple[tuple[float, ...], ...],
    observed_guess_items: tuple[tuple[int, float], ...],
) -> list[R.CertificationRow]:
    cases = tuple(
        R.CertificationConfig(
            num_inputs=num_inputs,
            max_photons=max_photons,
            resolution=resolution,
            intensity_cap=intensity_cap,
        )
        for num_inputs, max_photons, resolution, intensity_cap in records
    )
    return R.build_certification_table(
        mus=mus,
        cases=cases,
        measurement_probabilities=probabilities,
        observed_guesses=dict(observed_guess_items),
    )


def compute_results(
    intensities_frame: pd.DataFrame,
    probabilities_frame: pd.DataFrame,
    cases_frame: pd.DataFrame,
    use_manuscript_values: bool,
) -> pd.DataFrame | None:
    try:
        mus = parse_intensities(intensities_frame)
        probabilities = parse_probabilities(probabilities_frame)
        cases = parse_cases(cases_frame)
    except Exception as exc:
        st.error(f"Could not parse inputs: {exc}")
        return None

    if not cases:
        st.error("At least one certification case is required.")
        return None
    if not validate_inputs(mus, probabilities, cases):
        return None

    use_exact_manuscript_values = should_use_exact_manuscript_values(
        mus,
        probabilities,
        cases,
        use_manuscript_values,
    )
    if use_manuscript_values and not use_exact_manuscript_values:
        st.info(
            "Custom data or cases detected; computing P_guess directly from the probability table."
        )
    guesses = observed_guess_map(probabilities, cases, use_exact_manuscript_values)
    try:
        rows = build_certification_rows_cached(
            mus,
            case_records(cases),
            probabilities,
            tuple(sorted(guesses.items())),
        )
    except Exception as exc:
        st.error(f"Could not compute witnesses: {exc}")
        return None
    return rows_to_frame(rows, cases, mus)


def main() -> None:
    st.set_page_config(page_title="PNR Resolution Certification", layout="wide")
    initialize_session_state()
    st.title("PNR Resolution Certification")

    with st.sidebar:
        with st.expander("How to use and cite", expanded=True):
            st.markdown(
                r"""
Enter input intensities $\mu_i$ and click probabilities $p(b|\mu_i)$ in Data,
choose cases $(N,m,R,I)$ in Cases, then click Compute witnesses. The app reports
$P_{\rm guess}$, trusted and intensity-bounded bounds, certified resolution, and
certified $\eta_\ast$.

Method: Certification of the genuine resolution of photon number resolving detectors.
If you use this tool, please cite that paper.
"""
            )

        st.header("Options")
        use_manuscript_values = st.checkbox(
            r"Use exact bundled manuscript $P_{\rm guess}$ values",
            value=True,
            help=r"For the bundled example data this reproduces the manuscript table. "
            r"For edited or uploaded data, the app always computes $P_{\rm guess}$ "
            r"directly from the probability table.",
        )

    results_tab, data_tab, cases_tab, export_tab = st.tabs(["Results", "Data", "Cases", "Export"])
    uploads_ok = True

    with data_tab:
        left, right = st.columns([1, 2])
        with left:
            st.subheader(r"Input intensities $\mu_i$")
            intensity_upload = st.file_uploader("Upload intensity CSV", type="csv", key="intensity_csv")
            intensities_frame, intensity_upload_ok = read_csv_upload(
                intensity_upload,
                default_intensities_frame(),
                "intensity",
            )
            uploads_ok = uploads_ok and intensity_upload_ok
            intensities_frame = st.data_editor(
                intensities_frame,
                num_rows="dynamic",
                width="stretch",
                key="intensities_table",
            )
        with right:
            st.subheader(r"Click probabilities $p(b|\mu_i)$")
            probability_upload = st.file_uploader("Upload probability CSV", type="csv", key="probability_csv")
            probability_column_count = st.number_input(
                "Number of probability columns",
                min_value=1,
                max_value=MAX_PROBABILITY_COLUMNS,
                value=DEFAULT_PROBABILITY_COLUMN_COUNT,
                step=1,
                disabled=probability_upload is not None,
            )
            probabilities_frame, probability_upload_ok = read_csv_upload(
                probability_upload,
                default_probabilities_frame(int(probability_column_count)),
                "probability",
            )
            uploads_ok = uploads_ok and probability_upload_ok
            probabilities_frame = st.data_editor(
                probabilities_frame,
                num_rows="dynamic",
                width="stretch",
                key="probabilities_table",
            )

    with cases_tab:
        st.info(
            r"The bundled manuscript example uses a 3% safety cap by default; "
            r"edit $I$ to use another calibration bound. For intensity-bounded rows, leaving $I$ blank makes the app use "
            r"$\max_i \mu_i$ over the selected inputs. To be conservative under "
            r"calibration uncertainty, enter an upper calibration bound for $I$. "
            r"If $I>m+1$, the app uses the generalized conservative tail correction."
        )
        with st.expander(r"Advanced certification cases $(N,m,R,I)$", expanded=False):
            cases_upload = st.file_uploader("Upload cases CSV", type="csv", key="cases_csv")
            cases_frame, cases_upload_ok = read_csv_upload(
                cases_upload,
                default_cases_frame(),
                "cases",
            )
            uploads_ok = uploads_ok and cases_upload_ok
            cases_frame = st.data_editor(
                cases_frame,
                num_rows="dynamic",
                width="stretch",
                key="cases_table",
                column_config={
                    "I": st.column_config.NumberColumn(
                        "intensity bound I",
                        help=r"Blank means max selected input intensity; otherwise enter the certified bound $I$.",
                    ),
                },
            )

    with st.sidebar:
        compute_requested = st.button("Compute witnesses", type="primary", width="stretch")

    with results_tab:
        if compute_requested:
            if not uploads_ok:
                st.error("Fix or remove invalid uploaded CSV files before computing.")
            else:
                result = compute_results(
                    intensities_frame=intensities_frame,
                    probabilities_frame=probabilities_frame,
                    cases_frame=cases_frame,
                    use_manuscript_values=use_manuscript_values,
                )
                if result is not None:
                    st.session_state["result_frame"] = result
                    st.success("Computation complete.")

        result_frame = st.session_state["result_frame"]
        if result_frame is None:
            st.info("No computed results.")
        else:
            render_dashboard(result_frame)

    with export_tab:
        result_frame = st.session_state["result_frame"]
        if result_frame is None:
            st.info("No computed results.")
        else:
            latex_table = latex_results_table(result_frame)
            csv_data = result_frame.to_csv(index=False).encode("utf-8")
            col_csv, col_tex = st.columns(2)
            with col_csv:
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="pnr_certification_results.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with col_tex:
                st.download_button(
                    "Download LaTeX",
                    data=latex_table,
                    file_name="pnr_certification_results.tex",
                    mime="text/plain",
                    width="stretch",
                )
            st.subheader("LaTeX table")
            st.code(latex_table, language="latex")


if __name__ == "__main__":
    main()
