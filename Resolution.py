r"""Resolution workflow for measurement-outcome certification.

This module consolidates the numerical tasks required for the paper:

1. Trusted calibration witnesses (Methods: "Derivation of resolution witness
   bounds"):
   Enumerate the optimal subsets of the experimentally calibrated coherent-state
   intensities and compute the witness values
   \(W_{R,m,\{\mu_i\}} = \frac{1}{N}\sum_{n=0}^m \max_{|S|=R} q(n|\mu_S)\)
   with optional Poisson tails.
2. Intensity-bounded witnesses (Methods: "Numerical optimization of witness
   bounds"):
   Reproduce both the heuristic SLSQP optimisation over monotone intensity
   schedules and the globally optimal discrete partition search with cutoff
   ``n_max``.
3. Intrinsic resolution of efficiency-limited photon-number measurements
   (Methods: "Resolution of the efficiency-limited detector"):
   Build and solve the sparse linear programs that decide whether an
   efficiency-limited detector is simulable with ``R`` outcomes and tabulate
   the threshold efficiencies in Table 1 of the manuscript.
4. Certified efficiencies:
   Given an observed guessing probability, solve for the efficiency-limited
   PNR benchmark that matches the score in either the trusted or
   intensity-bounded source model.

Only ``numpy`` and ``scipy`` are required beyond the Python standard library.
Docstrings and comments refer to the section titles used in the manuscript.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb, exp, floor, lgamma, log
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix
from scipy.special import gammaincc, gammaln

import experimental_data

# ---------------------------------------------------------------------------
# Dataclasses


@dataclass
class TrustedWitnessResult:
    """Trusted calibration output for a specific subset size R."""

    subset_indices: Tuple[int, ...]
    subset_mus: Tuple[float, ...]
    weight: float
    probability: float


@dataclass
class HeuristicCalibrationResult:
    """Result of the SLSQP optimisation with an intensity upper bound."""

    increments: np.ndarray
    intensities: np.ndarray
    probability: float


@dataclass
class DiscreteCalibrationResult:
    """Globally optimal discrete partition result (fixed ``n_max``)."""

    cuts: Tuple[int, ...]
    intensities: np.ndarray
    probability: float


@dataclass
class PartitionLPSystem:
    """Sparse linear program capturing coarse-grained POVM feasibility."""

    matrix: csr_matrix
    b_template: np.ndarray
    eq3_offset: int
    eq3_size: int
    num_vars: int
    levels: int
    partitions: Tuple[Tuple[int, ...], ...]
    coarse_outputs: int


@dataclass(frozen=True)
class CertificationConfig:
    """Configuration triple used to reproduce Table ``tab:certification``."""

    num_inputs: int
    max_photons: int
    resolution: int
    intensity_cap: float | None = None


@dataclass
class CertificationRow:
    """Row containing trusted/untrusted witnesses and optional experimental data."""

    num_inputs: int
    max_photons: int
    resolution: int
    trusted_bound: float
    untrusted_bound: float
    untrusted_full_space: float
    experimental_guess: float | None = None
    trusted_efficiency: float | None = None
    untrusted_efficiency: float | None = None

    def certified_resolution_trusted(self) -> int | None:
        """Return certified resolution (trusted scenario) if data is available."""

        if self.experimental_guess is None:
            return None
        return self.resolution + 1 if self.experimental_guess > self.trusted_bound else self.resolution

    def certified_resolution_intensity(self) -> int | None:
        """Return certified resolution for the intensity-bounded scenario."""

        if self.experimental_guess is None:
            return None
        return self.resolution + 1 if self.experimental_guess > self.untrusted_bound else self.resolution


# ---------------------------------------------------------------------------
# Manuscript-data defaults


def certification_configs_from_records(
    records: Sequence[Tuple[int, int, int, float | None]],
) -> Tuple[CertificationConfig, ...]:
    """Convert data-file case records to ``CertificationConfig`` instances."""

    return tuple(
        CertificationConfig(
            num_inputs=num_inputs,
            max_photons=max_photons,
            resolution=resolution,
            intensity_cap=intensity_cap,
        )
        for num_inputs, max_photons, resolution, intensity_cap in records
    )


MANUSCRIPT_CERTIFICATION_CASES = certification_configs_from_records(experimental_data.CERTIFICATION_CASES)

# ---------------------------------------------------------------------------
# Shared Poisson helpers


def poisson_pmf(n: int, mu: float) -> float:
    """Return ``q(n|mu)`` with logarithmic stability."""

    if mu < 0:
        raise ValueError("mu must be non-negative")
    if n < 0:
        return 0.0
    if mu == 0:
        return 1.0 if n == 0 else 0.0
    return exp(-mu + n * log(mu) - lgamma(n + 1))


def poisson_tail(n_min: int, mu: float, tol: float = 1e-15, max_terms: int = 10_000) -> float:
    """Return ``sum_{n=n_min}^∞ q(n|mu)`` via a truncated series."""

    if n_min < 0:
        raise ValueError("n_min must be non-negative")
    term = poisson_pmf(n_min, mu)
    total = term
    n = n_min
    for _ in range(max_terms):
        if term < tol:
            break
        n += 1
        term *= mu / n
        total += term
    return total


def worst_case_poisson_tail(n_min: int, mu_cap: float) -> float:
    r"""Return ``sum_{n=n_min}^∞ max_{0 <= mu <= mu_cap} q(n|mu)``.

    For fixed ``n``, the Poisson mass is maximized at ``mu=n``. If the allowed
    intensity interval is capped at ``mu_cap``, the maximizing mean is therefore
    ``min(n, mu_cap)``. When ``mu_cap <= n_min``, this reduces to the ordinary
    Poisson tail at ``mu_cap``.
    """

    if n_min < 0:
        raise ValueError("n_min must be non-negative")
    if mu_cap < 0:
        raise ValueError("mu_cap must be non-negative")
    upper = int(floor(mu_cap))
    if upper < n_min:
        return poisson_tail(n_min, mu_cap)
    finite = sum(poisson_pmf(n, float(n)) for n in range(n_min, upper + 1))
    return finite + poisson_tail(upper + 1, mu_cap)


def pmf_table(mus: Sequence[float], n_max: int, loss: float = 1.0) -> np.ndarray:
    """Return a ``len(mus) x (n_max+1)`` table with ``q(n|loss*mu_i)``."""

    table = np.zeros((len(mus), n_max + 1), dtype=float)
    for i, mu in enumerate(mus):
        mu_eff = max(0.0, loss * mu)
        for n in range(n_max + 1):
            table[i, n] = poisson_pmf(n, mu_eff)
    return table


def total_guess_prob_from_mus(
    mus: Sequence[float],
    n_max: int,
    loss: float = 1.0,
    denominator: int | None = None,
) -> float:
    """Return ``(1/N) * sum_n max_i q(n|loss*mu_i)`` for the provided means."""

    pmf = pmf_table(mus, n_max, loss=loss)
    denom = denominator if denominator is not None else len(mus)
    return float(np.sum(np.max(pmf, axis=0)) / denom)


# ---------------------------------------------------------------------------
# Trusted calibration (Methods: Derivation of resolution witness bounds)


def trusted_guessing_weight(
    mu_vec: Sequence[float],
    photon_cutoff: int,
    include_vacuum: bool = True,
    tail_mu: float | None = None,
) -> float:
    """Unnormalised witness ``W = sum_{n=0}^m max_x q(n|mu_x)`` with Poisson tail."""

    if not mu_vec:
        raise ValueError("mu_vec cannot be empty")
    tail_mu = tail_mu if tail_mu is not None else mu_vec[-1]
    weight = 1.0 if include_vacuum else 0.0
    for n in range(1, photon_cutoff + 1):
        weight += max(poisson_pmf(n, mu) for mu in mu_vec)
    if tail_mu is not None:
        weight += poisson_tail(photon_cutoff + 1, tail_mu)
    return weight


def trusted_guessing_probability(
    mu_vec: Sequence[float],
    photon_cutoff: int,
    denominator: int | None = None,
    tail_mu: float | None = None,
) -> float:
    """Return ``W/N`` for a fixed subset of coherent-state intensities."""

    denom = denominator if denominator is not None else len(mu_vec)
    if denom <= 0:
        raise ValueError("denominator must be positive")
    weight = trusted_guessing_weight(mu_vec, photon_cutoff, True, tail_mu)
    return weight / denom


def _subset_indices(
    total: int,
    subset_size: int,
    required: Iterable[int] | None = None,
) -> Iterator[Tuple[int, ...]]:
    """Yield sorted index tuples, forcing ``required`` indices to appear."""

    if subset_size > total:
        raise ValueError("subset_size cannot exceed available intensities")
    required_set = set(required or ())
    if len(required_set) > subset_size:
        raise ValueError("Too many required indices for subset size")
    flex_indices = [i for i in range(total) if i not in required_set]
    flex_needed = subset_size - len(required_set)
    for combo in itertools.combinations(flex_indices, flex_needed):
        yield tuple(sorted((*required_set, *combo)))


def maximise_trusted_witness(
    mus: Sequence[float],
    subset_size: int,
    photon_cutoff: int,
    denominator: int | None = None,
    tail_mu: float | None = None,
    required_indices: Iterable[int] | None = None,
) -> TrustedWitnessResult:
    """Return the optimal subset of size ``subset_size`` for the trusted witness."""

    best: TrustedWitnessResult | None = None
    for indices in _subset_indices(len(mus), subset_size, required_indices):
        subset = tuple(mus[i] for i in indices)
        prob = trusted_guessing_probability(
            subset,
            photon_cutoff=photon_cutoff,
            denominator=denominator if denominator is not None else len(mus),
            tail_mu=tail_mu if tail_mu is not None else subset[-1],
        )
        weight = prob * (denominator if denominator is not None else len(mus))
        candidate = TrustedWitnessResult(indices, subset, weight, prob)
        if best is None or prob > best.probability:
            best = candidate
    if best is None:
        raise RuntimeError("No subset evaluated; check subset_size and required_indices")
    return best


def trusted_witness_scan(
    mus: Sequence[float],
    subset_sizes: Iterable[int],
    photon_cutoff: int,
    denominator: int | None = None,
    tail_mu: float | None = None,
    required_indices: Iterable[int] | None = None,
) -> List[TrustedWitnessResult]:
    """Compute trusted witnesses for every ``subset_size`` in ``subset_sizes``."""

    results: List[TrustedWitnessResult] = []
    for size in subset_sizes:
        if size <= 0:
            raise ValueError("subset sizes must be positive")
        results.append(
            maximise_trusted_witness(
                mus,
                subset_size=size,
                photon_cutoff=photon_cutoff,
                denominator=denominator,
                tail_mu=tail_mu,
                required_indices=required_indices,
            )
        )
    return results


def solve_efficiency_for_target(
    mus: Sequence[float],
    observed_probability: float,
    photon_cutoff: int,
    denominator: int,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> float:
    """Binary search the detection efficiency matching the observed witness."""

    low, high = 0.0, 1.0
    def scaled_prob(eta: float) -> float:
        return trusted_guessing_probability(
            tuple(eta * mu for mu in mus),
            photon_cutoff=photon_cutoff,
            denominator=denominator,
            tail_mu=None,
        )

    p_low, p_high = scaled_prob(low), scaled_prob(high)
    if not (p_low <= observed_probability <= p_high):
        raise ValueError("Target probability outside the achievable range")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid = scaled_prob(mid)
        if abs(p_mid - observed_probability) < tol:
            return mid
        if p_mid < observed_probability:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Certified efficiency benchmarks


def _validate_efficiency(efficiency: float) -> None:
    if not 0.0 <= efficiency <= 1.0:
        raise ValueError("efficiency must lie in [0, 1]")


def _max_poisson_tail(
    mus: Sequence[float],
    n_min: int,
    tol: float = 1e-15,
    max_terms: int = 100_000,
) -> float:
    """Return ``sum_{n=n_min}^∞ max_i q(n|mu_i)`` by a controlled tail sum."""

    if not mus:
        raise ValueError("mus cannot be empty")
    if n_min < 0:
        raise ValueError("n_min must be non-negative")
    max_mu = max(mus)
    total = 0.0
    n = n_min
    for _ in range(max_terms):
        term = max(poisson_pmf(n, mu) for mu in mus)
        total += term
        if n > max_mu and term < tol:
            return total
        n += 1
    raise RuntimeError("Poisson max-tail sum did not converge")


def _efficiency_limited_click_probability(mu: float, max_photons: int, efficiency: float, clicks: int) -> float:
    """Probability of ``clicks`` for the efficiency-limited PNR benchmark inside ``n <= m``."""

    if clicks < 0 or clicks > max_photons:
        return 0.0
    return sum(
        poisson_pmf(n, mu)
        * comb(n, clicks)
        * efficiency**clicks
        * (1.0 - efficiency) ** (n - clicks)
        for n in range(clicks, max_photons + 1)
    )


def trusted_subspace_efficiency_benchmark(
    mus: Sequence[float],
    max_photons: int,
    efficiency: float,
    denominator: int | None = None,
) -> float:
    r"""Return the trusted-source efficiency-limited benchmark ``F_m,N,{mu}(eta)``.

    The coherent-state intensities are fixed and known. Photon numbers above
    ``max_photons`` are treated by the same conservative outside-subspace term
    used in the subspace witness.
    """

    if not mus:
        raise ValueError("mus cannot be empty")
    if max_photons < 0:
        raise ValueError("max_photons must be non-negative")
    _validate_efficiency(efficiency)
    denom = denominator if denominator is not None else len(mus)
    if denom <= 0:
        raise ValueError("denominator must be positive")

    inside = 0.0
    for clicks in range(max_photons + 1):
        inside += max(
            _efficiency_limited_click_probability(mu, max_photons, efficiency, clicks)
            for mu in mus
        )
    tail = _max_poisson_tail(mus, max_photons + 1)
    return float((inside + tail) / denom)


def _solve_efficiency_bisection(
    benchmark,
    observed_probability: float,
    tol: float,
    max_iter: int,
) -> float:
    """Solve ``benchmark(eta) = observed_probability`` on ``eta in [0, 1]``."""

    if not 0.0 <= observed_probability <= 1.0:
        raise ValueError("observed_probability must lie in [0, 1]")
    low, high = 0.0, 1.0
    p_low, p_high = benchmark(low), benchmark(high)
    if observed_probability < p_low - tol or observed_probability > p_high + tol:
        raise ValueError(
            "Target probability outside the achievable range "
            f"[{p_low:.12g}, {p_high:.12g}]"
        )
    if observed_probability <= p_low + tol:
        return low
    if observed_probability >= p_high - tol:
        return high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid = benchmark(mid)
        if abs(p_mid - observed_probability) <= tol:
            return mid
        if p_mid < observed_probability:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def solve_trusted_subspace_efficiency(
    mus: Sequence[float],
    observed_probability: float,
    max_photons: int,
    denominator: int | None = None,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Return the certified efficiency for fixed calibrated intensities."""

    return _solve_efficiency_bisection(
        lambda eta: trusted_subspace_efficiency_benchmark(
            mus=mus,
            max_photons=max_photons,
            efficiency=eta,
            denominator=denominator,
        ),
        observed_probability=observed_probability,
        tol=tol,
        max_iter=max_iter,
    )


def _efficiency_block_polynomial(mask: int, max_photons: int, efficiency: float) -> np.ndarray:
    """Return ``P`` such that ``sum_{b in mask} p_eta(b|mu) = exp(-mu) P(mu)``."""

    coeffs = np.zeros(max_photons + 1, dtype=float)
    for n in range(max_photons + 1):
        total = 0.0
        for clicks in range(n + 1):
            if mask & (1 << clicks):
                total += comb(n, clicks) * efficiency**clicks * (1.0 - efficiency) ** (n - clicks)
        coeffs[n] = total / exp(lgamma(n + 1))
    return coeffs


def _exp_polynomial_value(mu: float, coeffs: np.ndarray) -> float:
    return float(exp(-mu) * poly.polyval(mu, coeffs))


def _max_exp_polynomial_on_interval(coeffs: np.ndarray, intensity_cap: float) -> float:
    """Maximize ``exp(-mu) P(mu)`` on ``0 <= mu <= intensity_cap``."""

    derivative = poly.polyder(coeffs)
    stationary = np.zeros_like(coeffs)
    stationary[: len(derivative)] += derivative
    stationary[: len(coeffs)] -= coeffs

    candidates = [0.0, float(intensity_cap)]
    trimmed = poly.polytrim(stationary, tol=1e-14)
    if len(trimmed) > 1:
        for root in poly.polyroots(trimmed):
            if abs(root.imag) <= 1e-8:
                mu = float(root.real)
                if -1e-10 <= mu <= intensity_cap + 1e-10:
                    candidates.append(min(max(mu, 0.0), float(intensity_cap)))
    return max(_exp_polynomial_value(mu, coeffs) for mu in candidates)


def _untrusted_efficiency_inside_value(
    num_inputs: int,
    max_photons: int,
    intensity_cap: float,
    efficiency: float,
) -> float:
    """Optimize the inside-subspace benchmark over unknown bounded intensities."""

    num_outputs = max_photons + 1
    full_mask = (1 << num_outputs) - 1
    block_values = np.zeros(full_mask + 1, dtype=float)
    for mask in range(1, full_mask + 1):
        coeffs = _efficiency_block_polynomial(mask, max_photons, efficiency)
        block_values[mask] = _max_exp_polynomial_on_interval(coeffs, intensity_cap)

    memo: Dict[Tuple[int, int], float] = {}

    def best_partition(mask: int, groups_left: int) -> float:
        key = (mask, groups_left)
        if key in memo:
            return memo[key]
        if mask == 0:
            value = 0.0
        elif groups_left == 0:
            value = -float("inf")
        else:
            first_bit = mask & -mask
            value = -float("inf")
            submask = mask
            while submask:
                if submask & first_bit:
                    value = max(
                        value,
                        block_values[submask] + best_partition(mask ^ submask, groups_left - 1),
                    )
                submask = (submask - 1) & mask
        memo[key] = value
        return value

    return best_partition(full_mask, min(num_inputs, num_outputs))


def untrusted_subspace_efficiency_benchmark(
    num_inputs: int,
    max_photons: int,
    intensity_cap: float,
    efficiency: float,
) -> float:
    r"""Return the intensity-bounded efficiency-limited benchmark ``F_m,N,I(eta)``.

    The out-of-subspace contribution uses the generalized worst-case tail, so
    intensity caps larger than ``m+1`` remain valid but can give looser bounds.
    """

    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    if max_photons < 0:
        raise ValueError("max_photons must be non-negative")
    if intensity_cap <= 0:
        raise ValueError("intensity_cap must be positive")
    _validate_efficiency(efficiency)
    inside = _untrusted_efficiency_inside_value(num_inputs, max_photons, intensity_cap, efficiency)
    tail = worst_case_poisson_tail(max_photons + 1, intensity_cap)
    return float((inside + tail) / num_inputs)


def solve_untrusted_subspace_efficiency(
    num_inputs: int,
    max_photons: int,
    intensity_cap: float,
    observed_probability: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Return the certified efficiency when only an intensity bound is trusted."""

    return _solve_efficiency_bisection(
        lambda eta: untrusted_subspace_efficiency_benchmark(
            num_inputs=num_inputs,
            max_photons=max_photons,
            intensity_cap=intensity_cap,
            efficiency=eta,
        ),
        observed_probability=observed_probability,
        tol=tol,
        max_iter=max_iter,
    )


# ---------------------------------------------------------------------------
# Intensity-bounded calibration (Methods: Numerical optimization of witness bounds)


VALID_METHODS = {
    "discrete",
    "heuristic",
    "heuristic_seeded",
    "heuristic_multistart",
}


def random_increment_seed(num_states: int, intensity_cap: float, rng: np.random.Generator) -> np.ndarray:
    """Draw non-negative increments that sum to ``intensity_cap`` via Dirichlet sampling."""

    weights = rng.dirichlet(np.ones(num_states))
    return weights * intensity_cap


def optimize_poisson_spacing(
    num_states: int,
    intensity_cap: float,
    loss: float,
    photon_cutoff: int,
    initial_increments: Sequence[float] | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-10,
) -> HeuristicCalibrationResult:
    """SLSQP optimiser over monotone intensities (Method A)."""

    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if intensity_cap <= 0:
        raise ValueError("intensity_cap must be positive")

    if initial_increments is None:
        initial_increments = np.full(num_states, intensity_cap / num_states)

    bounds = [(0.0, None)] * num_states
    constraints = [{"type": "ineq", "fun": lambda d: intensity_cap - float(np.sum(d))}]

    def negative_objective(delta: np.ndarray) -> float:
        mus = np.cumsum(np.clip(delta, 0.0, None))
        return -total_guess_prob_from_mus(mus, n_max=photon_cutoff, loss=loss, denominator=num_states)

    result = minimize(
        fun=negative_objective,
        x0=np.asarray(initial_increments, dtype=float),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"SLSQP optimisation failed: {result.message}")
    increments = np.clip(result.x, 0.0, None)
    if not np.all(np.isfinite(increments)):
        raise RuntimeError("SLSQP optimisation returned non-finite increments")
    if float(np.sum(increments)) > intensity_cap + 1e-8:
        raise RuntimeError("SLSQP optimisation violated the intensity bound")
    intensities = np.cumsum(increments)
    if not np.all(np.isfinite(intensities)):
        raise RuntimeError("SLSQP optimisation returned non-finite intensities")
    probability = total_guess_prob_from_mus(intensities, n_max=photon_cutoff, loss=loss, denominator=num_states)
    if not np.isfinite(probability):
        raise RuntimeError("SLSQP optimisation returned a non-finite objective value")
    return HeuristicCalibrationResult(increments=increments, intensities=intensities, probability=probability)


def optimize_poisson_spacing_multistart(
    num_states: int,
    intensity_cap: float,
    loss: float,
    photon_cutoff: int,
    num_random: int = 8,
    include_discrete_seed: bool = True,
    rng_seed: int | None = None,
) -> Tuple[HeuristicCalibrationResult, List[HeuristicCalibrationResult]]:
    """Run multiple heuristic seeds and return the best result plus all attempts."""

    rng = np.random.default_rng(rng_seed)
    seeds: List[np.ndarray] = [np.full(num_states, intensity_cap / num_states)]
    for _ in range(max(0, num_random)):
        seeds.append(random_increment_seed(num_states, intensity_cap, rng))

    discrete_seed: DiscreteCalibrationResult | None = None
    if include_discrete_seed:
        try:
            discrete_seed = discrete_optimize(num_states, intensity_cap, loss, photon_cutoff)
            increments = np.diff(np.concatenate(([0.0], discrete_seed.intensities)))
            seeds.append(increments)
        except Exception:
            pass  # Discrete search might fail for large parameters; skip silently.

    attempts: List[HeuristicCalibrationResult] = []
    best: HeuristicCalibrationResult | None = None
    for seed in seeds:
        res = optimize_poisson_spacing(num_states, intensity_cap, loss, photon_cutoff, seed)
        attempts.append(res)
        if best is None or res.probability > best.probability:
            best = res
    assert best is not None
    return best, attempts


def discrete_optimize(
    num_states: int,
    intensity_cap: float,
    loss: float,
    photon_cutoff: int,
    allow_empty: bool = False,
) -> DiscreteCalibrationResult:
    """Enumerate cut partitions (Method B) to obtain the global optimum."""

    if num_states < 1:
        raise ValueError("num_states must be positive")
    if photon_cutoff < 1:
        raise ValueError("photon_cutoff must be at least 1 for discrete enumeration")
    if num_states == 1:
        intensities = np.array([0.0], dtype=float)
        probability = total_guess_prob_from_mus(intensities, n_max=photon_cutoff, loss=loss, denominator=1)
        return DiscreteCalibrationResult(cuts=tuple(), intensities=intensities, probability=probability)

    best = DiscreteCalibrationResult(cuts=tuple(), intensities=np.zeros(num_states), probability=-1.0)
    iterator = (
        itertools.combinations(range(0, photon_cutoff), num_states - 1)
        if not allow_empty
        else itertools.combinations_with_replacement(range(0, photon_cutoff), num_states - 1)
    )
    for cuts in iterator:
        cuts = tuple(cuts)
        intensities: List[float] = [0.0]
        valid = True
        for idx in range(1, num_states - 1):
            prev = cuts[idx - 1]
            curr = cuts[idx]
            if curr == prev and not allow_empty:
                valid = False
                break
            mu_opt = interior_intensity(prev, curr)
            if mu_opt > intensity_cap:
                valid = False
                break
            intensities.append(mu_opt)
        if not valid:
            continue
        last_mu = interior_intensity(cuts[-1], photon_cutoff)
        intensities.append(min(intensity_cap, last_mu))
        probability = total_guess_prob_from_mus(
            intensities,
            n_max=photon_cutoff,
            loss=loss,
            denominator=num_states,
        )
        if probability > best.probability:
            best = DiscreteCalibrationResult(cuts=cuts, intensities=np.asarray(intensities, dtype=float), probability=probability)

    if best.probability < 0:
        raise RuntimeError("No discrete partition satisfied the constraints. Increase photon_cutoff or allow empties.")
    return best


def interior_intensity(prev: int, curr: int) -> float:
    """Closed-form maximiser satisfying ``q(prev|mu) = q(curr|mu)`` for an interval ``(prev, curr]``."""

    if curr <= prev:
        raise ValueError("curr must exceed prev for an interior interval")
    return exp((gammaln(curr + 1) - gammaln(prev + 1)) / (curr - prev))


def optimal_mus_for_method(
    subset_size: int,
    intensity_cap: float,
    loss: float,
    photon_cutoff: int,
    method: str,
    multistart_random: int = 8,
    include_discrete_seed: bool = False,
) -> np.ndarray:
    """Return the optimal intensities for the requested optimisation method."""

    method_key = method.lower()
    if method_key not in VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(VALID_METHODS)}")
    if method_key == "discrete":
        return discrete_optimize(subset_size, intensity_cap, loss, photon_cutoff).intensities
    if method_key == "heuristic":
        return optimize_poisson_spacing(subset_size, intensity_cap, loss, photon_cutoff).intensities
    if method_key == "heuristic_seeded":
        discrete_solution = discrete_optimize(subset_size, intensity_cap, loss, photon_cutoff)
        increments = np.diff(np.concatenate(([0.0], discrete_solution.intensities)))
        return optimize_poisson_spacing(subset_size, intensity_cap, loss, photon_cutoff, increments).intensities
    best, _ = optimize_poisson_spacing_multistart(
        subset_size,
        intensity_cap,
        loss,
        photon_cutoff,
        num_random=multistart_random,
        include_discrete_seed=include_discrete_seed,
    )
    return best.intensities


def untrusted_witness_table(
    num_states: int,
    intensity_cap: float,
    loss: float,
    photon_cutoff: int,
    method: str = "discrete",
) -> List[float]:
    """Return ``W_{R}(I, n_max)`` for ``R = 1..num_states`` under the chosen method."""

    values: List[float] = []
    for subset_size in range(1, num_states + 1):
        mus = optimal_mus_for_method(subset_size, intensity_cap, loss, photon_cutoff, method)
        values.append(total_guess_prob_from_mus(mus, n_max=photon_cutoff, loss=loss, denominator=num_states))
    return values


def untrusted_calibration_summary(
    num_states: int,
    intensity_cap: float,
    loss: float,
    photon_cutoff: int,
) -> Dict[str, float | np.ndarray]:
    """Compute both heuristic and discrete witnesses for quick comparison."""

    discrete_result = discrete_optimize(num_states, intensity_cap, loss, photon_cutoff)
    heuristic_result = optimize_poisson_spacing(num_states, intensity_cap, loss, photon_cutoff)
    return {
        "heuristic_prob": heuristic_result.probability,
        "heuristic_intensities": heuristic_result.intensities,
        "discrete_prob": discrete_result.probability,
        "discrete_intensities": discrete_result.intensities,
        "delta": abs(heuristic_result.probability - discrete_result.probability),
    }


def subspace_tail_term(
    num_states: int,
    subspace_cutoff: int,
    intensity_cap: float,
    loss: float,
) -> float:
    """Return the worst-case out-of-subspace tail term divided by ``N``."""

    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if subspace_cutoff < 0:
        raise ValueError("subspace_cutoff must be non-negative")
    if intensity_cap <= 0:
        raise ValueError("intensity_cap must be positive")
    return worst_case_poisson_tail(subspace_cutoff + 1, loss * intensity_cap) / num_states


def subspace_witness_table(
    num_states: int,
    intensity_cap: float,
    loss: float,
    subspace_cutoff: int,
    method: str = "discrete",
    photon_cutoff: int | None = None,
) -> List[float]:
    r"""Return the subspace-corrected witness table ``\widetilde W_R`` for ``R = 1..N``."""

    cutoff = subspace_cutoff if photon_cutoff is None else photon_cutoff
    tail = subspace_tail_term(num_states, subspace_cutoff, intensity_cap, loss)
    values: List[float] = []
    for subset_size in range(1, num_states + 1):
        mus = optimal_mus_for_method(
            subset_size,
            intensity_cap,
            loss,
            cutoff,
            method,
        )
        inside = total_guess_prob_from_mus(mus, n_max=cutoff, loss=loss, denominator=num_states)
        values.append(inside + tail)
    return values


def intensity_bounded_witness_tables(
    num_states: int,
    intensity_cap: float,
    loss: float,
    photon_cutoff: int,
    subspace_cutoff: int,
    method: str = "discrete",
) -> Dict[str, List[float] | float]:
    """Return both full-space and subspace-corrected witness tables."""

    full = untrusted_witness_table(num_states, intensity_cap, loss, photon_cutoff, method=method)
    subspace = subspace_witness_table(
        num_states,
        intensity_cap,
        loss,
        subspace_cutoff=subspace_cutoff,
        method=method,
        photon_cutoff=subspace_cutoff,
    )
    tail = subspace_tail_term(num_states, subspace_cutoff, intensity_cap, loss)
    return {"full": full, "subspace": subspace, "tail": tail}


def _poisson_interval_sum(a: int, b: int, mu: float) -> float:
    """Return ``sum_{n=a}^b q(n|mu)`` using incomplete gamma identities."""

    if b < a:
        return 0.0
    if a < 0 or b < 0:
        raise ValueError("Interval endpoints must be non-negative")
    # P[N <= k] = gammaincc(k+1, mu)
    return float(gammaincc(b + 1, mu) - gammaincc(a, mu))


def _poisson_tail_from(n_min: int, mu: float) -> float:
    """Return ``sum_{n=n_min}^∞ q(n|mu)`` as a regularized gamma tail."""

    if n_min < 0:
        raise ValueError("n_min must be non-negative")
    # P[N >= n_min] = 1 - P[N <= n_min-1]
    return float(1.0 - gammaincc(n_min, mu))


def global_intensity_bounded_witness(
    num_inputs: int,
    resolution: int,
    intensity_cap: float,
    loss: float = 1.0,
    tol: float = 1e-12,
    n_limit: int | None = None,
) -> float:
    r"""Return the full-space intensity-bounded witness (no subspace split).

    This corresponds to the optimisation

    $$\frac{1}{N} \max_{0 \le \mu_1,\dots,\mu_R \le I} \sum_{n=0}^{\infty} \max_i q(n\mid \ell\mu_i),$$

    i.e. a genuine *global* witness for an ``R``-outcome measurement, not a
    witness for subspace resolution.

    Implementation details
    ----------------------
    For Poisson families the maximiser index is monotone in ``n``, so the optimal
    strategy partitions photon numbers into ``R`` contiguous intervals, assigns
    one intensity to each interval, and uses ML decoding. The last (infinite)
    interval is always maximised at the boundary ``mu = I``.
    """

    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if resolution > num_inputs:
        # Still well-defined mathematically, but uninteresting for certification.
        resolution = num_inputs
    if intensity_cap <= 0:
        raise ValueError("intensity_cap must be positive")
    if not (0.0 < loss <= 1.0):
        raise ValueError("loss must lie in (0, 1]")

    mu_eff = float(loss * intensity_cap)

    # Choose a truncation point high enough so that the remaining tail at mu_eff
    # is negligible; the last interval uses the exact tail anyway.
    if n_limit is None:
        guess = int(np.ceil(mu_eff + 10.0 * np.sqrt(mu_eff + 1.0) + 10.0))
        n_limit = max(10, guess)
        while _poisson_tail_from(n_limit + 1, mu_eff) > tol:
            n_limit += 5

    R = resolution
    # R = 1: only one outcome; best is always to guess n=0 with mu=0.
    if R == 1:
        return 1.0 / num_inputs

    # Precompute finite-interval values f(a,b) for 1 <= a <= b <= n_limit.
    # Interval [a..b] is assigned the intensity that maximises its Poisson mass:
    # either the interior solution solving q(a-1|mu)=q(b|mu), clipped to I.
    f = np.full((n_limit + 2, n_limit + 2), -np.inf, dtype=float)
    for a in range(1, n_limit + 1):
        for b in range(a, n_limit + 1):
            mu_star = min(intensity_cap, interior_intensity(a - 1, b))
            f[a, b] = _poisson_interval_sum(a, b, loss * mu_star)

    # Tail values for the last interval starting at s (1..n_limit+1):
    # sum_{n=s}^∞ q(n|loss*I).
    tail = np.zeros(n_limit + 3, dtype=float)
    for s in range(1, n_limit + 2):
        tail[s] = _poisson_tail_from(s, mu_eff)

    # Dynamic programming over contiguous partitions of [1..n_limit] into
    # (R-1) intervals, where the last interval is infinite.
    # Choose cutpoints e_2 < ... < e_{R-1} in [1..n_limit].
    # Objective: 1 + sum_{i=2}^{R-1} f(e_{i-1}+1, e_i) + tail[e_{R-1}+1].
    # Special case R=2: 1 + tail[1].
    if R == 2:
        return float((1.0 + tail[1]) / num_inputs)

    # dp[k][e] = best value up to endpoint e using k finite intervals
    # (covering n=1..e), where k = 1 corresponds to interval2 only.
    # We need k = R-2 finite intervals; the last infinite interval begins at e+1.
    k_max = R - 2
    dp = np.full((k_max + 1, n_limit + 1), -np.inf, dtype=float)

    # Base: k=1 uses a single finite interval [1..e].
    for e in range(1, n_limit + 1):
        dp[1, e] = f[1, e]

    for k in range(2, k_max + 1):
        # k-th interval ends at e, previous ends at p < e.
        for e in range(k, n_limit + 1):
            best_val = -np.inf
            for p in range(k - 1, e):
                candidate = dp[k - 1, p] + f[p + 1, e]
                if candidate > best_val:
                    best_val = candidate
            dp[k, e] = best_val

    best_total = -np.inf
    for e_last in range(k_max, n_limit + 1):
        candidate = dp[k_max, e_last] + tail[e_last + 1]
        if candidate > best_total:
            best_total = candidate

    return float((1.0 + best_total) / num_inputs)


# ---------------------------------------------------------------------------
# Certification-table reproduction helpers


def experimental_guessing_probability(
    probabilities: Sequence[Sequence[float]],
    num_inputs: int,
) -> float:
    """Return ``P_guess`` computed from the experimental click statistics."""

    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    if len(probabilities) < num_inputs:
        raise ValueError("Not enough probability rows for the requested inputs")
    arr = np.asarray(probabilities[:num_inputs], dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    arr = arr / row_sums
    per_outcome = np.max(arr, axis=0)
    return float(np.sum(per_outcome) / num_inputs)


def build_certification_table(
    mus: Sequence[float],
    cases: Sequence[CertificationConfig],
    loss: float = 1.0,
    measurement_probabilities: Sequence[Sequence[float]] | None = None,
    observed_guesses: Dict[int, float] | None = None,
    method: str = "discrete",
) -> List[CertificationRow]:
    """Return trusted/untrusted certification rows for arbitrary input data."""

    rows: List[CertificationRow] = []
    for case in cases:
        if case.num_inputs > len(mus):
            raise ValueError("Not enough calibrated intensities for the requested case")
        subset = tuple(mus[: case.num_inputs])
        required_indices = (0,) if subset and subset[0] == 0.0 else None
        trusted = maximise_trusted_witness(
            subset,
            subset_size=case.resolution,
            photon_cutoff=case.max_photons,
            denominator=case.num_inputs,
            tail_mu=subset[-1],
            required_indices=required_indices,
        ).probability
        cap = float(case.intensity_cap if case.intensity_cap is not None else max(subset))
        if cap <= 0:
            raise ValueError("Intensity cap must be positive")
        subspace_table = subspace_witness_table(
            num_states=case.num_inputs,
            intensity_cap=cap,
            loss=loss,
            subspace_cutoff=case.max_photons,
            method=method,
            photon_cutoff=case.max_photons,
        )
        untrusted_global = global_intensity_bounded_witness(
            num_inputs=case.num_inputs,
            resolution=case.resolution,
            intensity_cap=cap,
            loss=loss,
        )
        untrusted = subspace_table[case.resolution - 1]
        experimental_value = None
        if observed_guesses is not None and case.num_inputs in observed_guesses:
            experimental_value = observed_guesses[case.num_inputs]
        elif measurement_probabilities is not None:
            experimental_value = experimental_guessing_probability(measurement_probabilities, case.num_inputs)
        trusted_efficiency = None
        untrusted_efficiency = None
        if experimental_value is not None:
            trusted_efficiency = solve_trusted_subspace_efficiency(
                subset,
                observed_probability=experimental_value,
                max_photons=case.max_photons,
                denominator=case.num_inputs,
            )
            untrusted_efficiency = solve_untrusted_subspace_efficiency(
                num_inputs=case.num_inputs,
                max_photons=case.max_photons,
                intensity_cap=cap,
                observed_probability=experimental_value,
            )
        rows.append(
            CertificationRow(
                num_inputs=case.num_inputs,
                max_photons=case.max_photons,
                resolution=case.resolution,
                trusted_bound=trusted,
                untrusted_bound=untrusted,
                untrusted_full_space=untrusted_global,
                experimental_guess=experimental_value,
                trusted_efficiency=trusted_efficiency,
                untrusted_efficiency=untrusted_efficiency,
            )
        )
    return rows


def build_manuscript_certification_table() -> List[CertificationRow]:
    """Return the certification rows for the data released with the manuscript."""

    return build_certification_table(
        mus=experimental_data.INTENSITIES,
        cases=MANUSCRIPT_CERTIFICATION_CASES,
        measurement_probabilities=experimental_data.CLICK_PROBABILITIES,
        observed_guesses=experimental_data.OBSERVED_GUESSING_PROBABILITIES,
    )


def format_certification_table(rows: Sequence[CertificationRow]) -> str:
    """Pretty table showing trusted vs. intensity-bounded witnesses."""

    if not rows:
        return ""
    include_guess = any(row.experimental_guess is not None for row in rows)
    include_efficiency = any(
        row.trusted_efficiency is not None or row.untrusted_efficiency is not None
        for row in rows
    )
    header = ["m", "N", "R", "Trusted [%]", "Int-bound (subspace) [%]", "Int-bound (global) [%]"]
    if include_guess:
        header.extend(["P_guess [%]", "cert. trusted", "cert. intensity"])
    if include_efficiency:
        header.extend(["eta trusted [%]", "eta intensity [%]"])
    lines = [" | ".join(header)]
    for row in rows:
        base = [
            f"{row.max_photons:2d}",
            f"{row.num_inputs:2d}",
            f"{row.resolution:2d}",
            f"{row.trusted_bound * 100:11.2f}",
            f"{row.untrusted_bound * 100:21.2f}",
            f"{row.untrusted_full_space * 100:18.2f}",
        ]
        if include_guess:
            guess = None if row.experimental_guess is None else row.experimental_guess * 100
            base.append(f"{guess:11.2f}" if guess is not None else "     -    ")
            cert_trusted = row.certified_resolution_trusted()
            cert_intensity = row.certified_resolution_intensity()
            base.append(f"{cert_trusted:13d}" if cert_trusted is not None else "      -     ")
            base.append(f"{cert_intensity:15d}" if cert_intensity is not None else "       -       ")
        if include_efficiency:
            eta_trusted = None if row.trusted_efficiency is None else row.trusted_efficiency * 100
            eta_intensity = None if row.untrusted_efficiency is None else row.untrusted_efficiency * 100
            base.append(f"{eta_trusted:15.2f}" if eta_trusted is not None else "       -       ")
            base.append(f"{eta_intensity:17.2f}" if eta_intensity is not None else "        -        ")
        lines.append(" | ".join(base))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Efficiency-limited intrinsic resolution
# (Methods: Resolution of the efficiency-limited detector)


def _binomial_matrix(n_max: int, efficiency: float) -> List[List[float]]:
    """Return the loss channel matrix ``Lambda_eta`` up to ``n_max`` photons."""

    levels = n_max + 1
    matrix = [[0.0 for _ in range(levels)] for _ in range(levels)]
    for n in range(levels):
        for clicks in range(levels):
            if clicks > n:
                continue
            matrix[clicks][n] = comb(n, clicks) * (efficiency ** clicks) * ((1.0 - efficiency) ** (n - clicks))
    return matrix


def _generate_memberships(levels: int, coarse_outputs: int) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate all partitions of ``levels`` photon numbers into ``coarse_outputs`` labels."""

    if coarse_outputs <= 0 or coarse_outputs > levels:
        return ()
    assignment = [0] * levels
    results: List[Tuple[int, ...]] = []

    def backtrack(index: int, used_labels: int) -> None:
        if index == levels:
            if used_labels == coarse_outputs:
                results.append(tuple(assignment))
            return
        remaining = levels - index - 1
        for label in range(min(used_labels, coarse_outputs)):
            assignment[index] = label
            backtrack(index + 1, used_labels)
        if used_labels < coarse_outputs and remaining >= (coarse_outputs - (used_labels + 1)):
            assignment[index] = used_labels
            backtrack(index + 1, used_labels + 1)

    backtrack(0, 0)
    return tuple(results)


def _build_partition_lp(n_max: int, coarse_outputs: int) -> PartitionLPSystem | None:
    """Construct the sparse LP encoding the Methods simulability test."""

    partitions = _generate_memberships(n_max + 1, coarse_outputs)
    if not partitions:
        return None
    levels = n_max + 1
    n_partitions = len(partitions)
    n_x = n_partitions * coarse_outputs * levels
    n_w = n_partitions
    variables = n_x + n_w
    eq1 = 1
    eq2 = n_partitions * coarse_outputs
    eq3 = levels * levels
    total_rows = eq1 + eq2 + eq3
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    b = np.zeros(total_rows, dtype=float)
    row_idx = 0
    for lam in range(n_partitions):
        rows.append(row_idx)
        cols.append(n_x + lam)
        data.append(1.0)
    b[row_idx] = 1.0
    row_idx += 1
    for lam in range(n_partitions):
        for i in range(coarse_outputs):
            base = (lam * coarse_outputs + i) * levels
            for clicks in range(levels):
                rows.append(row_idx)
                cols.append(base + clicks)
                data.append(1.0)
            rows.append(row_idx)
            cols.append(n_x + lam)
            data.append(-1.0)
            row_idx += 1
    eq3_offset = row_idx
    for n in range(levels):
        for clicks in range(levels):
            for lam, partition in enumerate(partitions):
                label = partition[n]
                base = (lam * coarse_outputs + label) * levels
                rows.append(row_idx)
                cols.append(base + clicks)
                data.append(1.0)
            row_idx += 1
    matrix = csr_matrix((data, (rows, cols)), shape=(total_rows, variables))
    return PartitionLPSystem(
        matrix=matrix,
        b_template=b,
        eq3_offset=eq3_offset,
        eq3_size=levels * levels,
        num_vars=variables,
        levels=levels,
        partitions=partitions,
        coarse_outputs=coarse_outputs,
    )


def _can_simulate_cached(n_max: int, coarse_outputs: int, efficiency_key: int) -> bool:
    """Solve the feasibility LP for a fixed efficiency discretised at 1e-6 resolution."""

    system = _build_partition_lp(n_max, coarse_outputs)
    if system is None:
        return False
    efficiency = efficiency_key / 1_000_000
    probabilities = _binomial_matrix(n_max, efficiency)
    b = system.b_template.copy()
    flat = [probabilities[clicks][n] for n in range(system.levels) for clicks in range(system.levels)]
    b[system.eq3_offset : system.eq3_offset + system.eq3_size] = flat
    result = linprog(
        c=np.zeros(system.num_vars),
        A_eq=system.matrix,
        b_eq=b,
        bounds=[(0.0, None)] * system.num_vars,
        method="highs",
    )
    return result.status == 0


def can_simulate_with_coarse_outputs(n_max: int, coarse_outputs: int, efficiency: float) -> bool:
    """Return ``True`` if the detector is simulable with ``coarse_outputs`` outcomes."""

    if coarse_outputs <= 0 or coarse_outputs > n_max + 1:
        return False
    efficiency_key = int(round(efficiency * 1_000_000))
    return _can_simulate_cached(n_max, coarse_outputs, efficiency_key)


def intrinsic_resolution_threshold(
    n_max: int,
    resolution: int,
    tol: float = 5e-4,
    max_iter: int = 25,
) -> float | None:
    """Binary search the efficiency where the detector ceases to be ``(resolution-1)`` simulable."""

    coarse_outputs = resolution - 1
    if coarse_outputs <= 0 or coarse_outputs > n_max + 1:
        return None
    if can_simulate_with_coarse_outputs(n_max, coarse_outputs, 1.0):
        return None
    if not can_simulate_with_coarse_outputs(n_max, coarse_outputs, 0.0):
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if can_simulate_with_coarse_outputs(n_max, coarse_outputs, mid):
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return high


def efficiency_resolution_table(
    n_levels: Sequence[int],
    resolutions: Sequence[int],
    tol: float = 5e-4,
) -> Dict[int, Dict[int, float | None]]:
    """Return the efficiency thresholds (in %) arranged as a nested dictionary."""

    table: Dict[int, Dict[int, float | None]] = {}
    for n_max in n_levels:
        row: Dict[int, float | None] = {}
        for resolution in resolutions:
            threshold = intrinsic_resolution_threshold(n_max, resolution, tol=tol)
            row[resolution] = None if threshold is None else 100.0 * threshold
        table[n_max] = row
    return table


def format_efficiency_table(table: Dict[int, Dict[int, float | None]]) -> str:
    """Pretty-print the nested dictionary as a manuscript-style table."""

    if not table:
        return ""
    resolution_values = sorted({res for row in table.values() for res in row})
    header = ["n\\M"] + [str(res) for res in resolution_values]
    lines = [" | ".join(header)]
    for n_max in sorted(table.keys()):
        row = [f"[0..{n_max}] "]
        for res in resolution_values:
            value = table[n_max].get(res)
            row.append("-" if value is None else f"{value:5.1f}")
        lines.append(" | ".join(row))
    return "\n".join(lines)


resolution_table = efficiency_resolution_table


# ---------------------------------------------------------------------------
# Demonstration block


if __name__ == "__main__":
    # -- Trusted calibration demo -------------------------------------------------
    trusted_results = trusted_witness_scan(
        experimental_data.INTENSITIES,
        subset_sizes=range(3, len(experimental_data.INTENSITIES) + 1),
        photon_cutoff=8,
        denominator=len(experimental_data.INTENSITIES),
    )
    print("Trusted calibration witness (Methods: witness-bound derivation):")
    for res in trusted_results:
        print(
            f"  R={len(res.subset_indices):2d} subset={res.subset_indices} "
            f"mu={tuple(round(x, 4) for x in res.subset_mus)} P_guess={res.probability:.9f}"
        )

    # -- Intensity-bounded demo ---------------------------------------------------
    summary = untrusted_calibration_summary(num_states=6, intensity_cap=8.0, loss=1.0, photon_cutoff=6)
    print("\nIntensity-bounded calibration (Methods: numerical optimization):")
    print(
        f"  heuristic={summary['heuristic_prob']:.9f} discrete={summary['discrete_prob']:.9f} "
        f"Δ={summary['delta']:.3e}"
    )
    tables = intensity_bounded_witness_tables(
        num_states=6,
        intensity_cap=8.0,
        loss=1.0,
        photon_cutoff=6,
        subspace_cutoff=6,
        method="discrete",
    )
    print("  Witness table (full space):")
    for idx, value in enumerate(tables["full"], start=1):
        print(f"    R={idx:2d}: W={value:.9f}")
    print(f"  Tail term added for subspace correction: {tables['tail']:.9f}")
    print("  Witness table (subspace corrected):")
    for idx, value in enumerate(tables["subspace"], start=1):
        print(f"    R={idx:2d}: W_tilde={value:.9f}")

    # -- Certification-table reproduction ----------------------------------------
    certification_rows = build_manuscript_certification_table()
    print("\nCertification table reproduction (trusted vs. intensity-bounded):")
    print(format_certification_table(certification_rows))

    # -- Resolution table demo ----------------------------------------------------
    table = resolution_table(n_levels=(2, 3, 4), resolutions=(3, 4, 5, 6))
    print("\nIntrinsic resolution table (% efficiencies):")
    print(format_efficiency_table(table))
