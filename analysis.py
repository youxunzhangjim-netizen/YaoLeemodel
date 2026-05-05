#!/usr/bin/env python3
"""Analysis and runtime helpers for the Yao-Lee driver.

This file owns lazy Tenax imports, progress bars, stage timing, entropy-profile
analysis, and alpha-beta phase-scan analysis/classification. Model
construction stays in ``models.py`` and plot rendering stays in
``plot_outputs.py``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

ENTROPY_ORDERS = (1, 2, 3, 4)
GROUND_MANIFOLD_ABS_TOL_DEFAULT = 1e-12
GROUND_MANIFOLD_REL_TOL_DEFAULT = 1e-12


_TENAX_API: Dict[str, Any] | None = None
_TENAX_COMPAT_WARNED = False


def get_tenax_api() -> Dict[str, Any]:
    """Lazy import so --help still works even if Tenax is not installed."""
    global _TENAX_API
    if _TENAX_API is not None:
        return _TENAX_API

    try:
        import tenax  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Tenax is required for DMRG. Install it in your active environment, then rerun."
        ) from exc

    global _TENAX_COMPAT_WARNED
    tenax_version = str(getattr(tenax, "__version__", "0.0.0"))
    numpy_major = int(str(np.__version__).split(".")[0])
    # Known risky combination:
    # Tenax 0.2.x + NumPy >= 2 can fail on complex AutoMPO terms if Tenax is unpatched.
    if tenax_version.startswith("0.2.") and numpy_major >= 2 and not _TENAX_COMPAT_WARNED:
        print(
            "[backend] Warning: Tenax 0.2.x with NumPy >=2 may fail on complex AutoMPO terms "
            "(common error: complex128 -> float64 casting)."
        )
        _TENAX_COMPAT_WARNED = True

    for required in ("DMRGConfig", "build_random_mps", "dmrg"):
        if not hasattr(tenax, required):
            raise RuntimeError(f"Tenax is missing required API '{required}'.")

    _TENAX_API = {
        "DMRGConfig": tenax.DMRGConfig,
        "build_random_mps": tenax.build_random_mps,
        "dmrg": tenax.dmrg,
        "iDMRGConfig": getattr(tenax, "iDMRGConfig", None),
        "idmrg": getattr(tenax, "idmrg", None),
        "build_auto_mpo": getattr(tenax, "build_auto_mpo", None),
        "AutoMPO": getattr(tenax, "AutoMPO", None),
        "expectation_value": getattr(tenax, "expectation_value", None),
        "correlation": getattr(tenax, "correlation", None),
        "expectation": getattr(tenax, "expectation", None),
    }
    return _TENAX_API


_TQDM_IMPORT_FAILED_WARNED = False


def _get_tqdm(enabled: bool):
    global _TQDM_IMPORT_FAILED_WARNED
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm
    except Exception:
        if not _TQDM_IMPORT_FAILED_WARNED:
            print("[progress] tqdm is not installed; continuing without progress bars.")
            _TQDM_IMPORT_FAILED_WARNED = True
        return None


def _make_progress_bar(
    enabled: bool,
    total: int,
    desc: str,
    unit: str,
    leave: bool = False,
):
    tqdm = _get_tqdm(enabled)
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave)


def _start_stage(name: str, enabled: bool) -> float:
    if enabled:
        print(f"[stage] {name} started")
    return time.perf_counter()


def _end_stage(name: str, stage_start: float, enabled: bool) -> float:
    elapsed = time.perf_counter() - stage_start
    if enabled:
        print(f"[stage] {name} finished in {elapsed:.2f}s")
    return elapsed


def resolve_low_energy_spectrum(
    eigenvalues: Sequence[float],
    *,
    check_ground_state_degeneracy: bool = True,
    hilbert_dim: int | None = None,
    degeneracy_tolerance_abs: float = GROUND_MANIFOLD_ABS_TOL_DEFAULT,
    degeneracy_tolerance_rel: float = GROUND_MANIFOLD_REL_TOL_DEFAULT,
) -> Dict[str, Any]:
    """Resolve the ground manifold and the first level above it.

    The default tolerance is intentionally strict, using 1e-12 scale, so ED
    gaps are not swallowed into the degeneracy count.
    """

    values = np.sort(np.asarray(eigenvalues, dtype=float))
    if values.size == 0:
        raise ValueError("Cannot resolve an empty low-energy spectrum.")

    e0 = float(values[0])
    raw_second_energy = float(values[1]) if values.size >= 2 else None
    raw_second_gap = (
        float(raw_second_energy - e0)
        if raw_second_energy is not None
        else None
    )

    if not bool(check_ground_state_degeneracy):
        return {
            "ground_state_energy": e0,
            "ground_state_degeneracy_check_enabled": False,
            "ground_state_degeneracy": None,
            "ground_state_degeneracy_tolerance": None,
            "ground_state_degeneracy_is_lower_bound": None,
            "ground_state_degeneracy_status": "not_checked",
            "ground_state_indices": [0],
            "first_excited_index": 1 if values.size >= 2 else None,
            "raw_second_energy": raw_second_energy,
            "raw_second_gap": raw_second_gap,
            "first_excited_energy": raw_second_energy,
            "spectral_gap": raw_second_gap,
            "low_energy_gaps_from_ground": [float(value - e0) for value in values],
            "note": (
                "Ground-state degeneracy check disabled: first_excited_energy is the raw "
                "second ED eigenvalue and may still belong to a degenerate ground manifold."
            ),
        }

    scale = max(1.0, abs(e0))
    degeneracy_tolerance = max(
        float(degeneracy_tolerance_abs),
        float(degeneracy_tolerance_rel) * scale,
    )
    gaps_from_ground = values - e0
    ground_indices = np.flatnonzero(np.abs(gaps_from_ground) <= degeneracy_tolerance)
    if ground_indices.size == 0:
        ground_indices = np.asarray([0], dtype=int)
    above_ground = np.flatnonzero(np.abs(gaps_from_ground) > degeneracy_tolerance)
    first_excited_index = int(above_ground[0]) if above_ground.size > 0 else None
    first_excited_energy = (
        float(values[first_excited_index])
        if first_excited_index is not None
        else None
    )
    spectral_gap = (
        float(first_excited_energy - e0)
        if first_excited_energy is not None
        else None
    )
    returned_count = int(values.size)
    lower_bound = bool(
        above_ground.size == 0
        and hilbert_dim is not None
        and returned_count < int(hilbert_dim)
    )
    return {
        "ground_state_energy": e0,
        "ground_state_degeneracy_check_enabled": True,
        "ground_state_degeneracy": int(ground_indices.size),
        "ground_state_degeneracy_tolerance": float(degeneracy_tolerance),
        "ground_state_degeneracy_absolute_tolerance": float(degeneracy_tolerance_abs),
        "ground_state_degeneracy_relative_tolerance": float(degeneracy_tolerance_rel),
        "ground_state_degeneracy_is_lower_bound": lower_bound,
        "ground_state_degeneracy_status": "lower_bound" if lower_bound else "resolved",
        "ground_state_indices": [int(index) for index in ground_indices],
        "first_excited_index": first_excited_index,
        "raw_second_energy": raw_second_energy,
        "raw_second_gap": raw_second_gap,
        "first_excited_energy": first_excited_energy,
        "spectral_gap": spectral_gap,
        "gap_above_ground_manifold": spectral_gap,
        "low_energy_gaps_from_ground": [float(gap) for gap in gaps_from_ground],
        "note": (
            "Ground-state degeneracy check enabled: first_excited_energy is the first "
            "resolved level above the ground-state manifold using the recorded tolerance."
        ),
    }


def penalty_excited_state_result(
    *,
    status: str,
    reason: str,
    first_excited_energy: float | None = None,
    spectral_gap: float | None = None,
    penalty_weight_used: float | None = None,
    candidate_variance: float | None = None,
    candidate_max_overlap: float | None = None,
) -> Dict[str, Any]:
    return {
        "method": "finite_dmrg_penalty_excited_state",
        "first_excited_energy": (
            float(first_excited_energy)
            if first_excited_energy is not None
            else None
        ),
        "spectral_gap": (
            float(spectral_gap)
            if spectral_gap is not None
            else None
        ),
        "status": str(status),
        "reason": str(reason),
        "penalty_weight_used": (
            float(penalty_weight_used)
            if penalty_weight_used is not None
            else None
        ),
        "candidate_variance": (
            float(candidate_variance)
            if candidate_variance is not None
            else None
        ),
        "candidate_max_overlap": (
            float(candidate_max_overlap)
            if candidate_max_overlap is not None
            else None
        ),
    }


def _canonicalize_and_normalize_mps(mps: Any) -> Any:
    state = mps
    for method_name in ("canonicalize", "right_canonicalize", "left_canonicalize"):
        method = getattr(state, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
        except TypeError:
            continue
        if result is not None:
            state = result
        break

    normalize = getattr(state, "normalize", None)
    if callable(normalize):
        result = normalize()
        if result is not None:
            state = result
    return state


def _full_mps_overlap(left_mps: Any, right_mps: Any) -> complex:
    overlap = getattr(left_mps, "overlap", None)
    if callable(overlap):
        return complex(overlap(right_mps))
    overlap = getattr(right_mps, "overlap", None)
    if callable(overlap):
        return complex(np.conjugate(overlap(left_mps)))
    inner = getattr(left_mps, "inner", None)
    if callable(inner):
        return complex(inner(right_mps))
    inner = getattr(right_mps, "inner", None)
    if callable(inner):
        return complex(np.conjugate(inner(left_mps)))
    raise RuntimeError(
        "No full-MPS overlap method is available; refusing tensor-wise overlap checks."
    )


def _evaluate_mps_mpo_energy(H_mpo: Any, mps: Any) -> float:
    if hasattr(mps, "expectation_value"):
        value = mps.expectation_value(H_mpo)
        return float(np.real_if_close(value))
    if hasattr(mps, "expectation"):
        value = mps.expectation(H_mpo)
        return float(np.real_if_close(value))
    exp_fn = get_tenax_api().get("expectation")
    if callable(exp_fn):
        value = exp_fn(H_mpo, mps)
        return float(np.real_if_close(value))
    raise RuntimeError(
        "No expectation evaluator found for original-Hamiltonian candidate energy."
    )


def _evaluate_mps_mpo_variance(H_mpo: Any, mps: Any, energy: float) -> float:
    for method_name in ("variance", "energy_variance", "expectation_variance"):
        method = getattr(mps, method_name, None)
        if not callable(method):
            continue
        try:
            value = method(H_mpo)
        except TypeError:
            continue
        return max(0.0, float(np.real_if_close(value)))
    h2_method = getattr(mps, "expectation_value_squared", None)
    if callable(h2_method):
        h2_value = h2_method(H_mpo)
        return max(0.0, float(np.real_if_close(h2_value)) - float(energy) ** 2)
    raise RuntimeError(
        "No MPO variance evaluator is available; refusing to certify a DMRG gap."
    )


def _suggest_penalty_weights(
    ED_gap_hint: float | None,
    penalty_weights: Sequence[float] | None,
) -> List[float]:
    if penalty_weights is not None:
        weights = [float(weight) for weight in penalty_weights if np.isfinite(float(weight))]
        return [weight for weight in weights if weight > 0.0]
    if ED_gap_hint is not None:
        gap = float(ED_gap_hint)
        if np.isfinite(gap) and gap > 0.0:
            return [5.0 * gap, 10.0 * gap, 50.0 * gap]
    return [1.0, 5.0, 10.0, 50.0]


def find_dmrg_excited_state(
    H_mpo: Any,
    ground_mps_list: Sequence[Any],
    E0: float,
    ED_gap_hint: float | None = None,
    penalty_weights: Sequence[float] | None = None,
    overlap_tol: float = 1e-6,
    energy_tol: float = 1e-7,
    variance_tol: float = 1e-7,
    max_attempts: int = 10,
    *,
    required_ground_degeneracy: int | None = None,
    penalty_dmrg_runner: Callable[..., Any] | None = None,
) -> Dict[str, Any]:
    """Analyze a penalty-state finite-DMRG excited-state search."""

    raw_ground_states = [state for state in ground_mps_list if state is not None]
    if required_ground_degeneracy is not None:
        try:
            required_count = int(required_ground_degeneracy)
        except (TypeError, ValueError):
            required_count = 0
        if required_count > 0 and len(raw_ground_states) < required_count:
            return penalty_excited_state_result(
                status="not_found",
                reason="incomplete ground manifold",
            )

    if len(raw_ground_states) == 0:
        return penalty_excited_state_result(
            status="not_found",
            reason="incomplete ground manifold",
        )

    try:
        ground_states = [
            _canonicalize_and_normalize_mps(state)
            for state in raw_ground_states
        ]
    except Exception as exc:
        return penalty_excited_state_result(
            status="not_found",
            reason=f"ground-state MPS canonicalization failed: {exc}",
        )

    weights = _suggest_penalty_weights(ED_gap_hint, penalty_weights)
    if len(weights) == 0:
        return penalty_excited_state_result(
            status="not_found",
            reason="no positive finite penalty weights",
        )

    if penalty_dmrg_runner is None:
        return penalty_excited_state_result(
            status="not_found",
            reason="penalty-state DMRG runner unavailable",
        )

    best_accepted: Dict[str, Any] | None = None
    best_rejected: Dict[str, Any] | None = None
    e0 = float(E0)
    attempts_done = 0
    for weight in weights:
        for attempt_index in range(max(1, int(max_attempts))):
            attempts_done += 1
            try:
                candidate_result = penalty_dmrg_runner(
                    H_mpo=H_mpo,
                    ground_mps_list=ground_states,
                    penalty_weight=float(weight),
                    attempt_index=int(attempt_index),
                )
            except Exception:
                continue

            if isinstance(candidate_result, dict):
                candidate_mps = candidate_result.get("mps")
                if candidate_mps is None:
                    candidate_mps = candidate_result.get("state")
            elif isinstance(candidate_result, (tuple, list)) and len(candidate_result) > 0:
                candidate_mps = candidate_result[0]
            else:
                candidate_mps = candidate_result
            if candidate_mps is None:
                continue

            try:
                candidate_mps = _canonicalize_and_normalize_mps(candidate_mps)
                overlaps = [
                    abs(_full_mps_overlap(ground_state, candidate_mps))
                    for ground_state in ground_states
                ]
                max_overlap = float(max(overlaps)) if overlaps else 0.0
                energy = _evaluate_mps_mpo_energy(H_mpo, candidate_mps)
                variance = _evaluate_mps_mpo_variance(H_mpo, candidate_mps, energy)
            except Exception:
                continue

            record = {
                "energy": float(energy),
                "gap": float(energy - e0),
                "variance": float(variance),
                "max_overlap": float(max_overlap),
                "penalty_weight": float(weight),
            }
            if best_rejected is None or record["energy"] < best_rejected["energy"]:
                best_rejected = record
            if (
                max_overlap < float(overlap_tol)
                and energy > e0 + float(energy_tol)
                and variance < float(variance_tol)
            ):
                if best_accepted is None or energy < best_accepted["energy"]:
                    best_accepted = record

    if best_accepted is None:
        reason = (
            "no distinct low-variance penalty-DMRG candidate"
            if attempts_done > 0
            else "penalty-state DMRG produced no candidates"
        )
        return penalty_excited_state_result(
            status="not_found",
            reason=reason,
            candidate_variance=(
                best_rejected.get("variance") if best_rejected is not None else None
            ),
            candidate_max_overlap=(
                best_rejected.get("max_overlap") if best_rejected is not None else None
            ),
        )

    return penalty_excited_state_result(
        status="found",
        reason="distinct low-variance state found",
        first_excited_energy=float(best_accepted["energy"]),
        spectral_gap=float(best_accepted["gap"]),
        penalty_weight_used=float(best_accepted["penalty_weight"]),
        candidate_variance=float(best_accepted["variance"]),
        candidate_max_overlap=float(best_accepted["max_overlap"]),
    )


def _entropy_from_probabilities(probabilities: np.ndarray, order_n: int) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 1e-15]
    if p.size == 0:
        return 0.0
    p = p / np.sum(p)
    if order_n == 1:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p ** order_n)) / (1.0 - float(order_n)))


def _entropy_dict_from_singular_values(singular_values: np.ndarray, orders: Tuple[int, ...]) -> Dict[str, float]:
    sv = np.asarray(singular_values, dtype=float)
    sv = sv[sv > 1e-15]
    if sv.size == 0:
        return {f"S{n}": 0.0 for n in orders}
    probabilities = sv ** 2
    probabilities = probabilities / np.sum(probabilities)
    return {f"S{n}": _entropy_from_probabilities(probabilities, n) for n in orders}


def _summarize_entropy_values(entropies: Dict[str, List[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, values in entropies.items():
        if len(values) == 0:
            continue
        values_arr = np.asarray(values, dtype=float)
        summary[f"{key}_mean"] = float(np.mean(values_arr))
        summary[f"{key}_max"] = float(np.max(values_arr))
        summary[f"{key}_min"] = float(np.min(values_arr))
    return summary


def _build_entropy_profile(
    method_label: str,
    cuts: List[float],
    total_span: float,
    entropies: Dict[str, List[float]],
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cuts_norm = [float(cut) / float(total_span) for cut in cuts] if total_span > 0 else [0.0 for _ in cuts]
    profile: Dict[str, Any] = {
        "method": method_label,
        "cuts": [float(cut) for cut in cuts],
        "cuts_normalized": cuts_norm,
        "entropies": {key: [float(value) for value in values] for key, values in entropies.items()},
        "summary": _summarize_entropy_values(entropies),
    }
    if context is not None:
        profile["context"] = context
    return profile


def compute_tenax_finite_mps_entropy_profile(
    mps: Any,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    show_progress: bool = True,
) -> Dict[str, Any]:
    mps_work = mps
    if hasattr(mps_work, "compute_singular_values"):
        mps_work = mps_work.compute_singular_values()
    if not hasattr(mps_work, "singular_values"):
        raise RuntimeError("Tenax finite MPS object does not expose singular_values.")

    n_sites = int(mps_work.n_nodes()) if hasattr(mps_work, "n_nodes") else int(len(getattr(mps_work, "tensors", [])))
    n_bonds = max(n_sites - 1, 0)
    entropies = {f"S{n}": [] for n in orders}
    cuts = [float(idx + 1) for idx in range(n_bonds)]

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=n_bonds,
        desc="DMRG entropies",
        unit="bond",
        leave=False,
    )
    for bond in range(n_bonds):
        singular_values = mps_work.singular_values[bond]
        if singular_values is None:
            raise RuntimeError(f"Missing singular values on bond {bond}.")
        entropy_values = _entropy_dict_from_singular_values(np.asarray(singular_values), orders)
        for key, value in entropy_values.items():
            entropies[key].append(value)
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()

    return _build_entropy_profile(
        method_label="DMRG",
        cuts=cuts,
        total_span=float(max(n_sites, 1)),
        entropies=entropies,
        context={"backend": "tenax", "n_sites": n_sites},
    )


def compute_tenpy_finite_mps_entropy_profile(
    psi: Any,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    show_progress: bool = True,
) -> Dict[str, Any]:
    if not hasattr(psi, "L") or not hasattr(psi, "entanglement_entropy"):
        raise RuntimeError("TeNPy MPS object does not expose required entropy methods.")

    n_sites = int(psi.L)
    bonds = list(range(1, n_sites))
    entropies = {f"S{n}": [] for n in orders}

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(orders),
        desc="DMRG entropies",
        unit="order",
        leave=False,
    )
    for order_n in orders:
        values = psi.entanglement_entropy(n=order_n, bonds=bonds)
        entropies[f"S{order_n}"] = [float(value) for value in values]
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()

    return _build_entropy_profile(
        method_label="DMRG",
        cuts=[float(bond) for bond in bonds],
        total_span=float(max(n_sites, 1)),
        entropies=entropies,
        context={"backend": "tenpy", "n_sites": n_sites},
    )


def compute_ed_entropy_profile_from_state(
    state: np.ndarray,
    n_sites: int,
    local_dim: int,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    show_progress: bool = True,
) -> Dict[str, Any]:
    state_vec = np.asarray(state).reshape(-1)
    entropies = {f"S{n}": [] for n in orders}
    cuts = [float(cut) for cut in range(1, n_sites)]

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=max(n_sites - 1, 0),
        desc="ED entropies",
        unit="cut",
        leave=False,
    )
    for cut in range(1, n_sites):
        left_dim = int(local_dim ** cut)
        right_dim = int(local_dim ** (n_sites - cut))
        psi_matrix = state_vec.reshape((left_dim, right_dim))
        singular_values = np.linalg.svd(psi_matrix, full_matrices=False, compute_uv=False)
        entropy_values = _entropy_dict_from_singular_values(singular_values, orders)
        for key, value in entropy_values.items():
            entropies[key].append(value)
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()

    return _build_entropy_profile(
        method_label="ED",
        cuts=cuts,
        total_span=float(max(n_sites, 1)),
        entropies=entropies,
        context={"n_sites": int(n_sites), "local_dim": int(local_dim)},
    )


def compute_tenax_infinite_mps_entropy_profile(
    mps: Any,
    sites_per_idmrg_site: int,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
) -> Dict[str, Any]:
    if not hasattr(mps, "singular_values") or not hasattr(mps, "unit_cell_size"):
        raise RuntimeError("Tenax infinite MPS object does not expose singular_values/unit_cell_size.")

    unit_cell_size = int(mps.unit_cell_size)
    if unit_cell_size <= 0:
        raise RuntimeError("Invalid iDMRG unit cell size.")

    entropies = {f"S{n}": [] for n in orders}
    cuts = [float(bond + 1) for bond in range(unit_cell_size)]
    for bond in range(unit_cell_size):
        singular_values = np.asarray(mps.singular_values[bond])
        entropy_values = _entropy_dict_from_singular_values(singular_values, orders)
        for key, value in entropy_values.items():
            entropies[key].append(value)

    return _build_entropy_profile(
        method_label="iDMRG-x",
        cuts=cuts,
        total_span=float(max(unit_cell_size, 1)),
        entropies=entropies,
        context={
            "unit_cell_size": unit_cell_size,
            "sites_per_idmrg_site": int(sites_per_idmrg_site),
        },
    )


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(np.real(value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def build_zero_temperature_dmrg_reference(
    geometry: Any,
    dmrg_energy: float,
    scalar_correlations: Dict[str, np.ndarray],
    bond_rows: List[Dict[str, Any]],
    structure_factor_rows: List[Dict[str, Any]],
    uniform_observables: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Package finite-DMRG ground-state data as T=0 references for thermal plots."""
    n_sites = int(getattr(geometry, "number_of_sites", 0))
    n_sites_safe = max(1, n_sites)
    bond_list = list(getattr(geometry, "bond_list", []))

    correlations: Dict[str, float] = {"T": 0.0}
    for channel_key, output_key in (
        ("S", "nearest_neighbor_S"),
        ("T", "nearest_neighbor_T"),
        ("ST", "nearest_neighbor_ST"),
    ):
        matrix = scalar_correlations.get(channel_key)
        if matrix is None or len(bond_list) == 0:
            continue
        matrix_array = np.asarray(matrix)
        values = []
        for bond in bond_list:
            value = _finite_float_or_none(matrix_array[int(bond.i), int(bond.j)])
            if value is not None:
                values.append(value)
        if values:
            correlations[output_key] = float(np.mean(np.asarray(values, dtype=float)))

    bond_energy_values = [
        _finite_float_or_none(row.get("O_ij_gamma"))
        for row in bond_rows
        if isinstance(row, dict)
    ]
    bond_energy_values = [value for value in bond_energy_values if value is not None]
    if bond_energy_values:
        correlations["bond_energy_per_site"] = float(np.sum(bond_energy_values) / float(n_sites_safe))

    observables: Dict[str, float] = {
        "T": 0.0,
        "ground_state_energy": float(dmrg_energy),
        "energy_per_site": float(dmrg_energy) / float(n_sites_safe),
    }
    for key in ("spin_z_per_site", "orbital_z_per_site"):
        if uniform_observables and key in uniform_observables:
            value = _finite_float_or_none(uniform_observables.get(key))
            if value is not None:
                observables[key] = value

    structure_rows: List[Dict[str, Any]] = []
    for row in structure_factor_rows:
        if not isinstance(row, dict):
            continue
        ref_row: Dict[str, Any] = {"T": 0.0}
        for key in ("Q_label", "Qx", "Qy", "S(Q)", "T(Q)", "ST(Q)"):
            if key not in row:
                continue
            if key == "Q_label":
                ref_row[key] = row[key]
                continue
            value = _finite_float_or_none(row[key])
            if value is not None:
                ref_row[key] = value
        structure_rows.append(ref_row)

    return {
        "method": "DMRG",
        "temperature": 0.0,
        "note": (
            "Finite-DMRG ground-state reference shown on finite-temperature ED plots; "
            "this is not a finite-temperature DMRG calculation."
        ),
        "observables": observables,
        "correlations": correlations,
        "structure_factors": structure_rows,
    }


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Phase-scan analysis
# ----------------------------------------------------------------------

DEFAULT_PHASE_CLASSIFIER_THRESHOLDS = {
    "quantum_weak_order": 0.035,
    "classical_weak_order": 0.075,
    "quantum_bond_nematicity": 0.10,
    "classical_bond_nematicity": 0.08,
}


def phase_classifier_thresholds_from_args(args: Any) -> Dict[str, float]:
    """Read classifier thresholds from CLI args when present, otherwise defaults."""
    return {
        "quantum_weak_order": float(
            getattr(args, "phase_scan_quantum_weak_order_threshold", DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["quantum_weak_order"])
        ),
        "classical_weak_order": float(
            getattr(args, "phase_scan_classical_weak_order_threshold", DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["classical_weak_order"])
        ),
        "quantum_bond_nematicity": float(
            getattr(
                args,
                "phase_scan_quantum_nematicity_threshold",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["quantum_bond_nematicity"],
            )
        ),
        "classical_bond_nematicity": float(
            getattr(
                args,
                "phase_scan_classical_nematicity_threshold",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["classical_bond_nematicity"],
            )
        ),
    }


def _phase_scan_axis_values(axis_min: float, axis_max: float, points: int) -> List[float]:
    count = int(points)
    if count < 1:
        raise ValueError("Phase-scan grid point counts must be at least 1.")
    if count == 1:
        return [float(axis_min)]
    if float(axis_max) < float(axis_min):
        raise ValueError("Phase-scan axis max must be >= min.")
    return [float(value) for value in np.linspace(float(axis_min), float(axis_max), count)]


def _phase_scan_grid_from_args(args: Any) -> Tuple[List[float], List[float]]:
    return (
        _phase_scan_axis_values(args.phase_scan_alpha_min, args.phase_scan_alpha_max, args.phase_scan_alpha_points),
        _phase_scan_axis_values(args.phase_scan_beta_min, args.phase_scan_beta_max, args.phase_scan_beta_points),
    )


def _random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(int(count), 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms <= 1e-15] = 1.0
    return vectors / norms


def _classical_operator_value(op_name: str, site: int, spin_vectors: np.ndarray, orbital_vectors: np.ndarray) -> float:
    axis_map = {"x": 0, "y": 1, "z": 2}
    text = str(op_name)
    if len(text) < 2:
        raise ValueError(f"Unsupported classical operator '{op_name}'.")
    axis = text[-1].lower()
    if axis not in axis_map:
        raise ValueError(f"Unsupported classical operator '{op_name}'.")
    idx = axis_map[axis]
    if text.startswith("S") and not text.startswith("ST"):
        return float(spin_vectors[site, idx])
    if text.startswith("T"):
        return float(orbital_vectors[site, idx])
    if text.startswith("ST"):
        return float(spin_vectors[site, idx] * orbital_vectors[site, idx])
    raise ValueError(f"Unsupported classical operator '{op_name}'.")


def _classical_bond_energy_value(
    bond: Any,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
) -> float:
    from models import model_terms_for_bond

    value = 0.0
    for coefficient, op_name in model_terms_for_bond(
        bond.gamma,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
    ):
        left = _classical_operator_value(op_name, bond.i, spin_vectors, orbital_vectors)
        right = _classical_operator_value(op_name, bond.j, spin_vectors, orbital_vectors)
        value += float(coefficient) * left * right
    return float(value)


def _classical_total_energy(
    geometry: Any,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
    external_field_terms: List[Tuple[float, str]] | None = None,
) -> float:
    bond_energy = sum(
        _classical_bond_energy_value(
            bond,
            spin_vectors,
            orbital_vectors,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx,
            jy,
            jz,
        )
        for bond in geometry.bond_list
    )
    field_energy = 0.0
    if external_field_terms:
        for site in range(int(spin_vectors.shape[0])):
            for coefficient, op_name in external_field_terms:
                field_energy += float(coefficient) * _classical_operator_value(
                    op_name,
                    site,
                    spin_vectors,
                    orbital_vectors,
                )
    return float(bond_energy + field_energy)


def _classical_bond_rows(
    geometry: Any,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
) -> List[Dict[str, Any]]:
    return [
        {
            "i": int(bond.i),
            "j": int(bond.j),
            "gamma": str(bond.gamma),
            "O_ij_gamma": _classical_bond_energy_value(
                bond,
                spin_vectors,
                orbital_vectors,
                model_spec,
                alpha,
                beta,
                coupling_j,
                jx,
                jy,
                jz,
            ),
        }
        for bond in geometry.bond_list
    ]


def _classical_vector_structure_factor(
    vectors: np.ndarray,
    q: np.ndarray,
    positions: np.ndarray,
) -> float:
    phases = np.exp(1.0j * np.dot(np.asarray(positions, dtype=float), np.asarray(q, dtype=float)))
    amplitude = np.sum(vectors * phases[:, None], axis=0)
    return float(np.real(np.vdot(amplitude, amplitude)) / float(max(1, vectors.shape[0])))


def _classical_structure_factor_rows(
    geometry: Any,
    lattice: str,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
) -> List[Dict[str, Any]]:
    from models import default_high_symmetry_momenta

    st_vectors = spin_vectors * orbital_vectors
    rows: List[Dict[str, Any]] = []
    for label, q in default_high_symmetry_momenta(lattice).items():
        rows.append(
            {
                "Q_label": str(label),
                "Qx": float(q[0]),
                "Qy": float(q[1]),
                "S(Q)": _classical_vector_structure_factor(spin_vectors, q, geometry.positions),
                "T(Q)": _classical_vector_structure_factor(orbital_vectors, q, geometry.positions),
                "ST(Q)": _classical_vector_structure_factor(st_vectors, q, geometry.positions),
            }
        )
    return rows


def _run_classical_product_ground_state(
    geometry: Any,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
    *,
    restarts: int,
    sweeps: int,
    initial_temperature: float,
    final_temperature: float,
    initial_step: float,
    final_step: float,
    seed: int,
    external_field_terms: List[Tuple[float, str]] | None = None,
) -> Dict[str, Any]:
    n_sites = int(geometry.number_of_sites)
    restart_count = max(1, int(restarts))
    sweep_count = max(1, int(sweeps))
    spin_length = max(float(getattr(model_spec, "spin_value", 1.0)), 0.0)
    orbital_length = max(float(getattr(model_spec, "orbital_value", 1.0)), 0.0)
    best_energy = float("inf")
    best_spin = None
    best_orbital = None
    rng_master = np.random.default_rng(int(seed))

    for _ in range(restart_count):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        spin_vectors = spin_length * _random_unit_vectors(rng, n_sites)
        orbital_vectors = orbital_length * _random_unit_vectors(rng, n_sites)
        current_energy = _classical_total_energy(
            geometry,
            spin_vectors,
            orbital_vectors,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx,
            jy,
            jz,
            external_field_terms,
        )
        local_best_energy = current_energy
        local_best_spin = spin_vectors.copy()
        local_best_orbital = orbital_vectors.copy()

        for sweep in range(sweep_count):
            frac = float(sweep) / float(max(1, sweep_count - 1))
            temperature = float(initial_temperature) * (
                float(final_temperature) / max(float(initial_temperature), 1e-15)
            ) ** frac
            step_size = float(initial_step) * (
                float(final_step) / max(float(initial_step), 1e-15)
            ) ** frac
            for site in rng.permutation(n_sites):
                for target in ("spin", "orbital"):
                    proposed_spin = spin_vectors.copy()
                    proposed_orbital = orbital_vectors.copy()
                    if target == "spin":
                        if spin_length <= 1e-15:
                            continue
                        random_vector = spin_length * _random_unit_vectors(rng, 1)[0]
                        proposal = (1.0 - step_size) * spin_vectors[site] + step_size * random_vector
                        proposed_spin[site] = spin_length * proposal / max(float(np.linalg.norm(proposal)), 1e-15)
                    else:
                        if orbital_length <= 1e-15:
                            continue
                        random_vector = orbital_length * _random_unit_vectors(rng, 1)[0]
                        proposal = (1.0 - step_size) * orbital_vectors[site] + step_size * random_vector
                        proposed_orbital[site] = orbital_length * proposal / max(float(np.linalg.norm(proposal)), 1e-15)
                    proposed_energy = _classical_total_energy(
                        geometry,
                        proposed_spin,
                        proposed_orbital,
                        model_spec,
                        alpha,
                        beta,
                        coupling_j,
                        jx,
                        jy,
                        jz,
                        external_field_terms,
                    )
                    delta = proposed_energy - current_energy
                    accept = delta <= 0.0
                    if (not accept) and temperature > 0.0:
                        accept = rng.random() < float(np.exp(-delta / max(temperature, 1e-15)))
                    if accept:
                        spin_vectors = proposed_spin
                        orbital_vectors = proposed_orbital
                        current_energy = proposed_energy
                        if current_energy < local_best_energy:
                            local_best_energy = current_energy
                            local_best_spin = spin_vectors.copy()
                            local_best_orbital = orbital_vectors.copy()

        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_spin = local_best_spin
            best_orbital = local_best_orbital

    if best_spin is None or best_orbital is None:
        raise RuntimeError("Classical product minimization did not produce a state.")
    return {
        "energy": float(best_energy),
        "energy_per_site": float(best_energy / float(max(1, n_sites))),
        "spin_vectors": best_spin,
        "orbital_vectors": best_orbital,
    }


def _dominant_structure_channel(structure_rows: List[Dict[str, Any]], channel: str) -> Dict[str, Any]:
    if len(structure_rows) == 0:
        return {"label": "none", "value": 0.0, "family": "none"}
    values = [(str(row["Q_label"]), float(row.get(channel, 0.0))) for row in structure_rows]
    label, value = max(values, key=lambda item: item[1])
    if label.startswith("M"):
        family = "stripy"
    elif label.startswith("K"):
        family = "afm"
    elif label == "Gamma":
        family = "uniform"
    else:
        family = "other"
    return {"label": label, "value": float(value), "family": family}


def _bond_energy_diagnostics(bond_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(bond_rows) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "range": 0.0,
            "nematicity": 0.0,
            "lowest_energy_gamma": "none",
            "by_gamma": {},
        }
    values = np.asarray([float(row["O_ij_gamma"]) for row in bond_rows], dtype=float)
    by_gamma: Dict[str, Dict[str, float]] = {}
    for gamma in sorted({str(row["gamma"]) for row in bond_rows}):
        gamma_values = np.asarray(
            [float(row["O_ij_gamma"]) for row in bond_rows if str(row["gamma"]) == gamma],
            dtype=float,
        )
        by_gamma[gamma] = {
            "mean": float(np.mean(gamma_values)),
            "std": float(np.std(gamma_values)),
            "min": float(np.min(gamma_values)),
            "max": float(np.max(gamma_values)),
        }
    gamma_means = {gamma: item["mean"] for gamma, item in by_gamma.items()}
    lowest_gamma = min(gamma_means.items(), key=lambda item: item[1])[0] if gamma_means else "none"
    mean_range = float(max(gamma_means.values()) - min(gamma_means.values())) if gamma_means else 0.0
    scale = max(float(np.max(np.abs(values))), abs(float(np.mean(values))), 1e-12)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "range": float(np.max(values) - np.min(values)),
        "nematicity": float(mean_range / scale),
        "lowest_energy_gamma": str(lowest_gamma),
        "by_gamma": by_gamma,
    }


def _phase_observable_diagnostics(
    structure_rows: List[Dict[str, Any]],
    bond_rows: List[Dict[str, Any]],
    n_sites: int,
) -> Dict[str, Any]:
    spin_peak = _dominant_structure_channel(structure_rows, "S(Q)")
    orbital_peak = _dominant_structure_channel(structure_rows, "T(Q)")
    mixed_peak = _dominant_structure_channel(structure_rows, "ST(Q)")
    bond_diag = _bond_energy_diagnostics(bond_rows)
    norm = float(max(1, n_sites))
    return {
        "spin_peak": spin_peak,
        "orbital_peak": orbital_peak,
        "mixed_peak": mixed_peak,
        "spin_order_strength": float(spin_peak["value"] / norm),
        "orbital_order_strength": float(orbital_peak["value"] / norm),
        "mixed_order_strength": float(mixed_peak["value"] / norm),
        "bond_energy": bond_diag,
    }


def _classify_phase_from_diagnostics(
    diagnostics: Dict[str, Any],
    alpha: float,
    beta: float,
    diagram_kind: str,
    thresholds: Dict[str, float],
) -> str:
    spin_family = str(diagnostics.get("spin_peak", {}).get("family", "none"))
    orbital_family = str(diagnostics.get("orbital_peak", {}).get("family", "none"))
    spin_strength = float(diagnostics.get("spin_order_strength", 0.0))
    orbital_strength = float(diagnostics.get("orbital_order_strength", 0.0))
    nematicity = float(diagnostics.get("bond_energy", {}).get("nematicity", 0.0))

    ordered = max(spin_strength, orbital_strength)
    weak_order = ordered < (
        thresholds["quantum_weak_order"]
        if diagram_kind == "quantum_ed"
        else thresholds["classical_weak_order"]
    )
    nematic = nematicity > (
        thresholds["quantum_bond_nematicity"]
        if diagram_kind == "quantum_ed"
        else thresholds["classical_bond_nematicity"]
    )

    if diagram_kind == "quantum_ed":
        if alpha <= 0.08 and beta >= 0.045 and weak_order:
            return "Spin liquid"
        if alpha <= 0.20 and beta >= 0.06 and nematic:
            return "NP3"
        if nematic or weak_order:
            return "NP1"
    else:
        if alpha <= 0.035 and beta >= 0.02 and weak_order:
            return "Spin liquid"
        if alpha >= 2.0 and beta <= 0.035 and nematic:
            return "NP2"
        if beta <= 0.035 and (nematic or weak_order):
            return "NP1"

    if spin_family == "stripy" or orbital_family == "stripy":
        return "Stripy S / AFO"
    if spin_family in ("afm", "uniform") or orbital_family in ("afm", "uniform"):
        return "AFM / AFO"
    if nematic:
        return "NP1"
    return "Weak/undetermined"


def _phase_scan_quantum_point(
    geometry: Any,
    model_spec: Any,
    lattice_name: str,
    alpha: float,
    beta: float,
    args: Any,
    hamiltonian_external_field_terms: List[Tuple[float, str]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    from models import (
        all_bond_energies,
        all_high_symmetry_structure_factors,
        build_spin_orbital_scalar_correlations,
        collect_correlation_matrices_from_ed,
        run_small_cluster_exact_diagonalization,
    )

    local_dim = int(model_spec.physical_dim)
    hilbert_dim = int(local_dim ** int(geometry.number_of_sites))
    if int(geometry.number_of_sites) > int(args.phase_scan_ed_max_sites):
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
        }
    if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": (
                f"Quantum phase scan Hilbert dimension {hilbert_dim} exceeds "
                f"{int(args.phase_scan_ed_max_hilbert_dim)}."
            ),
        }
    energy, state = run_small_cluster_exact_diagonalization(
        geometry=geometry,
        model_spec=model_spec,
        alpha=alpha,
        beta=beta,
        coupling_j=args.coupling_j,
        jx=args.jx,
        jy=args.jy,
        jz=args.jz,
        external_field_terms=hamiltonian_external_field_terms,
        show_progress=False,
    )
    correlations = collect_correlation_matrices_from_ed(
        geometry,
        state,
        model_spec=model_spec,
        show_progress=False,
    )
    scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
    structure_rows = all_high_symmetry_structure_factors(
        scalar_correlations,
        geometry,
        lattice=lattice_name,
        show_progress=False,
    )
    bond_rows = all_bond_energies(
        geometry,
        correlations,
        model_spec,
        alpha,
        beta,
        args.coupling_j,
        jx=args.jx,
        jy=args.jy,
        jz=args.jz,
        show_progress=False,
    )
    diagnostics = _phase_observable_diagnostics(structure_rows, bond_rows, geometry.number_of_sites)
    phase_label = _classify_phase_from_diagnostics(diagnostics, alpha, beta, "quantum_ed", thresholds)
    return {
        "status": "completed",
        "alpha": float(alpha),
        "beta": float(beta),
        "phase_label": phase_label,
        "energy": float(energy),
        "energy_per_site": float(energy / float(max(1, geometry.number_of_sites))),
        "diagnostics": diagnostics,
        "structure_factors": structure_rows,
        "bond_energies": bond_rows,
    }


def _phase_scan_classical_point(
    geometry: Any,
    model_spec: Any,
    lattice_name: str,
    alpha: float,
    beta: float,
    args: Any,
    point_index: int,
    hamiltonian_external_field_terms: List[Tuple[float, str]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    result = _run_classical_product_ground_state(
        geometry=geometry,
        model_spec=model_spec,
        alpha=alpha,
        beta=beta,
        coupling_j=args.coupling_j,
        jx=args.jx,
        jy=args.jy,
        jz=args.jz,
        restarts=args.phase_scan_classical_restarts,
        sweeps=args.phase_scan_classical_sweeps,
        initial_temperature=args.phase_scan_classical_initial_temperature,
        final_temperature=args.phase_scan_classical_final_temperature,
        initial_step=args.phase_scan_classical_initial_step,
        final_step=args.phase_scan_classical_final_step,
        seed=int(args.phase_scan_random_seed) + int(point_index),
        external_field_terms=hamiltonian_external_field_terms,
    )
    spin_vectors = np.asarray(result["spin_vectors"], dtype=float)
    orbital_vectors = np.asarray(result["orbital_vectors"], dtype=float)
    structure_rows = _classical_structure_factor_rows(geometry, lattice_name, spin_vectors, orbital_vectors)
    bond_rows = _classical_bond_rows(
        geometry,
        spin_vectors,
        orbital_vectors,
        model_spec,
        alpha,
        beta,
        args.coupling_j,
        args.jx,
        args.jy,
        args.jz,
    )
    diagnostics = _phase_observable_diagnostics(structure_rows, bond_rows, geometry.number_of_sites)
    phase_label = _classify_phase_from_diagnostics(diagnostics, alpha, beta, "classical_product", thresholds)
    return {
        "status": "completed",
        "alpha": float(alpha),
        "beta": float(beta),
        "phase_label": phase_label,
        "energy": float(result["energy"]),
        "energy_per_site": float(result["energy_per_site"]),
        "diagnostics": diagnostics,
        "structure_factors": structure_rows,
        "bond_energies": bond_rows,
        "spin_vectors": spin_vectors.tolist(),
        "orbital_vectors": orbital_vectors.tolist(),
    }


def run_alpha_beta_phase_scan(
    geometry: Any,
    model_spec: Any,
    lattice_name: str,
    args: Any,
    hamiltonian_external_field_terms: List[Tuple[float, str]],
    show_progress: bool,
) -> Dict[str, Any]:
    alphas, betas = _phase_scan_grid_from_args(args)
    thresholds = phase_classifier_thresholds_from_args(args)
    modes = ["quantum_ed", "classical_product"] if args.phase_scan_mode == "both" else [args.phase_scan_mode]
    total_points = len(alphas) * len(betas)
    output: Dict[str, Any] = {
        "status": "completed",
        "mode": str(args.phase_scan_mode),
        "grid": {
            "alpha_min": float(min(alphas)),
            "alpha_max": float(max(alphas)),
            "alpha_points": int(len(alphas)),
            "alpha_values": alphas,
            "beta_min": float(min(betas)),
            "beta_max": float(max(betas)),
            "beta_points": int(len(betas)),
            "beta_values": betas,
        },
        "solver_controls": {
            "quantum_ed_max_sites": int(args.phase_scan_ed_max_sites),
            "quantum_ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
            "classical_restarts": int(args.phase_scan_classical_restarts),
            "classical_sweeps": int(args.phase_scan_classical_sweeps),
            "classical_initial_temperature": float(args.phase_scan_classical_initial_temperature),
            "classical_final_temperature": float(args.phase_scan_classical_final_temperature),
            "classical_initial_step": float(args.phase_scan_classical_initial_step),
            "classical_final_step": float(args.phase_scan_classical_final_step),
            "random_seed": int(args.phase_scan_random_seed),
        },
        "classifier_thresholds": thresholds,
        "classifier_note": (
            "Phase labels are finite-size, observable-based assignments from dominant "
            "spin/orbital structure factors plus bond-energy nematicity. They are saved "
            "with diagnostics for reproducibility and should be checked against larger "
            "clusters or denser grids before quoting thermodynamic boundaries."
        ),
    }
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=total_points * len(modes),
        desc="phase scan",
        unit="point",
        leave=False,
    )
    point_index = 0
    for mode in modes:
        rows: List[Dict[str, Any]] = []
        for beta in betas:
            for alpha in alphas:
                try:
                    if mode == "quantum_ed":
                        row = _phase_scan_quantum_point(
                            geometry,
                            model_spec,
                            lattice_name,
                            alpha,
                            beta,
                            args,
                            hamiltonian_external_field_terms,
                            thresholds,
                        )
                    else:
                        row = _phase_scan_classical_point(
                            geometry,
                            model_spec,
                            lattice_name,
                            alpha,
                            beta,
                            args,
                            point_index,
                            hamiltonian_external_field_terms,
                            thresholds,
                        )
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "error": str(exc),
                    }
                rows.append(row)
                point_index += 1
                if progress_bar is not None:
                    progress_bar.update(1)
        failed_count = int(sum(1 for row in rows if row.get("status") == "failed"))
        output[mode] = {
            "status": "completed_with_warnings" if failed_count > 0 else "completed",
            "rows": rows,
            "completed_points": int(sum(1 for row in rows if row.get("status") == "completed")),
            "failed_points": failed_count,
            "skipped_points": int(sum(1 for row in rows if row.get("status") == "skipped")),
        }
    if progress_bar is not None:
        progress_bar.close()
    failed_points = sum(
        int(mode_data.get("failed_points", 0))
        for mode_data in output.values()
        if isinstance(mode_data, dict)
    )
    if failed_points > 0:
        output["status"] = "completed_with_warnings"
    return output
