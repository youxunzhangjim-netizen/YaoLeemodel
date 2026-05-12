#!/usr/bin/env python3
"""Analysis and runtime helpers for the Yao-Lee driver.

This file owns lazy Tenax imports, progress bars, stage timing, entropy-profile
analysis, and alpha-beta phase-scan analysis/classification. Model
construction stays in ``models.py`` and plot rendering stays in
``plot_outputs.py``.
"""

from __future__ import annotations

import time
import math
import importlib.util
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


def _geometry_positions_array(geometry: Any) -> np.ndarray:
    if hasattr(geometry, "positions"):
        return np.asarray(geometry.positions, dtype=float)
    if hasattr(geometry, "coordinates"):
        return np.asarray(geometry.coordinates, dtype=float)
    raise AttributeError("Geometry object must provide positions or coordinates.")


def select_geometric_center_site(
    geometry: Any,
    reference_site_idx: int | None = None,
) -> int:
    """Choose the site closest to the geometric center of the finite cluster."""
    n_sites = int(getattr(geometry, "number_of_sites"))
    if n_sites <= 0:
        raise ValueError("Cannot select a reference site for an empty geometry.")
    if reference_site_idx is not None:
        site = int(reference_site_idx)
        if site < 0:
            site = n_sites + site
        if site < 0 or site >= n_sites:
            raise IndexError(f"reference_site_idx={reference_site_idx} is outside [0, {n_sites - 1}].")
        return site

    positions = _geometry_positions_array(geometry)
    if positions.shape[0] != n_sites:
        raise ValueError(
            f"Geometry has number_of_sites={n_sites}, but positions has length {positions.shape[0]}."
        )
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center[None, :], axis=1)
    return int(np.argmin(distances))


def find_reference_site_idx(
    geometry: Any,
    reference_site_idx: int | None = None,
) -> int:
    """Return the requested reference site, or the site nearest the cluster center."""
    return select_geometric_center_site(geometry, reference_site_idx)


def extract_spin_reference_correlation(
    correlations: Dict[str, np.ndarray],
    reference_site_idx: int,
) -> np.ndarray:
    """Extract C_S[j] = <S_ref . S_j> from component correlation matrices."""
    component_keys = ("Sx_Sx", "Sy_Sy", "Sz_Sz")
    missing = [key for key in component_keys if key not in correlations]
    if missing:
        raise KeyError(f"correlations is missing spin component matrices: {', '.join(missing)}.")

    matrices = [
        np.asarray(correlations[key], dtype=np.complex128)
        for key in component_keys
    ]
    first_shape = matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError("Spin correlation component matrices must be square.")
    for key, matrix in zip(component_keys[1:], matrices[1:]):
        if matrix.shape != first_shape:
            raise ValueError(
                f"Spin correlation component '{key}' has shape {matrix.shape}, "
                f"but expected {first_shape}."
            )

    n_sites = int(first_shape[0])
    ref_site = int(reference_site_idx)
    if ref_site < 0:
        ref_site = n_sites + ref_site
    if ref_site < 0 or ref_site >= n_sites:
        raise IndexError(f"reference_site_idx={reference_site_idx} is outside [0, {n_sites - 1}].")

    spin_scalar = matrices[0] + matrices[1] + matrices[2]
    return np.real_if_close(spin_scalar[ref_site, :]).astype(float)


def build_spin_reference_correlation_pattern(
    geometry: Any,
    correlations: Dict[str, np.ndarray],
    reference_site_idx: int | None = None,
) -> Dict[str, Any]:
    """Build the real-space spin reference-site pattern from raw correlations."""
    ref_site = find_reference_site_idx(geometry, reference_site_idx)
    positions = _geometry_positions_array(geometry)
    c_s = extract_spin_reference_correlation(correlations, ref_site)
    return {
        "reference_site_idx": int(ref_site),
        "reference_position": [float(value) for value in positions[ref_site]],
        "C_S": c_s,
        "correlations": {"S": c_s},
        "max_abs_correlation": {"S": float(np.max(np.abs(c_s))) if c_s.size > 0 else 0.0},
        "method": "reference_site_correlation",
        "definition": "C_S[j] = <S_ref . S_j> with ref chosen near the geometric center",
    }


def _scalar_correlation_matrix(
    scalar_correlations: Dict[str, np.ndarray],
    key: str,
    aliases: Sequence[str],
) -> np.ndarray:
    for candidate in (key, *aliases):
        if candidate in scalar_correlations:
            matrix = np.asarray(scalar_correlations[candidate], dtype=np.complex128)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Scalar correlation '{candidate}' must be a square matrix.")
            return matrix
    alias_text = ", ".join((key, *aliases))
    raise KeyError(f"scalar_correlations must contain one of: {alias_text}.")


def build_reference_site_correlation_patterns(
    geometry: Any,
    scalar_correlations: Dict[str, np.ndarray],
    reference_site_idx: int | None = None,
) -> Dict[str, Any]:
    """Extract real-space pattern rows C[j] = <O_ref O_j>.

    Finite ED/DMRG eigenstates often keep the exact symmetry, so one-point
    order parameters can vanish even in ordered phases. A row of the two-point
    scalar correlation matrix reveals the relative ordering pattern without
    forcing spontaneous symmetry breaking.
    """
    ref_site = select_geometric_center_site(geometry, reference_site_idx)
    positions = _geometry_positions_array(geometry)
    n_sites = int(getattr(geometry, "number_of_sites"))
    if positions.shape[0] != n_sites:
        raise ValueError(
            f"Geometry has number_of_sites={n_sites}, but positions has length {positions.shape[0]}."
        )

    spin_matrix = _scalar_correlation_matrix(scalar_correlations, "S", ("spin_scalar",))
    orbital_matrix = _scalar_correlation_matrix(scalar_correlations, "T", ("orbital_scalar",))
    mixed_matrix: np.ndarray | None = None
    try:
        mixed_matrix = _scalar_correlation_matrix(scalar_correlations, "ST", ("mixed_scalar",))
    except KeyError:
        mixed_matrix = None

    for name, matrix in (("S", spin_matrix), ("T", orbital_matrix)):
        if matrix.shape[0] != n_sites:
            raise ValueError(
                f"Scalar correlation '{name}' has shape {matrix.shape}, but geometry has {n_sites} sites."
            )
    if mixed_matrix is not None and mixed_matrix.shape[0] != n_sites:
        raise ValueError(
            f"Scalar correlation 'ST' has shape {mixed_matrix.shape}, but geometry has {n_sites} sites."
        )

    correlations: Dict[str, np.ndarray] = {
        "S": np.real_if_close(spin_matrix[ref_site, :]).astype(float),
        "T": np.real_if_close(orbital_matrix[ref_site, :]).astype(float),
    }
    if mixed_matrix is not None:
        correlations["ST"] = np.real_if_close(mixed_matrix[ref_site, :]).astype(float)

    max_abs = {
        key: float(np.max(np.abs(values))) if values.size > 0 else 0.0
        for key, values in correlations.items()
    }
    return {
        "reference_site_idx": int(ref_site),
        "reference_position": [float(value) for value in positions[ref_site]],
        "correlations": correlations,
        "max_abs_correlation": max_abs,
        "method": "reference_site_correlation",
        "definition": "C_O[j] = <O_ref O_j> with ref chosen near the geometric center",
    }


def _classical_scalar_correlations(
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build scalar S, T, and ST correlation matrices for a classical product state."""
    spin = np.asarray(spin_vectors, dtype=float)
    orbital = np.asarray(orbital_vectors, dtype=float)
    if spin.ndim != 2 or orbital.ndim != 2 or spin.shape != orbital.shape:
        raise ValueError("Classical spin_vectors and orbital_vectors must have matching shape (N, 3).")
    spin_scalar = spin @ spin.T
    orbital_scalar = orbital @ orbital.T
    return {
        "S": spin_scalar,
        "T": orbital_scalar,
        "ST": spin_scalar * orbital_scalar,
    }


def _reference_patterns_or_warning(
    geometry: Any,
    scalar_correlations: Dict[str, np.ndarray],
    reference_site_idx: int | None,
) -> Dict[str, Any]:
    """Return reference-site patterns without failing the whole phase-scan point."""
    try:
        return build_reference_site_correlation_patterns(
            geometry,
            scalar_correlations,
            reference_site_idx=reference_site_idx,
        )
    except Exception as exc:
        return {
            "available": False,
            "warning": f"Failed to extract reference-site patterns: {exc}",
        }


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
    finite_n_sites: int | None = None,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
) -> Dict[str, Any]:
    if not hasattr(mps, "singular_values") or not hasattr(mps, "unit_cell_size"):
        raise RuntimeError("Tenax infinite MPS object does not expose singular_values/unit_cell_size.")

    unit_cell_size = int(mps.unit_cell_size)
    if unit_cell_size <= 0:
        raise RuntimeError("Invalid iDMRG unit cell size.")
    original_sites_per_idmrg_site = max(int(sites_per_idmrg_site), 1)

    entropies = {f"S{n}": [] for n in orders}
    for bond in range(unit_cell_size):
        singular_values = np.asarray(mps.singular_values[bond])
        entropy_values = _entropy_dict_from_singular_values(singular_values, orders)
        for key, value in entropy_values.items():
            entropies[key].append(value)

    context: Dict[str, Any] = {
        "unit_cell_size": unit_cell_size,
        "sites_per_idmrg_site": original_sites_per_idmrg_site,
        "unit_cell_original_sites": int(unit_cell_size * original_sites_per_idmrg_site),
    }
    if finite_n_sites is not None:
        finite_site_count = int(finite_n_sites)
        finite_cuts = [
            float(cut)
            for cut in range(original_sites_per_idmrg_site, finite_site_count, original_sites_per_idmrg_site)
        ]
        if finite_site_count > 0 and len(finite_cuts) > 0:
            finite_entropies = {key: [] for key in entropies}
            for cut in finite_cuts:
                bond_index = (int(cut // original_sites_per_idmrg_site) - 1) % unit_cell_size
                for key, values in entropies.items():
                    finite_entropies[key].append(float(values[bond_index]))
            context.update(
                {
                    "finite_n_sites_for_normalized_cuts": finite_site_count,
                    "available_original_cut_spacing": original_sites_per_idmrg_site,
                    "profile_mapping": (
                        "iDMRG bond entropies are repeated over finite-chain cut positions "
                        "that fall between coarse-grained iDMRG sites."
                    ),
                }
            )
            return _build_entropy_profile(
                method_label="iDMRG-x",
                cuts=finite_cuts,
                total_span=float(finite_site_count),
                entropies=finite_entropies,
                context=context,
            )

    cuts = [float(bond + 1) for bond in range(unit_cell_size)]
    return _build_entropy_profile(
        method_label="iDMRG-x",
        cuts=cuts,
        total_span=float(max(unit_cell_size, 1)),
        entropies=entropies,
        context=context,
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
    "plaquette_flux_target": 1.0,
    "plaquette_flux_tolerance": 0.15,
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
        "plaquette_flux_target": float(
            getattr(
                args,
                "phase_scan_plaquette_flux_target",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["plaquette_flux_target"],
            )
        ),
        "plaquette_flux_tolerance": float(
            getattr(
                args,
                "phase_scan_plaquette_flux_tolerance",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["plaquette_flux_tolerance"],
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

    if (
        str(getattr(model_spec, "model_family", "")).strip().lower() == "yao_lee"
        and int(getattr(model_spec, "orbital_dim", 1)) > 1
    ):
        axis_map = {"x": 0, "y": 1, "z": 2}
        gamma = str(bond.gamma).strip().lower()
        axis_index = axis_map[gamma]
        spin_dot = float(np.dot(spin_vectors[bond.i], spin_vectors[bond.j]))
        orbital_gamma = float(orbital_vectors[bond.i, axis_index] * orbital_vectors[bond.j, axis_index])
        return float(
            float(coupling_j)
            * (
                (1.0 + float(beta)) * spin_dot
                + (1.0 - float(beta)) * orbital_gamma
                + float(alpha) * spin_dot * orbital_gamma
            )
        )

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
    from models import model_terms_for_bond

    def channel_for_operator(op_name: str) -> str:
        text = str(op_name)
        if text.startswith("ST"):
            return "ST"
        if text.startswith("T"):
            return "T"
        if text.startswith("S"):
            return "S"
        return "total"

    rows: List[Dict[str, Any]] = []
    is_yao_lee_with_orbital = (
        str(getattr(model_spec, "model_family", "")).strip().lower() == "yao_lee"
        and int(getattr(model_spec, "orbital_dim", 1)) > 1
    )
    axis_map = {"x": 0, "y": 1, "z": 2}
    for bond in geometry.bond_list:
        if is_yao_lee_with_orbital:
            gamma = str(bond.gamma).strip().lower()
            axis_index = axis_map[gamma]
            spin_dot = float(np.dot(spin_vectors[bond.i], spin_vectors[bond.j]))
            orbital_gamma = float(orbital_vectors[bond.i, axis_index] * orbital_vectors[bond.j, axis_index])
            channel_energies = {
                "S": float(coupling_j) * (1.0 + float(beta)) * spin_dot,
                "T": float(coupling_j) * (1.0 - float(beta)) * orbital_gamma,
                "ST": float(coupling_j) * float(alpha) * spin_dot * orbital_gamma,
            }
        else:
            channel_energies: Dict[str, float] = {}
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
                channel = channel_for_operator(str(op_name))
                channel_energies[channel] = channel_energies.get(channel, 0.0) + float(coefficient) * left * right
        components = [
            {"channel": channel, "energy": float(energy)}
            for channel, energy in channel_energies.items()
        ]
        rows.append(
            {
                "i": int(bond.i),
                "j": int(bond.j),
                "gamma": str(bond.gamma),
                "O_ij_gamma": float(sum(channel_energies.values())),
                "components": components,
                "channel_energies": channel_energies,
            }
        )
    return rows


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


def extract_all_plaquette_fluxes(plaquette_flux: Dict[str, Any] | None) -> Dict[str, float]:
    """Extract a JSON-friendly plaquette-index -> W_p map from flux diagnostics."""
    if not isinstance(plaquette_flux, dict):
        return {}
    for map_key in ("all_plaquette_fluxes", "plaquette_flux_map"):
        flux_map = plaquette_flux.get(map_key)
        if isinstance(flux_map, dict):
            output: Dict[str, float] = {}
            for index, value in flux_map.items():
                try:
                    flux_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(flux_value):
                    output[str(index)] = flux_value
            if output:
                return output
    plaquette_details = plaquette_flux.get("plaquettes")
    if isinstance(plaquette_details, dict):
        output = {}
        for index, detail in plaquette_details.items():
            if not isinstance(detail, dict):
                continue
            try:
                flux_value = float(detail.get("W_p", detail.get("value")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(flux_value):
                output[str(index)] = flux_value
        return output
    return {}


def _phase_observable_diagnostics(
    structure_rows: List[Dict[str, Any]],
    bond_rows: List[Dict[str, Any]],
    n_sites: int,
    plaquette_flux: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    spin_peak = _dominant_structure_channel(structure_rows, "S(Q)")
    orbital_peak = _dominant_structure_channel(structure_rows, "T(Q)")
    mixed_peak = _dominant_structure_channel(structure_rows, "ST(Q)")
    bond_diag = _bond_energy_diagnostics(bond_rows)
    norm = float(max(1, n_sites))
    diagnostics = {
        "spin_peak": spin_peak,
        "orbital_peak": orbital_peak,
        "mixed_peak": mixed_peak,
        "spin_order_strength": float(spin_peak["value"] / norm),
        "orbital_order_strength": float(orbital_peak["value"] / norm),
        "mixed_order_strength": float(mixed_peak["value"] / norm),
        "bond_energy": bond_diag,
    }
    if isinstance(plaquette_flux, dict):
        diagnostics["plaquette_flux"] = plaquette_flux
        all_fluxes = extract_all_plaquette_fluxes(plaquette_flux)
        if all_fluxes:
            diagnostics["all_plaquette_fluxes"] = all_fluxes
    return diagnostics


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
    flux_diag = diagnostics.get("plaquette_flux", {})
    is_quantum = str(diagram_kind) != "classical_product"

    ordered = max(spin_strength, orbital_strength)
    weak_order = ordered < (
        thresholds["quantum_weak_order"]
        if is_quantum
        else thresholds["classical_weak_order"]
    )
    if str(diagram_kind) in ("quantum_ed", "tenpy_dmrg", "tenpy_idmrg", "tenax", "quspin"):
        try:
            from models import plaquette_flux_close_to_target

            flux_value = (
                flux_diag.get("W_p", flux_diag.get("value"))
                if isinstance(flux_diag, dict)
                else None
            )
            flux_is_conserved = bool(
                isinstance(flux_diag, dict)
                and flux_diag.get("available", False)
                and plaquette_flux_close_to_target(
                    flux_value,
                    target=float(thresholds.get("plaquette_flux_target", 1.0)),
                    tolerance=float(thresholds.get("plaquette_flux_tolerance", 0.15)),
                )
            )
        except Exception:
            flux_is_conserved = False
        if flux_is_conserved and weak_order:
            return "Spin-Orbital Liquid"
    nematic = nematicity > (
        thresholds["quantum_bond_nematicity"]
        if is_quantum
        else thresholds["classical_bond_nematicity"]
    )

    if is_quantum:
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


def _phase_scan_symmetry_mode(args: Any) -> str:
    mode = str(getattr(args, "symmetry_mode", "none")).strip().lower()
    aliases = {"u1sz": "u1_sz", "u1-sz": "u1_sz", "u1tz": "u1_tz", "u1-tz": "u1_tz"}
    return aliases.get(mode, mode)


def _phase_scan_reductions(args: Any) -> Tuple[str, ...]:
    value = getattr(args, "symmetry_reductions", None)
    if value is None:
        mode = _phase_scan_symmetry_mode(args)
        if mode == "auto":
            return ("auto",)
        if mode == "u1":
            return ("sz", "tz")
        if mode == "u1_sz":
            return ("sz",)
        if mode == "u1_tz":
            return ("tz",)
        if mode == "z2":
            return ("z2",)
        return ("none",)
    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [
            item.strip()
            for item in str(value).replace("+", ",").replace(";", ",").split(",")
            if item.strip()
        ]
    aliases = {
        "auto": "auto",
        "none": "none",
        "u1": "u1",
        "u1_sz": "sz",
        "u1-sz": "sz",
        "u1sz": "sz",
        "sz": "sz",
        "spin": "sz",
        "u1_tz": "tz",
        "u1-tz": "tz",
        "u1tz": "tz",
        "tz": "tz",
        "tau_z": "tz",
        "z2": "z2",
        "parity": "z2",
    }
    reductions: List[str] = []
    for raw in raw_items:
        item = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
        if item == "auto":
            return ("auto",)
        if item == "none":
            return ("none",)
        if item == "u1":
            for reduction in ("sz", "tz"):
                if reduction not in reductions:
                    reductions.append(reduction)
        elif item in ("sz", "tz", "z2") and item not in reductions:
            reductions.append(item)
    return tuple(reductions or ["none"])


def _phase_scan_uses_sz_block(args: Any) -> bool:
    if hasattr(args, "use_sz_block"):
        return bool(getattr(args, "use_sz_block"))
    reductions = set(_phase_scan_reductions(args))
    return bool("auto" in reductions or "sz" in reductions)


def _phase_scan_uses_tau_z_block(args: Any) -> bool:
    if hasattr(args, "use_tau_z_block"):
        return bool(getattr(args, "use_tau_z_block"))
    reductions = set(_phase_scan_reductions(args))
    return bool("tz" in reductions)


def _sector_dimension_for_spin_half(n_sites: int, target_m2: int) -> int:
    n = int(n_sites)
    numerator = n + int(target_m2)
    if numerator % 2 != 0:
        return 0
    nup = numerator // 2
    if nup < 0 or nup > n:
        return 0
    return int(math.comb(n, nup))


def _phase_scan_spin_orbital_block_dimension(
    n_sites: int,
    use_sz_block: bool,
    target_sz2: int,
    use_tau_z_block: bool,
    target_tz2: int,
) -> int:
    n = int(n_sites)
    spin_dim = _sector_dimension_for_spin_half(n, target_sz2) if bool(use_sz_block) else (1 << n)
    orbital_dim = _sector_dimension_for_spin_half(n, target_tz2) if bool(use_tau_z_block) else (1 << n)
    return int(spin_dim * orbital_dim)


def _phase_scan_quantum_point(
    geometry: Any,
    model_spec: Any,
    lattice_name: str,
    alpha: float,
    beta: float,
    args: Any,
    hamiltonian_external_field_terms: List[Tuple[float, str]],
    thresholds: Dict[str, float],
    show_progress: bool = True,
) -> Dict[str, Any]:
    from models import all_high_symmetry_structure_factors
    from ed_backend import (
        all_bond_energies,
        build_spin_orbital_scalar_correlations,
        collect_correlation_matrices_from_ed,
        plaquette_flux_from_ed_state,
        run_small_cluster_exact_diagonalization,
    )

    local_dim = int(model_spec.physical_dim)
    n_sites = int(geometry.number_of_sites)
    full_hilbert_dim = int(local_dim ** n_sites)
    ed_backend_name = str(getattr(args, "ed_backend", "standard")).strip().lower()
    if ed_backend_name == "ed":
        ed_backend_name = "standard"
    use_sz_block = _phase_scan_uses_sz_block(args)
    use_tau_z_block = _phase_scan_uses_tau_z_block(args)
    use_z2_block = bool(getattr(args, "use_z2_block", False))
    use_translation_x_block = bool(getattr(args, "use_translation_x_block", False))
    use_translation_y_block = bool(getattr(args, "use_translation_y_block", False))
    use_translation_block = bool(use_translation_x_block or use_translation_y_block)
    use_reflection_block = bool(getattr(args, "use_reflection_block", False))
    requested_translation_block = bool(use_translation_block)
    requested_translation_x_block = bool(use_translation_x_block)
    requested_translation_y_block = bool(use_translation_y_block)
    requested_reflection_block = bool(use_reflection_block)
    reflection_block = int(getattr(args, "reflection_block", 0))
    momentum_x_block = int(getattr(args, "momentum_x_block", 0))
    momentum_y_block = int(getattr(args, "momentum_y_block", 0))
    target_sz2 = int(getattr(args, "u1_target_sz2", 0))
    target_tz2 = int(getattr(args, "u1_target_tz2", 0))
    field_ops = {
        str(op_name)
        for coefficient, op_name in list(hamiltonian_external_field_terms or [])
        if abs(float(coefficient)) > 1e-14
    }
    if ed_backend_name == "quspin":
        if bool(use_sz_block) and bool(field_ops.intersection({"Sx", "Sy"})):
            use_sz_block = False
            use_z2_block = False
        if bool(use_z2_block) and bool(field_ops.intersection({"Sx", "Sy", "Sz"})):
            use_z2_block = False
        quspin_package_available = importlib.util.find_spec("quspin") is not None
        quspin_translation_x_reason = None
        quspin_translation_y_reason = None
        if use_translation_block:
            if not quspin_package_available:
                reason = "QuSpin package is not installed, so translation blocks cannot be checked."
                use_translation_x_block = False
                use_translation_y_block = False
                quspin_translation_x_reason = reason if requested_translation_x_block else None
                quspin_translation_y_reason = reason if requested_translation_y_block else None
            else:
                try:
                    import quspin_backend as quspin_validation_backend

                    support = quspin_validation_backend.quspin_translation_block_support(geometry)
                    x_support = support.get("x", {})
                    y_support = support.get("y", {})
                    use_translation_x_block = bool(
                        requested_translation_x_block and x_support.get("supported", False)
                    )
                    use_translation_y_block = bool(
                        requested_translation_y_block and y_support.get("supported", False)
                    )
                    quspin_translation_x_reason = x_support.get("reason") if requested_translation_x_block else None
                    quspin_translation_y_reason = y_support.get("reason") if requested_translation_y_block else None
                except Exception as exc:
                    reason = str(exc)
                    use_translation_x_block = False
                    use_translation_y_block = False
                    quspin_translation_x_reason = reason if requested_translation_x_block else None
                    quspin_translation_y_reason = reason if requested_translation_y_block else None
        use_translation_block = bool(use_translation_x_block or use_translation_y_block)
        quspin_translation_reason = {
            "x": quspin_translation_x_reason,
            "y": quspin_translation_y_reason,
        }
        quspin_reflection_reason = None
        if requested_reflection_block or reflection_block != 0:
            quspin_reflection_reason = (
                "QuSpin reflection/C3 blocks are not applied for the bond-directional Yao-Lee Hamiltonian; "
                "they can permute x/y/z bond types unless a gauge map is implemented."
            )
        use_reflection_block = False
        reflection_block = 0
        compatible = (
            quspin_package_available
            and
            str(getattr(model_spec, "spin_rep", "")) == "1/2"
            and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
            and str(getattr(model_spec, "model_family", "")) == "yao_lee"
            and str(getattr(model_spec, "ising_axis", "")) == "z"
            and _phase_scan_spin_orbital_block_dimension(
                n_sites,
                use_sz_block,
                target_sz2,
                use_tau_z_block,
                target_tz2,
            ) > 0
            and (not use_z2_block or (use_sz_block and target_sz2 == 0))
            and not (use_tau_z_block and (use_z2_block or use_translation_block))
        )
        hilbert_dim = _phase_scan_spin_orbital_block_dimension(
            n_sites,
            use_sz_block,
            target_sz2,
            use_tau_z_block,
            target_tz2,
        )
        basis_type = (
            "quspin_tensor_"
            f"spin_{'u1_block' if use_sz_block else 'full'}_"
            f"orbital_{'u1_block' if use_tau_z_block else 'full'}"
        )
        pre_quspin_hilbert_dim = int(hilbert_dim)
        quspin_basis_build_reason = None
        if compatible:
            try:
                import quspin_backend as quspin_basis_backend

                preflight_basis = quspin_basis_backend.build_quspin_yao_lee_basis(
                    n_sites,
                    geometry=geometry,
                    use_sz_block=use_sz_block,
                    target_sz2=target_sz2,
                    use_tau_z_block=use_tau_z_block,
                    target_tz2=target_tz2,
                    use_z2_block=use_z2_block,
                    z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
                    use_translation_block=use_translation_block,
                    use_translation_x_block=use_translation_x_block,
                    use_translation_y_block=use_translation_y_block,
                    momentum_block_1=momentum_x_block,
                    momentum_block_2=momentum_y_block,
                    momentum_x_block=momentum_x_block,
                    momentum_y_block=momentum_y_block,
                    use_reflection_block=False,
                    reflection_block=0,
                )
                hilbert_dim = int(preflight_basis.Ns)
            except Exception as exc:
                compatible = False
                quspin_basis_build_reason = f"Failed to build the requested QuSpin reduced basis: {exc}"
        if not compatible:
            return {
                "status": "skipped",
                "alpha": float(alpha),
                "beta": float(beta),
                "reason": (
                    quspin_basis_build_reason
                    if quspin_basis_build_reason is not None
                    else
                    "QuSpin ED phase scan requires reachable shared U1 target sectors, "
                    "the quspin Python package, "
                    "spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, "
                    "and ising_axis=z."
                    " Reflection/C3 blocks are forbidden; spin-flip Z2 requires total Sz=0; "
                    "tau_z is not combined with Z2/2D translations."
                ),
                "ed_backend": "quspin",
                "basis_type": basis_type,
                "effective_hilbert_dimension": int(hilbert_dim),
                "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
                "full_hilbert_dimension": int(full_hilbert_dim),
                "use_sz_conserved": bool(use_sz_block),
                "use_sz_block": bool(use_sz_block),
                "quspin_package_available": bool(quspin_package_available),
                "quspin_requested_translation_block": bool(requested_translation_block),
                "quspin_requested_translation_x_block": bool(requested_translation_x_block),
                "quspin_requested_translation_y_block": bool(requested_translation_y_block),
                "quspin_use_translation_block": bool(use_translation_block),
                "quspin_use_translation_x_block": bool(use_translation_x_block),
                "quspin_use_translation_y_block": bool(use_translation_y_block),
                "quspin_translation_reason": quspin_translation_reason,
                "quspin_translation_x_reason": quspin_translation_x_reason,
                "quspin_translation_y_reason": quspin_translation_y_reason,
                "quspin_requested_reflection_block": bool(requested_reflection_block),
                "quspin_use_reflection_block": False,
                "quspin_reflection_reason": quspin_reflection_reason,
            }
        if n_sites > int(args.phase_scan_ed_max_sites):
            return {
                "status": "skipped",
                "alpha": float(alpha),
                "beta": float(beta),
                "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
                "ed_backend": "quspin",
                "basis_type": basis_type,
                "effective_hilbert_dimension": int(hilbert_dim),
                "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
                "full_hilbert_dimension": int(full_hilbert_dim),
                "use_sz_conserved": bool(use_sz_block),
                "use_sz_block": bool(use_sz_block),
            }
        if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
            return {
                "status": "skipped",
                "alpha": float(alpha),
                "beta": float(beta),
                "reason": (
                    f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                    f"{int(args.phase_scan_ed_max_hilbert_dim)}."
                ),
                "ed_backend": "quspin",
                "basis_type": basis_type,
                "effective_hilbert_dimension": int(hilbert_dim),
                "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
                "full_hilbert_dimension": int(full_hilbert_dim),
                "use_sz_conserved": bool(use_sz_block),
                "use_sz_block": bool(use_sz_block),
            }

        import quspin_backend as quspin_ed_backend

        spectrum, vectors = quspin_ed_backend.run_small_cluster_exact_spectrum(
            geometry=geometry,
            model_spec=model_spec,
            alpha=alpha,
            beta=beta,
            coupling_j=args.coupling_j,
            eigenstate_count=max(1, int(getattr(args, "ed_max_eigenstates", 2))),
            check_ground_state_degeneracy=False,
            jx=args.jx,
            jy=args.jy,
            jz=args.jz,
            external_field_terms=hamiltonian_external_field_terms,
            show_progress=show_progress,
            solver=getattr(args, "ed_solver", "auto"),
            sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
            sparse_maxiter=(
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
            use_sz_block=use_sz_block,
            target_sz2=target_sz2,
            use_tau_z_block=use_tau_z_block,
            target_tz2=target_tz2,
            use_z2_block=use_z2_block,
            z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
            use_translation_block=use_translation_block,
            use_translation_x_block=use_translation_x_block,
            use_translation_y_block=use_translation_y_block,
            momentum_block_1=momentum_x_block,
            momentum_block_2=momentum_y_block,
            momentum_x_block=momentum_x_block,
            momentum_y_block=momentum_y_block,
            use_reflection_block=use_reflection_block,
            reflection_block=reflection_block,
            check_symm=bool(getattr(args, "quspin_check_symmetries", False)),
            check_herm=bool(getattr(args, "quspin_check_hermiticity", False)),
            check_pcon=bool(getattr(args, "quspin_check_particle_conservation", False)),
        )
        basis_use_sz_block = bool(spectrum.get("use_sz_block", use_sz_block))
        basis_target_sz2 = int(spectrum.get("target_sz2", target_sz2))
        basis_use_z2_block = bool(spectrum.get("use_z2_block", use_z2_block))
        basis = quspin_ed_backend.build_quspin_yao_lee_basis(
            n_sites,
            geometry=geometry,
            use_sz_block=basis_use_sz_block,
            target_sz2=basis_target_sz2,
            use_tau_z_block=use_tau_z_block,
            target_tz2=target_tz2,
            use_z2_block=basis_use_z2_block,
            z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
            use_translation_block=use_translation_block,
            use_translation_x_block=use_translation_x_block,
            use_translation_y_block=use_translation_y_block,
            momentum_block_1=momentum_x_block,
            momentum_block_2=momentum_y_block,
            momentum_x_block=momentum_x_block,
            momentum_y_block=momentum_y_block,
            use_reflection_block=use_reflection_block,
            reflection_block=reflection_block,
        )
        energy = float(spectrum["ground_state_energy"])
        state = vectors[:, 0]
        scalar_correlations = quspin_ed_backend.build_spin_orbital_scalar_correlations(
            basis,
            state,
            n_sites,
        )
        bond_rows = quspin_ed_backend.all_bond_energies(
            geometry,
            scalar_correlations,
            alpha,
            beta,
            args.coupling_j,
        )
        structure_rows = quspin_ed_backend.all_high_symmetry_structure_factors(
            scalar_correlations,
            geometry,
        )
        plaquette_flux = spectrum.get("plaquette_flux") if isinstance(spectrum, dict) else None
        if plaquette_flux is None:
            try:
                plaquette_flux = quspin_ed_backend.compute_plaquette_flux(
                    basis,
                    state,
                    geometry,
                    plaquette_center_idx=None,
                )
            except Exception as exc:
                plaquette_flux = {"available": False, "warning": str(exc)}
        diagnostics = _phase_observable_diagnostics(
            structure_rows,
            bond_rows,
            geometry.number_of_sites,
            plaquette_flux=plaquette_flux,
        )
        phase_label = _classify_phase_from_diagnostics(diagnostics, alpha, beta, "quantum_ed", thresholds)
        return {
            "status": "completed",
            "alpha": float(alpha),
            "beta": float(beta),
            "ed_backend": "quspin",
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(spectrum.get("hilbert_dimension", hilbert_dim)),
            "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_block),
            "use_sz_block": bool(use_sz_block),
            "selected_target_sz2": int(spectrum.get("target_sz2", target_sz2)),
            "quspin_effective_use_z2_block": bool(spectrum.get("use_z2_block", use_z2_block)),
            "quspin_sz_sector_scan": spectrum.get("sz_sector_scan"),
            "quspin_package_available": bool(quspin_package_available),
            "quspin_requested_translation_block": bool(requested_translation_block),
            "quspin_requested_translation_x_block": bool(requested_translation_x_block),
            "quspin_requested_translation_y_block": bool(requested_translation_y_block),
            "quspin_use_translation_block": bool(use_translation_block),
            "quspin_use_translation_x_block": bool(use_translation_x_block),
            "quspin_use_translation_y_block": bool(use_translation_y_block),
            "quspin_translation_reason": quspin_translation_reason,
            "quspin_translation_x_reason": quspin_translation_x_reason,
            "quspin_translation_y_reason": quspin_translation_y_reason,
            "quspin_requested_reflection_block": bool(requested_reflection_block),
            "quspin_use_reflection_block": False,
            "quspin_reflection_reason": quspin_reflection_reason,
            "phase_label": phase_label,
            "energy": float(energy),
            "energy_per_site": float(energy / float(max(1, geometry.number_of_sites))),
            "diagnostics": diagnostics,
            "plaquette_flux": plaquette_flux,
            "all_plaquette_fluxes": extract_all_plaquette_fluxes(plaquette_flux),
            "structure_factors": structure_rows,
            "bond_energies": bond_rows,
        }

    use_sz_conserved_requested = bool(use_sz_block)
    use_sz_conserved = (
        use_sz_conserved_requested
        and str(getattr(model_spec, "spin_rep", "")) == "1/2"
        and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
        and _sector_dimension_for_spin_half(n_sites, target_sz2) > 0
    )
    if use_sz_conserved_requested and not use_sz_conserved and str(getattr(model_spec, "orbital_rep", "")) != "0":
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": "Standard Sz-block ED requires a reachable target 2*Sz sector with spin_rep=1/2 and orbital_rep=1/2.",
            "ed_backend": "standard",
            "use_sz_conserved_requested": bool(use_sz_conserved_requested),
        }
    if use_sz_conserved:
        hilbert_dim = int(_sector_dimension_for_spin_half(n_sites, target_sz2) * (1 << n_sites))
        basis_type = "bitwise_spin_orbital_total_sz_block"
    else:
        hilbert_dim = full_hilbert_dim
        basis_type = "legacy_full_tensor_product"
    if int(geometry.number_of_sites) > int(args.phase_scan_ed_max_sites):
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
            "ed_backend": "standard",
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_conserved),
        }
    if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": (
                f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                f"{int(args.phase_scan_ed_max_hilbert_dim)}."
            ),
            "ed_backend": "standard",
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_conserved),
        }
    if use_sz_conserved:
        from ed_backend import (
            all_bond_energies_sz_conserved,
            build_sz_conserved_scalar_correlations,
            collect_correlation_matrices_from_sz_conserved_ed,
            plaquette_flux_from_sz_conserved_ed_state,
            run_sz_conserved_exact_spectrum,
        )

        spectrum, vectors, basis_list, basis_map = run_sz_conserved_exact_spectrum(
            geometry=geometry,
            alpha=alpha,
            beta=beta,
            coupling_j=args.coupling_j,
            eigenstate_count=3,
            check_ground_state_degeneracy=False,
            external_field_terms=hamiltonian_external_field_terms,
            show_progress=show_progress,
            sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
            sparse_maxiter=(
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
            target_sz2=target_sz2,
        )
        energy = float(spectrum["ground_state_energy"])
        state = vectors[:, 0]
        correlations = collect_correlation_matrices_from_sz_conserved_ed(
            geometry,
            state,
            basis_list,
            basis_map,
            show_progress=show_progress,
        )
        scalar_correlations = build_sz_conserved_scalar_correlations(correlations)
        bond_rows = all_bond_energies_sz_conserved(
            geometry,
            correlations,
            alpha,
            beta,
            args.coupling_j,
            show_progress=show_progress,
        )
        try:
            plaquette_flux = plaquette_flux_from_sz_conserved_ed_state(
                geometry,
                state,
                basis_list,
                basis_map,
                plaquette_center_idx=None,
            )
        except Exception as exc:
            plaquette_flux = {"available": False, "warning": str(exc)}
    else:
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
            show_progress=show_progress,
            solver=getattr(args, "ed_solver", "auto"),
            sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
            sparse_maxiter=(
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
        )
        correlations = collect_correlation_matrices_from_ed(
            geometry,
            state,
            model_spec=model_spec,
            show_progress=show_progress,
        )
        scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
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
            show_progress=show_progress,
        )
        try:
            plaquette_flux = plaquette_flux_from_ed_state(
                geometry,
                state,
                model_spec,
                plaquette_center_idx=None,
            )
        except Exception as exc:
            plaquette_flux = {"available": False, "warning": str(exc)}
    structure_rows = all_high_symmetry_structure_factors(
        scalar_correlations,
        geometry,
        lattice=lattice_name,
        show_progress=show_progress,
    )
    diagnostics = _phase_observable_diagnostics(
        structure_rows,
        bond_rows,
        geometry.number_of_sites,
        plaquette_flux=plaquette_flux,
    )
    phase_label = _classify_phase_from_diagnostics(diagnostics, alpha, beta, "quantum_ed", thresholds)
    return {
        "status": "completed",
        "alpha": float(alpha),
        "beta": float(beta),
        "ed_backend": "standard",
        "basis_type": basis_type,
        "effective_hilbert_dimension": int(hilbert_dim),
        "full_hilbert_dimension": int(full_hilbert_dim),
        "use_sz_conserved": bool(use_sz_conserved),
        "phase_label": phase_label,
        "energy": float(energy),
        "energy_per_site": float(energy / float(max(1, geometry.number_of_sites))),
        "diagnostics": diagnostics,
        "plaquette_flux": plaquette_flux,
        "all_plaquette_fluxes": extract_all_plaquette_fluxes(plaquette_flux),
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


def _phase_scan_quantum_global_skip(
    geometry: Any,
    model_spec: Any,
    args: Any,
    hamiltonian_external_field_terms: List[Tuple[float, str]] | None = None,
) -> Dict[str, Any] | None:
    local_dim = int(model_spec.physical_dim)
    n_sites = int(geometry.number_of_sites)
    full_hilbert_dim = int(local_dim ** n_sites)
    external_field_terms = list(hamiltonian_external_field_terms or [])
    ed_backend_name = str(getattr(args, "ed_backend", "standard")).strip().lower()
    if ed_backend_name == "ed":
        ed_backend_name = "standard"
    use_sz_block = _phase_scan_uses_sz_block(args)
    use_tau_z_block = _phase_scan_uses_tau_z_block(args)
    use_z2_block = bool(getattr(args, "use_z2_block", False))
    use_translation_x_block = bool(getattr(args, "use_translation_x_block", False))
    use_translation_y_block = bool(getattr(args, "use_translation_y_block", False))
    use_translation_block = bool(use_translation_x_block or use_translation_y_block)
    use_reflection_block = bool(getattr(args, "use_reflection_block", False))
    requested_translation_block = bool(use_translation_block)
    requested_translation_x_block = bool(use_translation_x_block)
    requested_translation_y_block = bool(use_translation_y_block)
    requested_reflection_block = bool(use_reflection_block)
    reflection_block = int(getattr(args, "reflection_block", 0))
    target_sz2 = int(getattr(args, "u1_target_sz2", 0))
    target_tz2 = int(getattr(args, "u1_target_tz2", 0))
    field_ops = {
        str(op_name)
        for coefficient, op_name in list(external_field_terms or [])
        if abs(float(coefficient)) > 1e-14
    }

    if ed_backend_name == "quspin":
        if bool(use_sz_block) and bool(field_ops.intersection({"Sx", "Sy"})):
            use_sz_block = False
            use_z2_block = False
        if bool(use_z2_block) and bool(field_ops.intersection({"Sx", "Sy", "Sz"})):
            use_z2_block = False
        quspin_package_available = importlib.util.find_spec("quspin") is not None
        quspin_translation_x_reason = None
        quspin_translation_y_reason = None
        if use_translation_block:
            if not quspin_package_available:
                reason = "QuSpin package is not installed, so translation blocks cannot be checked."
                use_translation_x_block = False
                use_translation_y_block = False
                quspin_translation_x_reason = reason if requested_translation_x_block else None
                quspin_translation_y_reason = reason if requested_translation_y_block else None
            else:
                try:
                    import quspin_backend as quspin_validation_backend

                    support = quspin_validation_backend.quspin_translation_block_support(geometry)
                    x_support = support.get("x", {})
                    y_support = support.get("y", {})
                    use_translation_x_block = bool(
                        requested_translation_x_block and x_support.get("supported", False)
                    )
                    use_translation_y_block = bool(
                        requested_translation_y_block and y_support.get("supported", False)
                    )
                    quspin_translation_x_reason = x_support.get("reason") if requested_translation_x_block else None
                    quspin_translation_y_reason = y_support.get("reason") if requested_translation_y_block else None
                except Exception as exc:
                    reason = str(exc)
                    use_translation_x_block = False
                    use_translation_y_block = False
                    quspin_translation_x_reason = reason if requested_translation_x_block else None
                    quspin_translation_y_reason = reason if requested_translation_y_block else None
        use_translation_block = bool(use_translation_x_block or use_translation_y_block)
        quspin_translation_reason = {
            "x": quspin_translation_x_reason,
            "y": quspin_translation_y_reason,
        }
        quspin_reflection_reason = None
        if requested_reflection_block or reflection_block != 0:
            quspin_reflection_reason = (
                "QuSpin reflection/C3 blocks are not applied for the bond-directional Yao-Lee Hamiltonian; "
                "they can permute x/y/z bond types unless a gauge map is implemented."
            )
        use_reflection_block = False
        reflection_block = 0
        hilbert_dim = _phase_scan_spin_orbital_block_dimension(
            n_sites,
            use_sz_block,
            target_sz2,
            use_tau_z_block,
            target_tz2,
        )
        basis_type = (
            "quspin_tensor_"
            f"spin_{'u1_block' if use_sz_block else 'full'}_"
            f"orbital_{'u1_block' if use_tau_z_block else 'full'}"
        )
        compatible = (
            quspin_package_available
            and str(getattr(model_spec, "spin_rep", "")) == "1/2"
            and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
            and str(getattr(model_spec, "model_family", "")) == "yao_lee"
            and str(getattr(model_spec, "ising_axis", "")) == "z"
            and int(hilbert_dim) > 0
            and (not use_z2_block or (use_sz_block and target_sz2 == 0))
            and not (use_tau_z_block and (use_z2_block or use_translation_block))
        )
        pre_quspin_hilbert_dim = int(hilbert_dim)
        quspin_basis_build_reason = None
        if compatible:
            try:
                import quspin_backend as quspin_basis_backend

                preflight_basis = quspin_basis_backend.build_quspin_yao_lee_basis(
                    n_sites,
                    geometry=geometry,
                    use_sz_block=use_sz_block,
                    target_sz2=target_sz2,
                    use_tau_z_block=use_tau_z_block,
                    target_tz2=target_tz2,
                    use_z2_block=use_z2_block,
                    z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
                    use_translation_block=use_translation_block,
                    use_translation_x_block=use_translation_x_block,
                    use_translation_y_block=use_translation_y_block,
                    momentum_block_1=int(getattr(args, "momentum_x_block", 0)),
                    momentum_block_2=int(getattr(args, "momentum_y_block", 0)),
                    momentum_x_block=int(getattr(args, "momentum_x_block", 0)),
                    momentum_y_block=int(getattr(args, "momentum_y_block", 0)),
                    use_reflection_block=False,
                    reflection_block=0,
                )
                hilbert_dim = int(preflight_basis.Ns)
            except Exception as exc:
                compatible = False
                quspin_basis_build_reason = f"Failed to build the requested QuSpin reduced basis: {exc}"
        common = {
            "ed_backend": "quspin",
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(hilbert_dim),
            "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_block),
            "use_sz_block": bool(use_sz_block),
            "use_sz_conserved_requested": bool(use_sz_block),
            "quspin_package_available": bool(quspin_package_available),
            "quspin_requested_translation_block": bool(requested_translation_block),
            "quspin_requested_translation_x_block": bool(requested_translation_x_block),
            "quspin_requested_translation_y_block": bool(requested_translation_y_block),
            "quspin_use_translation_block": bool(use_translation_block),
            "quspin_use_translation_x_block": bool(use_translation_x_block),
            "quspin_use_translation_y_block": bool(use_translation_y_block),
            "quspin_translation_reason": quspin_translation_reason,
            "quspin_translation_x_reason": quspin_translation_x_reason,
            "quspin_translation_y_reason": quspin_translation_y_reason,
            "quspin_requested_reflection_block": bool(requested_reflection_block),
            "quspin_use_reflection_block": False,
            "quspin_reflection_reason": quspin_reflection_reason,
        }
        if not compatible:
            return {
                **common,
                "reason": (
                    quspin_basis_build_reason
                    if quspin_basis_build_reason is not None
                    else
                    "QuSpin ED phase scan requires reachable shared U1 target sectors, "
                    "the quspin Python package, "
                    "spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, "
                    "and ising_axis=z. "
                    "Reflection/C3 blocks are forbidden; spin-flip Z2 requires total Sz=0; "
                    "tau_z is not combined with Z2/2D translations."
                ),
            }
        if n_sites > int(args.phase_scan_ed_max_sites):
            return {
                **common,
                "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
            }
        if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
            return {
                **common,
                "reason": (
                    f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                    f"{int(args.phase_scan_ed_max_hilbert_dim)}."
                ),
            }
        return None

    use_sz_conserved_requested = bool(use_sz_block)
    use_sz_conserved = (
        use_sz_conserved_requested
        and str(getattr(model_spec, "spin_rep", "")) == "1/2"
        and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
        and _sector_dimension_for_spin_half(n_sites, target_sz2) > 0
    )
    if use_sz_conserved:
        hilbert_dim = int(_sector_dimension_for_spin_half(n_sites, target_sz2) * (1 << n_sites))
        basis_type = "bitwise_spin_orbital_total_sz_block"
    else:
        hilbert_dim = full_hilbert_dim
        basis_type = "legacy_full_tensor_product"

    common = {
        "ed_backend": "standard",
        "basis_type": basis_type,
        "effective_hilbert_dimension": int(hilbert_dim),
        "full_hilbert_dimension": int(full_hilbert_dim),
        "use_sz_conserved": bool(use_sz_conserved),
        "use_sz_conserved_requested": bool(use_sz_conserved_requested),
    }
    if use_sz_conserved_requested and not use_sz_conserved and str(getattr(model_spec, "orbital_rep", "")) != "0":
        return {
            **common,
            "reason": "Standard Sz-block ED requires a reachable target 2*Sz sector with spin_rep=1/2 and orbital_rep=1/2.",
        }
    if n_sites > int(args.phase_scan_ed_max_sites):
        return {
            **common,
            "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
        }
    if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
        return {
            **common,
            "reason": (
                f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                f"{int(args.phase_scan_ed_max_hilbert_dim)}."
            ),
        }
    return None


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
    raw_mode = str(getattr(args, "phase_scan_mode", "quantum_ed")).strip().lower()
    if raw_mode in ("both", "all"):
        modes = ["quantum_ed", "classical_product"]
    elif raw_mode in ("classical", "classical_product"):
        modes = ["classical_product"]
    elif raw_mode in ("quantum", "methods", "quantum_ed", "ed", "exact", "exact_diagonalization"):
        modes = ["quantum_ed"]
    else:
        modes = [raw_mode]
    total_points = len(alphas) * len(betas)
    quantum_ed_backend = str(getattr(args, "ed_backend", "standard")).strip().lower()
    if quantum_ed_backend == "ed":
        quantum_ed_backend = "standard"
    quantum_ed_use_sz_block = _phase_scan_uses_sz_block(args)
    quantum_ed_use_tau_z_block = _phase_scan_uses_tau_z_block(args)
    quantum_ed_reductions = _phase_scan_reductions(args)
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
            "quantum_ed_backend": quantum_ed_backend,
            "quantum_ed_solver": str(getattr(args, "ed_solver", "auto")),
            "quantum_ed_use_sz_block": bool(quantum_ed_use_sz_block),
            "quantum_ed_use_tau_z_block": bool(quantum_ed_use_tau_z_block),
            "quantum_ed_use_z2_block": bool(getattr(args, "use_z2_block", False)),
            "quantum_ed_use_translation_x_block": bool(getattr(args, "use_translation_x_block", False)),
            "quantum_ed_use_translation_y_block": bool(getattr(args, "use_translation_y_block", False)),
            "symmetry_reductions": list(quantum_ed_reductions),
            "symmetry_mode": str(getattr(args, "symmetry_mode", "none")),
            "target_sz2": int(getattr(args, "u1_target_sz2", 0)),
            "target_tz2": int(getattr(args, "u1_target_tz2", 0)),
            "z2_target_parity": int(getattr(args, "z2_target_parity", 0)) % 2,
            "quantum_ed_sparse_tol": float(getattr(args, "ed_sparse_tol", 0.0)),
            "quantum_ed_sparse_maxiter": (
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
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
            "spin/orbital structure factors, plaquette flux, and bond-energy nematicity. "
            "They are saved with diagnostics for reproducibility and should be checked "
            "against larger clusters or denser grids before quoting thermodynamic boundaries."
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
        if mode == "quantum_ed":
            skip_info = _phase_scan_quantum_global_skip(
                geometry,
                model_spec,
                args,
                hamiltonian_external_field_terms,
            )
            if skip_info is not None:
                output[mode] = {
                    "status": "skipped",
                    "reason": str(skip_info["reason"]),
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(total_points),
                    **skip_info,
                    "note": (
                        "Quantum ED phase scan was skipped once at solver setup; "
                        "per-alpha/beta skipped rows are intentionally omitted."
                    ),
                }
                if progress_bar is not None:
                    progress_bar.update(total_points)
                point_index += int(total_points)
                continue

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
                            show_progress=False,
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
