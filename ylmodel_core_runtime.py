#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

ENTROPY_ORDERS = (1, 2, 3, 4)


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


# ----------------------------------------------------------------------
