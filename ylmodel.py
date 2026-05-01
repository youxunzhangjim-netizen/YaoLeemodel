#!/usr/bin/env python3
"""
Yao-Lee model benchmarking with Tenax DMRG (plus optional ED) using PNG outputs.

This version is designed to:
1) run from sensible defaults without required CLI arguments,
2) keep Tenax integration robust across minor API differences, and
3) save analysis as plots/diagrams (PNG) instead of CSV tables.
"""

from __future__ import annotations
import argparse
import contextlib
import importlib
import inspect
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg


# ----------------------------------------------------------------------
# Configuration (edit this top block for normal runs)
# ----------------------------------------------------------------------

# Available choices used by the defaults and CLI.
LATTICE_OPTIONS = ("honeycomb", "square", "triangular")
MODEL_FAMILY_OPTIONS = ("yao_lee", "ising_like")
SPIN_REP_OPTIONS = ("1/2", "3/2")
ORBITAL_REP_OPTIONS = ("0", "1/2")
AXIS_OPTIONS = ("x", "y", "z")
INITIAL_STATE_OPTIONS = ("alternating", "random")
SYMMETRY_MODE_OPTIONS = ("none", "u1", "z2")
Z2_PARITY_OPTIONS = (0, 1)
IDMRG_BULK_KIND_OPTIONS = ("auto", "pair", "single")
BACKEND_OPTIONS = ("auto", "tenax", "tenpy")
ENTROPY_ORDERS = (1, 2, 3, 4)

# Geometry.
LENGTH_X = 2
CIRCUMFERENCE_Y = 2
PERIODIC_AROUND_CYLINDER = True
LATTICE_TYPE = "honeycomb"  # honeycomb | square | triangular

# Hamiltonian/model.
MODEL_FAMILY = "yao_lee"    # yao_lee | ising_like
SPIN_REP = "1/2"            # 1/2 | 3/2
ORBITAL_REP = "1/2"         # 0 | 1/2 ; CLI also accepts legacy alias 1 -> 0
ISING_AXIS = "z"            # x | y | z
ALPHA = 1.0
BETA = 0
COUPLING_J = 0.0

# Symmetry simplification/block-sparse controls.
# none: dense tensors, no symmetry constraints.
# u1:   encoded U(1)xU(1) charges using target (2*Sz, 2*Tz).
#       This is only valid when the Hamiltonian conserves total Sz/Tz.
#       Bond-dependent x/y Yao-Lee terms should use z2 or none instead.
# z2:   parity charges using target even/odd sector. This supports x/y flip pairs.
SYMMETRY_MODE = "u1"      # none | u1 | z2
U1_TARGET_TOTAL_SZ2 = 0     # equals 2 * total S^z
U1_TARGET_TOTAL_TZ2 = 0     # equals 2 * total T^z
Z2_TARGET_PARITY = 0        # 0=even, 1=odd
STRICT_SYMMETRY_SELECTION_RULES = True

# Finite DMRG solver.
MAX_BOND_DIMENSION = 400
MAX_SWEEPS = 20
TRUNCATION_CUTOFF = 1e-8
SEED = 42
INITIAL_STATE_STYLE = "random"  # alternating | random

# Optional comparison workflows.
RUN_ED = True
MAX_ED_SITES = 18
MAX_ED_HILBERT_DIM = 4 ** MAX_ED_SITES
RUN_IDMRG = True
IDMRG_MAX_ITERATIONS = MAX_SWEEPS
IDMRG_MAX_LOCAL_DIM = 256
IDMRG_BULK_KIND = "auto"  # auto | pair | single

# Output/runtime behavior.
OUTPUT_FOLDER = "DMRG/outputs"
BACKEND = "auto"  # auto | tenax | tenpy
OVERWRITE_EXISTING_PLOTS = False
CONTINUE_AFTER_PLOT_ERROR = True
STRICT_PLOT_ERRORS = not CONTINUE_AFTER_PLOT_ERROR
SHOW_PROGRESS = True


# ----------------------------------------------------------------------
# Tenax API loading
# ----------------------------------------------------------------------

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
# Local operators
# ----------------------------------------------------------------------

AXES = AXIS_OPTIONS
SPIN_REP_VALUES = {"1/2": 0.5, "3/2": 1.5}
# "1" is kept as a legacy alias and normalized to "0".
ORBITAL_REP_VALUES = {"0": 0.0, "1": 0.0, "1/2": 0.5}


@dataclass(frozen=True)
class ModelSpec:
    spin_rep: str
    orbital_rep: str
    model_family: str
    ising_axis: str
    spin_value: float
    orbital_value: float
    spin_dim: int
    orbital_dim: int
    physical_dim: int


def build_model_spec(
    spin_rep: str,
    orbital_rep: str,
    model_family: str,
    ising_axis: str,
) -> ModelSpec:
    spin_text = str(spin_rep).strip()
    orbital_text = str(orbital_rep).strip()
    if spin_text not in SPIN_REP_VALUES:
        raise ValueError(f"Unsupported spin_rep '{spin_rep}'. Choose from {sorted(SPIN_REP_VALUES.keys())}.")
    if orbital_text not in ORBITAL_REP_VALUES:
        raise ValueError(
            f"Unsupported orbital_rep '{orbital_rep}'. Choose from ['0', '1/2']."
        )
    if orbital_text == "1":
        orbital_text = "0"

    family = str(model_family).strip().lower()
    if family not in ("yao_lee", "ising_like"):
        raise ValueError("model_family must be one of: yao_lee, ising_like.")

    axis = str(ising_axis).strip().lower()
    if axis not in AXES:
        raise ValueError("ising_axis must be one of: x, y, z.")

    spin_value = SPIN_REP_VALUES[spin_text]
    orbital_value = ORBITAL_REP_VALUES[orbital_text]
    spin_dim = int(round(2.0 * spin_value + 1.0))
    orbital_dim = int(round(2.0 * orbital_value + 1.0))
    return ModelSpec(
        spin_rep=spin_text,
        orbital_rep=orbital_text,
        model_family=family,
        ising_axis=axis,
        spin_value=spin_value,
        orbital_value=orbital_value,
        spin_dim=spin_dim,
        orbital_dim=orbital_dim,
        physical_dim=spin_dim * orbital_dim,
    )


def is_trivial_orbital(model_spec: ModelSpec) -> bool:
    return model_spec.orbital_dim == 1


def _normalize_symmetry_mode(mode: str | None) -> str:
    text = str(mode if mode is not None else "none").strip().lower()
    if text in ("none", "off", "dense", "false", "0"):
        return "none"
    if text in ("u1", "u(1)", "u1x", "u1xu1"):
        return "u1"
    if text in ("z2", "z_2", "z(2)", "parity"):
        return "z2"
    raise ValueError(f"Unsupported symmetry mode '{mode}'. Choose from: {', '.join(SYMMETRY_MODE_OPTIONS)}.")


def _m2_values_from_spin_value(spin_value: float) -> List[int]:
    two_s = int(round(2.0 * spin_value))
    return [two_s - 2 * index for index in range(two_s + 1)]


def _get_z2_symmetry_object() -> Any:
    """Return a Z2 symmetry object from Tenax (direct class or ZnSymmetry(2) fallback)."""
    try:
        from tenax.core.symmetry import Z2Symmetry  # type: ignore
        return Z2Symmetry()
    except Exception:
        from tenax.core.symmetry import ZnSymmetry
        return ZnSymmetry(2)


def _u1_encoded_phys_charges_for_model(model_spec: ModelSpec) -> np.ndarray:
    """Encode (2*Sz, 2*Tz) into one integer charge for Tenax U(1) tensors."""
    try:
        from tenax.core.symmetry import ProductSymmetry
    except Exception as exc:
        raise RuntimeError("Tenax ProductSymmetry is required for encoded (Sz,Tz) charges.") from exc

    spin_m2_values = _m2_values_from_spin_value(model_spec.spin_value)
    orb_m2_values = _m2_values_from_spin_value(model_spec.orbital_value)
    encoded: List[int] = []
    for sz2 in spin_m2_values:
        for tz2 in orb_m2_values:
            encoded.append(int(ProductSymmetry.encode(int(sz2), int(tz2))))
    return np.asarray(encoded, dtype=np.int32)


def _z2_phys_charges_for_model(model_spec: ModelSpec) -> np.ndarray:
    """Parity charge per local basis state: even=0, odd=1."""
    charges: List[int] = []
    for spin_index in range(int(model_spec.spin_dim)):
        for orbital_index in range(int(model_spec.orbital_dim)):
            charges.append(int((spin_index + orbital_index) % 2))
    return np.asarray(charges, dtype=np.int32)


def _u1_basis_charge_table_for_model(model_spec: ModelSpec) -> List[Dict[str, Any]]:
    spin_m2_values = _m2_values_from_spin_value(model_spec.spin_value)
    orb_m2_values = _m2_values_from_spin_value(model_spec.orbital_value)
    encoded = _u1_encoded_phys_charges_for_model(model_spec)
    table: List[Dict[str, Any]] = []
    idx = 0
    for sz2 in spin_m2_values:
        for tz2 in orb_m2_values:
            table.append(
                {
                    "basis": f"|Sz={0.5 * sz2:+g},Tz={0.5 * tz2:+g}>",
                    "Sz_times_2": int(sz2),
                    "Tz_times_2": int(tz2),
                    "encoded_u1_charge": int(encoded[idx]),
                }
            )
            idx += 1
    return table


def _z2_basis_charge_table_for_model(model_spec: ModelSpec) -> List[Dict[str, Any]]:
    spin_m2_values = _m2_values_from_spin_value(model_spec.spin_value)
    orb_m2_values = _m2_values_from_spin_value(model_spec.orbital_value)
    charges = _z2_phys_charges_for_model(model_spec)
    table: List[Dict[str, Any]] = []
    idx = 0
    for spin_index, sz2 in enumerate(spin_m2_values):
        for orbital_index, tz2 in enumerate(orb_m2_values):
            table.append(
                {
                    "basis": f"|Sz={0.5 * sz2:+g},Tz={0.5 * tz2:+g}>",
                    "Sz_times_2": int(sz2),
                    "Tz_times_2": int(tz2),
                    "parity_charge": int(charges[idx]),
                    "parity_label": "even" if int(charges[idx]) == 0 else "odd",
                    "parity_formula": f"({spin_index}+{orbital_index}) mod 2",
                }
            )
            idx += 1
    return table


def _u1_encoded_target_charge(total_sz2: int, total_tz2: int) -> int:
    try:
        from tenax.core.symmetry import ProductSymmetry
    except Exception as exc:
        raise RuntimeError("Tenax ProductSymmetry is required for encoded target charge.") from exc
    return int(ProductSymmetry.encode(int(total_sz2), int(total_tz2)))


def _operator_charge_transfer(
    op: np.ndarray,
    phys_charges: np.ndarray,
    symmetry_mode: str,
) -> int:
    """Return the unique charge transfer of an operator; raise if it mixes sectors."""
    mode = _normalize_symmetry_mode(symmetry_mode)
    charge: int | None = None
    abs_op = np.abs(op)
    for row in range(op.shape[0]):
        for col in range(op.shape[1]):
            if abs_op[row, col] <= 1e-14:
                continue
            transfer = int(phys_charges[row]) - int(phys_charges[col])
            if mode == "z2":
                transfer = int(transfer % 2)
            if charge is None:
                charge = transfer
            elif charge != transfer:
                raise ValueError(
                    f"Operator mixes {mode.upper()} sectors and is not symmetry-adapted for strict symmetric MPO build."
                )
    return 0 if charge is None else int(charge)


def _validate_symmetry_conserving_terms(
    terms: List[Tuple[Any, ...]],
    site_ops: Dict[str, np.ndarray],
    phys_charges: np.ndarray,
    symmetry_mode: str,
) -> None:
    """Ensure every AutoMPO term has net symmetry charge 0."""
    mode = _normalize_symmetry_mode(symmetry_mode)
    if mode == "none":
        return

    for term in terms:
        args = term[1:]
        if len(args) % 2 != 0:
            raise ValueError(f"Invalid term format for AutoMPO: {term}")
        net_charge = 0
        for idx in range(0, len(args), 2):
            op_name = str(args[idx])
            if op_name not in site_ops:
                raise KeyError(f"Unknown operator '{op_name}' in term {term}.")
            try:
                net_charge += _operator_charge_transfer(site_ops[op_name], phys_charges, mode)
            except ValueError as exc:
                raise ValueError(
                    f"Operator '{op_name}' in term {term} is not compatible with strict "
                    f"{mode.upper()} symmetry. Use a symmetry-adapted operator pair, "
                    f"switch to symmetry_mode=z2/none, or choose a U1-conserving model."
                ) from exc
        if mode == "z2":
            net_charge = int(net_charge % 2)
        if int(net_charge) != 0:
            raise ValueError(
                f"Found non-charge-conserving term for strict {mode.upper()} build: "
                f"{term} (net charge transfer={net_charge})."
            )


def build_spin_operators(spin_value: float) -> Dict[str, np.ndarray]:
    two_s = int(round(2.0 * spin_value))
    dim = int(two_s + 1)
    m2_values = [two_s - 2 * index for index in range(dim)]  # stores 2*m
    m2_to_index = {m2: index for index, m2 in enumerate(m2_values)}

    s_plus = np.zeros((dim, dim), dtype=np.complex128)
    s_minus = np.zeros((dim, dim), dtype=np.complex128)
    for col_index, m2 in enumerate(m2_values):
        m = 0.5 * m2
        m2_plus = m2 + 2
        if m2_plus in m2_to_index:
            row_index = m2_to_index[m2_plus]
            coeff = np.sqrt(spin_value * (spin_value + 1.0) - m * (m + 1.0))
            s_plus[row_index, col_index] = coeff
        m2_minus = m2 - 2
        if m2_minus in m2_to_index:
            row_index = m2_to_index[m2_minus]
            coeff = np.sqrt(spin_value * (spin_value + 1.0) - m * (m - 1.0))
            s_minus[row_index, col_index] = coeff

    sx = 0.5 * (s_plus + s_minus)
    sy = -0.5j * (s_plus - s_minus)
    sz = np.diag([0.5 * m2 for m2 in m2_values]).astype(np.complex128)
    ident = np.eye(dim, dtype=np.complex128)
    return {"Id": ident, "Sx": sx, "Sy": sy, "Sz": sz}


def build_site_ops(model_spec: ModelSpec) -> Dict[str, np.ndarray]:
    spin_ops = build_spin_operators(model_spec.spin_value)
    orbital_ops = build_spin_operators(model_spec.orbital_value)
    ident_spin = spin_ops["Id"]
    ident_orb = orbital_ops["Id"]
    s_plus = spin_ops["Sx"] + 1.0j * spin_ops["Sy"]
    s_minus = spin_ops["Sx"] - 1.0j * spin_ops["Sy"]
    t_plus = orbital_ops["Sx"] + 1.0j * orbital_ops["Sy"]
    t_minus = orbital_ops["Sx"] - 1.0j * orbital_ops["Sy"]

    ops: Dict[str, np.ndarray] = {
        "Id": np.kron(ident_spin, ident_orb),
        "Sx": np.kron(spin_ops["Sx"], ident_orb),
        "Sy": np.kron(spin_ops["Sy"], ident_orb),
        "Sz": np.kron(spin_ops["Sz"], ident_orb),
        "Sp": np.kron(s_plus, ident_orb),
        "Sm": np.kron(s_minus, ident_orb),
        "Tx": np.kron(ident_spin, orbital_ops["Sx"]),
        "Ty": np.kron(ident_spin, orbital_ops["Sy"]),
        "Tz": np.kron(ident_spin, orbital_ops["Sz"]),
        "Tp": np.kron(ident_spin, t_plus),
        "Tm": np.kron(ident_spin, t_minus),
    }
    ops["STx"] = ops["Sx"] @ ops["Tx"]
    ops["STy"] = ops["Sy"] @ ops["Ty"]
    ops["STz"] = ops["Sz"] @ ops["Tz"]
    ops["STp"] = ops["Sp"] @ ops["Tp"]
    ops["STm"] = ops["Sm"] @ ops["Tm"]
    return ops


def build_yao_lee_site_ops() -> Dict[str, np.ndarray]:
    """Legacy compatibility wrapper using current default model settings."""
    default_spec = build_model_spec(
        spin_rep=SPIN_REP,
        orbital_rep=ORBITAL_REP,
        model_family=MODEL_FAMILY,
        ising_axis=ISING_AXIS,
    )
    return build_site_ops(default_spec)


def model_terms_for_bond(
    gamma: str,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
) -> List[Tuple[float, str]]:
    axis_gamma = str(gamma).lower()
    if axis_gamma not in AXES:
        raise ValueError(f"Unknown bond axis '{gamma}'.")

    # Requested behavior:
    # orbital_rep == "1" means no orbital DOF, and Yao-Lee orbital-dependent
    # terms reduce to spin-only Ising-like couplings.
    if is_trivial_orbital(model_spec):
        axis = model_spec.ising_axis
        return [(coupling_j * (1.0 + beta), f"S{axis}")]

    if model_spec.model_family == "ising_like":
        axis = model_spec.ising_axis
        return [
            (coupling_j * (1.0 + beta), f"S{axis}"),
            (coupling_j * (1.0 - beta), f"T{axis}"),
            (coupling_j * alpha, f"ST{axis}"),
        ]

    # Bond-dependent Yao-Lee-like channel.
    return [
        (coupling_j * (1.0 + beta), f"S{axis_gamma}"),
        (coupling_j * (1.0 - beta), f"T{axis_gamma}"),
        (coupling_j * alpha, f"ST{axis_gamma}"),
    ]


def _is_zero_coefficient(value: complex, tol: float = 1e-12) -> bool:
    return abs(complex(value)) <= tol


def _u1_pair_terms_for_bond_terms(
    bond_terms: List[Tuple[float, str]],
    i: int,
    j: int,
) -> List[Tuple[complex, str, int, str, int]]:
    """Convert U1-conserving x/y pairs into raising/lowering AutoMPO terms."""
    combined: Dict[str, complex] = {}
    order: List[str] = []
    for coefficient, op_name in bond_terms:
        op_text = str(op_name)
        coeff = complex(coefficient)
        if _is_zero_coefficient(coeff):
            continue
        if op_text not in combined:
            order.append(op_text)
            combined[op_text] = 0.0j
        combined[op_text] += coeff

    terms: List[Tuple[complex, str, int, str, int]] = []

    for x_op, y_op, plus_op, minus_op in (
        ("Sx", "Sy", "Sp", "Sm"),
        ("Tx", "Ty", "Tp", "Tm"),
    ):
        coeff_x = combined.pop(x_op, 0.0j)
        coeff_y = combined.pop(y_op, 0.0j)
        if _is_zero_coefficient(coeff_x) and _is_zero_coefficient(coeff_y):
            continue
        if _is_zero_coefficient(coeff_x - coeff_y):
            coeff = 0.5 * coeff_x
            terms.append((coeff, plus_op, i, minus_op, j))
            terms.append((coeff, minus_op, i, plus_op, j))
            continue
        raise ValueError(
            "A U1 symmetric MPO requires transverse pair terms to appear as "
            f"{x_op}_{i}{x_op}_{j} + {y_op}_{i}{y_op}_{j} with equal coefficients. "
            f"Got {x_op} coefficient {coeff_x:g} and {y_op} coefficient {coeff_y:g}. "
            "Use symmetry_mode=z2/none for single-axis x/y flip terms."
        )

    for op_name in ("STx", "STy"):
        coeff = combined.get(op_name, 0.0j)
        if not _is_zero_coefficient(coeff):
            raise ValueError(
                f"U1 symmetry cannot preserve the current {op_name}_{i}{op_name}_{j} "
                "term without changing the Hamiltonian. Use symmetry_mode=z2 or none for "
                "Yao-Lee x/y spin-orbital channels."
            )

    for op_name in order:
        coeff = combined.get(op_name, 0.0j)
        if _is_zero_coefficient(coeff):
            continue
        terms.append((coeff, op_name, i, op_name, j))
    return terms


def auto_mpo_pair_terms_for_bond_terms(
    bond_terms: List[Tuple[float, str]],
    i: int,
    j: int,
    *,
    symmetry_mode: str,
    strict_charge_conservation: bool,
) -> List[Tuple[Any, str, int, str, int]]:
    mode = _normalize_symmetry_mode(symmetry_mode)
    if mode == "u1":
        return list(_u1_pair_terms_for_bond_terms(bond_terms, i, j))
    return [(coefficient, op_name, i, op_name, j) for coefficient, op_name in bond_terms]


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

@dataclass
class Bond:
    i: int
    j: int
    gamma: str


@dataclass
class GeometryData:
    number_of_sites: int
    bond_list: List[Bond]
    positions: np.ndarray
    cell_indices: List[Tuple[int, int]]
    sublattice_indices: List[int]


def honeycomb_real_space_position(x_cell: int, y_cell: int, sublattice: int) -> np.ndarray:
    a = 1.0
    a1 = np.array([np.sqrt(3.0) * a, 0.0], dtype=float)
    a2 = np.array([np.sqrt(3.0) * a / 2.0, 3.0 * a / 2.0], dtype=float)
    delta_b = np.array([0.0, a], dtype=float)

    position = x_cell * a1 + y_cell * a2
    if sublattice == 1:
        position = position + delta_b
    return position


def snake_y_values(x_cell: int, circumference_y: int):
    if x_cell % 2 == 0:
        return range(circumference_y)
    return range(circumference_y - 1, -1, -1)


def build_honeycomb_cylinder_geometry(
    length_x: int,
    circumference_y: int,
    periodic_y: bool = True,
) -> GeometryData:
    bond_list: List[Bond] = []
    positions: List[np.ndarray] = []
    cell_indices: List[Tuple[int, int]] = []
    sublattice_indices: List[int] = []
    n_sites = length_x * circumference_y * 2
    site_to_index: Dict[Tuple[int, int, int], int] = {}

    for x in range(length_x):
        for y in snake_y_values(x, circumference_y):
            for sub in (0, 1):
                site_to_index[(x, y, sub)] = len(positions)
                positions.append(honeycomb_real_space_position(x, y, sub))
                cell_indices.append((x, y))
                sublattice_indices.append(sub)
    if len(positions) != n_sites:
        raise RuntimeError("Internal geometry error: honeycomb snake ordering generated wrong site count.")

    for x in range(length_x):
        for y in range(circumference_y):
            i_a = site_to_index[(x, y, 0)]
            i_b = site_to_index[(x, y, 1)]
            bond_list.append(Bond(i_a, i_b, "z"))

            y_plus_1 = (y + 1) % circumference_y
            if periodic_y or (y + 1 < circumference_y):
                j_b_y = site_to_index[(x, y_plus_1, 1)]
                bond_list.append(Bond(i_a, j_b_y, "y"))

            if x + 1 < length_x:
                j_b_x = site_to_index[(x + 1, y, 1)]
                bond_list.append(Bond(i_a, j_b_x, "x"))

    return GeometryData(
        number_of_sites=n_sites,
        bond_list=bond_list,
        positions=np.asarray(positions, dtype=float),
        cell_indices=cell_indices,
        sublattice_indices=sublattice_indices,
    )


def square_real_space_position(x_cell: int, y_cell: int) -> np.ndarray:
    return np.asarray([float(x_cell), float(y_cell)], dtype=float)


def build_square_cylinder_geometry(
    length_x: int,
    circumference_y: int,
    periodic_y: bool = True,
) -> GeometryData:
    bond_list: List[Bond] = []
    positions: List[np.ndarray] = []
    cell_indices: List[Tuple[int, int]] = []
    sublattice_indices: List[int] = []
    n_sites = length_x * circumference_y
    site_to_index: Dict[Tuple[int, int], int] = {}

    for x in range(length_x):
        for y in snake_y_values(x, circumference_y):
            site_to_index[(x, y)] = len(positions)
            positions.append(square_real_space_position(x, y))
            cell_indices.append((x, y))
            sublattice_indices.append(0)
    if len(positions) != n_sites:
        raise RuntimeError("Internal geometry error: square snake ordering generated wrong site count.")

    for x in range(length_x):
        for y in range(circumference_y):
            i_site = site_to_index[(x, y)]
            if x + 1 < length_x:
                j_x = site_to_index[(x + 1, y)]
                bond_list.append(Bond(i_site, j_x, "x"))

            y_plus_1 = (y + 1) % circumference_y
            if periodic_y or (y + 1 < circumference_y):
                j_y = site_to_index[(x, y_plus_1)]
                if j_y != i_site:
                    bond_list.append(Bond(i_site, j_y, "y"))

    return GeometryData(
        number_of_sites=n_sites,
        bond_list=bond_list,
        positions=np.asarray(positions, dtype=float),
        cell_indices=cell_indices,
        sublattice_indices=sublattice_indices,
    )


def triangular_bravais_vectors() -> Tuple[np.ndarray, np.ndarray]:
    a = 1.0
    a1 = np.array([np.sqrt(3.0) * a, 0.0], dtype=float)
    a2 = np.array([np.sqrt(3.0) * a / 2.0, 3.0 * a / 2.0], dtype=float)
    return a1, a2


def triangular_real_space_position(x_cell: int, y_cell: int) -> np.ndarray:
    a1, a2 = triangular_bravais_vectors()
    return x_cell * a1 + y_cell * a2


def build_triangular_cylinder_geometry(
    length_x: int,
    circumference_y: int,
    periodic_y: bool = True,
) -> GeometryData:
    bond_list: List[Bond] = []
    positions: List[np.ndarray] = []
    cell_indices: List[Tuple[int, int]] = []
    sublattice_indices: List[int] = []
    n_sites = length_x * circumference_y
    site_to_index: Dict[Tuple[int, int], int] = {}

    for x in range(length_x):
        for y in snake_y_values(x, circumference_y):
            site_to_index[(x, y)] = len(positions)
            positions.append(triangular_real_space_position(x, y))
            cell_indices.append((x, y))
            sublattice_indices.append(0)
    if len(positions) != n_sites:
        raise RuntimeError("Internal geometry error: triangular snake ordering generated wrong site count.")

    for x in range(length_x):
        for y in range(circumference_y):
            i_site = site_to_index[(x, y)]

            if x + 1 < length_x:
                j_x = site_to_index[(x + 1, y)]
                bond_list.append(Bond(i_site, j_x, "x"))

            y_plus_1 = (y + 1) % circumference_y
            if periodic_y or (y + 1 < circumference_y):
                j_y = site_to_index[(x, y_plus_1)]
                if j_y != i_site:
                    bond_list.append(Bond(i_site, j_y, "y"))

            if x + 1 < length_x:
                y_minus_1 = (y - 1) % circumference_y
                if periodic_y or (y - 1 >= 0):
                    j_z = site_to_index[(x + 1, y_minus_1)]
                    bond_list.append(Bond(i_site, j_z, "z"))

    return GeometryData(
        number_of_sites=n_sites,
        bond_list=bond_list,
        positions=np.asarray(positions, dtype=float),
        cell_indices=cell_indices,
        sublattice_indices=sublattice_indices,
    )


def build_lattice_geometry(
    lattice: str,
    length_x: int,
    circumference_y: int,
    periodic_y: bool = True,
) -> GeometryData:
    lattice_name = lattice.lower()
    if lattice_name == "honeycomb":
        return build_honeycomb_cylinder_geometry(length_x, circumference_y, periodic_y)
    if lattice_name == "square":
        return build_square_cylinder_geometry(length_x, circumference_y, periodic_y)
    if lattice_name == "triangular":
        return build_triangular_cylinder_geometry(length_x, circumference_y, periodic_y)
    raise ValueError(f"Unsupported lattice '{lattice}'. Choose from: honeycomb, square, triangular.")


# ----------------------------------------------------------------------
# DMRG
# ----------------------------------------------------------------------

def _build_auto_mpo_from_terms(
    terms: List[Tuple[Any, ...]],
    length: int,
    site_ops: Dict[str, np.ndarray],
    *,
    symmetry_mode: str = "none",
    phys_charges: np.ndarray | None = None,
    strict_charge_conservation: bool = True,
) -> Any:
    api = get_tenax_api()
    build_auto_mpo = api["build_auto_mpo"]
    auto_mpo_cls = api["AutoMPO"]
    mode = _normalize_symmetry_mode(symmetry_mode)
    use_symmetric_tensors = (mode != "none")

    if use_symmetric_tensors:
        if phys_charges is None:
            raise ValueError(f"{mode.upper()} symmetric MPO build requires explicit phys_charges.")
        if strict_charge_conservation:
            _validate_symmetry_conserving_terms(terms, site_ops, phys_charges, mode)

    if build_auto_mpo is not None:
        signature = inspect.signature(build_auto_mpo)
        kwargs: Dict[str, Any] = {"L": length}
        local_dim = int(next(iter(site_ops.values())).shape[0])
        build_fn_supports_symmetry = (
            (not use_symmetric_tensors)
            or (
                "symmetric" in signature.parameters
                and "phys_charges" in signature.parameters
            )
        )
        if build_fn_supports_symmetry:
            if "d" in signature.parameters:
                kwargs["d"] = local_dim
            if "site_ops" in signature.parameters:
                kwargs["site_ops"] = site_ops
            # Our local operators include Sy/Ty, so terms are complex.
            # Force complex MPO dtype to avoid float-casting errors inside Tenax.
            if "dtype" in signature.parameters:
                kwargs["dtype"] = np.complex128
            if use_symmetric_tensors:
                kwargs["symmetric"] = True
                kwargs["phys_charges"] = np.asarray(phys_charges, dtype=np.int32)
            return build_auto_mpo(terms, **kwargs)
        if auto_mpo_cls is None:
            raise RuntimeError(
                "The installed Tenax build_auto_mpo does not expose symmetric/phys_charges "
                "arguments, and AutoMPO fallback is unavailable for U1/Z2 construction."
            )

    if auto_mpo_cls is not None:
        local_dim = int(next(iter(site_ops.values())).shape[0])
        try:
            auto = auto_mpo_cls(L=length, d=local_dim)
        except TypeError:
            auto = auto_mpo_cls(length)
        for term in terms:
            auto += term
        try:
            if use_symmetric_tensors:
                return auto.to_mpo(
                    compress=True,
                    symmetric=True,
                    phys_charges=np.asarray(phys_charges, dtype=np.int32),
                    dtype=np.complex128,
                )
            return auto.to_mpo(compress=True)
        except TypeError:
            if use_symmetric_tensors:
                return auto.to_mpo(
                    symmetric=True,
                    phys_charges=np.asarray(phys_charges, dtype=np.int32),
                    dtype=np.complex128,
                )
            return auto.to_mpo()

    raise RuntimeError("Tenax provides neither build_auto_mpo nor AutoMPO.")


def build_tenax_yao_lee_mpo(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    symmetry_mode: str = "none",
    symmetry_phys_charges: np.ndarray | None = None,
    strict_charge_conservation: bool = True,
    show_progress: bool = True,
) -> Any:
    length = geometry.number_of_sites
    custom_ops = build_site_ops(model_spec)
    terms: List[Tuple[Any, ...]] = []

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(geometry.bond_list),
        desc="Tenax MPO bonds",
        unit="bond",
        leave=False,
    )
    for bond in geometry.bond_list:
        i, j, gamma = bond.i, bond.j, bond.gamma.lower()
        bond_terms = model_terms_for_bond(gamma, model_spec, alpha, beta, coupling_j)
        terms.extend(
            auto_mpo_pair_terms_for_bond_terms(
                bond_terms,
                i,
                j,
                symmetry_mode=symmetry_mode,
                strict_charge_conservation=bool(strict_charge_conservation),
            )
        )
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    return _build_auto_mpo_from_terms(
        terms,
        length,
        custom_ops,
        symmetry_mode=symmetry_mode,
        phys_charges=symmetry_phys_charges,
        strict_charge_conservation=bool(strict_charge_conservation),
    )


def _extract_dmrg_result(result: Any, initial_mps: Any) -> Tuple[Any, Dict[str, Any]]:
    mps_out = initial_mps
    energy = None
    converged = None
    energies_per_sweep = None

    if isinstance(result, dict):
        mps_out = result.get("mps", result.get("state", initial_mps))
        energy = result.get("energy", result.get("E", None))
        converged = result.get("converged", None)
        energies_per_sweep = result.get("energies_per_sweep", None)
    elif hasattr(result, "energy") or hasattr(result, "E"):
        mps_out = getattr(result, "mps", getattr(result, "state", initial_mps))
        energy = getattr(result, "energy", getattr(result, "E", None))
        converged = getattr(result, "converged", None)
        energies_per_sweep = getattr(result, "energies_per_sweep", None)
    elif isinstance(result, tuple):
        for item in result:
            if (
                hasattr(item, "n_nodes")
                and hasattr(item, "get_tensor")
            ) or hasattr(item, "tensors"):
                mps_out = item
                break
        if mps_out is initial_mps:
            for item in result:
                if hasattr(item, "expectation_value"):
                    mps_out = item
                    break
        for item in result:
            if isinstance(item, (int, float, np.floating)):
                energy = float(item)
                break
    else:
        mps_out = getattr(result, "mps", getattr(result, "state", initial_mps))
        energy = getattr(result, "energy", getattr(result, "E", None))
        converged = getattr(result, "converged", None)
        energies_per_sweep = getattr(result, "energies_per_sweep", None)

    if energy is None:
        raise RuntimeError("Could not read ground-state energy from Tenax dmrg result.")

    info = {"E": float(energy), "converged": converged}
    if energies_per_sweep is not None:
        energies = [float(val) for val in list(energies_per_sweep)]
        info["energies_per_sweep"] = energies
        info["sweeps_done"] = len(energies)
    return mps_out, info


class _TenaxSweepProgressStream(io.TextIOBase):
    _SWEEP_PATTERN = re.compile(r"Sweep\s+(\d+)\s*/\s*(\d+)\s*:\s*E\s*=\s*([-\d.eE+]+)")

    def __init__(self, original_stream: Any, progress_bar: Any):
        self._original_stream = original_stream
        self._progress_bar = progress_bar
        self._buffer = ""
        self._last_sweep = 0

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._original_stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(line.strip())
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._process_line(self._buffer.strip())
            self._buffer = ""
        self._original_stream.flush()

    def _process_line(self, line: str) -> None:
        if not line:
            return
        match = self._SWEEP_PATTERN.search(line)
        if match is None:
            return
        sweep_idx = int(match.group(1))
        sweep_total = int(match.group(2))
        sweep_energy = match.group(3)

        if self._progress_bar.total != sweep_total:
            self._progress_bar.total = sweep_total
        if sweep_idx > self._last_sweep:
            self._progress_bar.update(sweep_idx - self._last_sweep)
            self._last_sweep = sweep_idx
        self._progress_bar.set_postfix({"E": sweep_energy})


def _build_random_symmetric_mps(
    length: int,
    physical_dim: int,
    bond_dim: int,
    seed: int,
    phys_charges: np.ndarray,
    target_charge: int = 0,
    symmetry_mode: str = "u1",
) -> Any:
    """Construct a random U1/Z2 symmetry-adapted FiniteMPS with custom physical charges."""
    import jax
    from tenax import FiniteMPS
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import SymmetricTensor

    if len(phys_charges) != physical_dim:
        raise ValueError(
            f"phys_charges length {len(phys_charges)} does not match physical_dim={physical_dim}."
        )

    mode = _normalize_symmetry_mode(symmetry_mode)
    if mode == "u1":
        symmetry = U1Symmetry()
        charge_modulus = None
        target_charge = int(target_charge)
        phys = np.asarray(phys_charges, dtype=np.int32)
    elif mode == "z2":
        symmetry = _get_z2_symmetry_object()
        charge_modulus = 2
        target_charge = int(target_charge) % 2
        phys = np.asarray(phys_charges, dtype=np.int32) % 2
    else:
        raise ValueError(f"Symmetric MPS builder requires symmetry mode u1/z2, got '{mode}'.")

    key = jax.random.PRNGKey(int(seed))
    reachable = {0}
    for _ in range(max(length - 1, 1)):
        next_reachable = set()
        for charge_left in reachable:
            for phys_q in phys:
                next_charge = int(charge_left + int(phys_q))
                if charge_modulus is not None:
                    next_charge %= charge_modulus
                next_reachable.add(next_charge)
        reachable = next_reachable
        if len(reachable) > 8 * max(4, int(bond_dim)):
            sorted_charges = sorted(reachable, key=lambda q: (abs(q - target_charge), abs(q)))
            reachable = set(sorted_charges[: 8 * max(4, int(bond_dim))])

    required = sorted(set([0, int(target_charge)] + [int(q) for q in phys] + list(reachable)))
    if len(required) > max(2, int(bond_dim)):
        required = sorted(required, key=lambda q: (abs(q - target_charge), abs(q)))[: int(bond_dim)]
        required = sorted(set(required + [0, int(target_charge)]))
    if charge_modulus is not None:
        required = sorted({int(q % charge_modulus) for q in required})
    virt_charges = np.asarray(required, dtype=np.int32)
    if virt_charges.size == 0:
        virt_charges = np.asarray([0], dtype=np.int32)

    tensors: List[Any] = []
    for site in range(length):
        key, subkey = jax.random.split(key)
        site_target = int(target_charge) if site == length - 1 else None
        if length == 1:
            left = np.asarray([0], dtype=np.int32)
            right = np.asarray([0], dtype=np.int32)
        elif site == 0:
            left = np.asarray([0], dtype=np.int32)
            right = virt_charges
        elif site == length - 1:
            left = virt_charges
            right = np.asarray([0], dtype=np.int32)
        else:
            left = virt_charges
            right = virt_charges

        left_label = "v_left_0" if site == 0 else f"v{site - 1}_{site}"
        right_label = "v_right" if site == length - 1 else f"v{site}_{site + 1}"
        indices = (
            TensorIndex(symmetry, left, FlowDirection.IN, label=left_label),
            TensorIndex(symmetry, phys, FlowDirection.IN, label=f"p_{site}"),
            TensorIndex(symmetry, right, FlowDirection.OUT, label=right_label),
        )
        tensor = SymmetricTensor.random_normal(
            indices,
            key=subkey,
            dtype=np.complex128,
            target=site_target,
        )
        tensors.append(tensor)

    return FiniteMPS.from_tensors(tensors, target_charge=int(target_charge)).right_canonicalize()


def run_tenax_cylindrical_dmrg(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    max_bond_dimension: int,
    max_sweeps: int,
    random_seed: int,
    symmetry_mode: str = SYMMETRY_MODE,
    u1_target_total_sz2: int = U1_TARGET_TOTAL_SZ2,
    u1_target_total_tz2: int = U1_TARGET_TOTAL_TZ2,
    z2_target_parity: int = Z2_TARGET_PARITY,
    strict_symmetry_selection_rules: bool = STRICT_SYMMETRY_SELECTION_RULES,
    show_progress: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    np.random.seed(random_seed)
    api = get_tenax_api()
    n_sites = geometry.number_of_sites

    stage_start = _start_stage("Tenax MPO+DMRG", show_progress)
    sweep_bar = _make_progress_bar(
        enabled=show_progress,
        total=max_sweeps,
        desc="Tenax sweeps",
        unit="sweep",
        leave=False,
    )
    sym_mode = _normalize_symmetry_mode(symmetry_mode)
    symmetry_enabled = (sym_mode != "none")
    symmetry_phys_charges = None
    symmetry_target_charge = None
    symmetry_basis_table = None
    try:
        if sym_mode == "u1":
            symmetry_phys_charges = _u1_encoded_phys_charges_for_model(model_spec)
            symmetry_target_charge = _u1_encoded_target_charge(
                total_sz2=int(u1_target_total_sz2),
                total_tz2=int(u1_target_total_tz2),
            )
            symmetry_basis_table = _u1_basis_charge_table_for_model(model_spec)
        elif sym_mode == "z2":
            symmetry_phys_charges = _z2_phys_charges_for_model(model_spec)
            symmetry_target_charge = int(z2_target_parity) % 2
            symmetry_basis_table = _z2_basis_charge_table_for_model(model_spec)

        mpo = build_tenax_yao_lee_mpo(
            geometry,
            model_spec,
            alpha,
            beta,
            coupling_j,
            symmetry_mode=sym_mode,
            symmetry_phys_charges=symmetry_phys_charges,
            strict_charge_conservation=bool(strict_symmetry_selection_rules),
            show_progress=show_progress,
        )
        if symmetry_enabled:
            mps = _build_random_symmetric_mps(
                length=n_sites,
                physical_dim=model_spec.physical_dim,
                bond_dim=min(16, max_bond_dimension),
                seed=int(random_seed),
                phys_charges=np.asarray(symmetry_phys_charges, dtype=np.int32),
                target_charge=int(symmetry_target_charge),
                symmetry_mode=sym_mode,
            )
        else:
            mps = api["build_random_mps"](
                n_sites,
                physical_dim=model_spec.physical_dim,
                bond_dim=min(16, max_bond_dimension),
            )
        config_kwargs: Dict[str, Any] = {
            "max_bond_dim": max_bond_dimension,
            "num_sweeps": max_sweeps,
            "verbose": bool(show_progress),
        }
        config_signature = inspect.signature(api["DMRGConfig"])
        if symmetry_enabled and "target_charge" in config_signature.parameters:
            config_kwargs["target_charge"] = int(symmetry_target_charge)
        config = api["DMRGConfig"](**config_kwargs)
        if sweep_bar is not None:
            sweep_stdout_proxy = _TenaxSweepProgressStream(sys.stdout, sweep_bar)
            with contextlib.redirect_stdout(sweep_stdout_proxy):
                result = api["dmrg"](mpo, mps, config)
            sweep_stdout_proxy.flush()
        else:
            result = api["dmrg"](mpo, mps, config)
    except Exception as exc:
        if sweep_bar is not None:
            sweep_bar.close()
        if show_progress:
            elapsed = time.perf_counter() - stage_start
            print(f"[stage] Tenax MPO+DMRG failed in {elapsed:.2f}s: {exc}")
        raise

    mps_out, dmrg_info = _extract_dmrg_result(result, mps)
    dmrg_info["symmetry_mode"] = sym_mode
    dmrg_info["symmetry_enabled"] = bool(symmetry_enabled)
    dmrg_info["u1_symmetry_enabled"] = bool(sym_mode == "u1")
    dmrg_info["z2_symmetry_enabled"] = bool(sym_mode == "z2")
    if symmetry_enabled and symmetry_phys_charges is not None:
        dmrg_info["symmetry_phys_charges"] = [
            int(val) for val in list(np.asarray(symmetry_phys_charges, dtype=np.int32))
        ]
        dmrg_info["symmetry_target_charge"] = int(symmetry_target_charge)
        if symmetry_basis_table is not None:
            dmrg_info["symmetry_basis_charge_table"] = symmetry_basis_table
        if sym_mode == "u1":
            dmrg_info["u1_target_sector"] = {
                "total_Sz_times_2": int(u1_target_total_sz2),
                "total_Tz_times_2": int(u1_target_total_tz2),
            }
        if sym_mode == "z2":
            dmrg_info["z2_target_sector"] = {"global_parity": int(z2_target_parity) % 2}
    if sweep_bar is not None:
        sweeps_done = int(dmrg_info.get("sweeps_done", 0) or 0)
        if sweeps_done > sweep_bar.n:
            sweep_bar.update(sweeps_done - sweep_bar.n)
        if "E" in dmrg_info:
            sweep_bar.set_postfix({"E": f"{float(dmrg_info['E']):.10f}"})
        sweep_bar.close()
    _end_stage("Tenax MPO+DMRG", stage_start, show_progress)
    return mps_out, mpo, dmrg_info


class _TenaxIDMRGSweepProgressStream(io.TextIOBase):
    _SWEEP_PATTERN = re.compile(r"iDMRG sweep\s+(\d+):.*e/site=([-\d.eE+]+)")
    _ENV_PATTERN = re.compile(r"Env warmup\s+(\d+)\s*/\s*(\d+)")

    def __init__(self, original_stream: Any, sweep_bar: Any):
        self._original_stream = original_stream
        self._sweep_bar = sweep_bar
        self._buffer = ""
        self._last_sweep = 0
        self._env_bar = None

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._original_stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(line.strip())
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._process_line(self._buffer.strip())
            self._buffer = ""
        self._original_stream.flush()

    def close_progress(self) -> None:
        if self._env_bar is not None:
            self._env_bar.close()
            self._env_bar = None

    def _process_line(self, line: str) -> None:
        if not line:
            return
        sweep_match = self._SWEEP_PATTERN.search(line)
        if sweep_match is not None:
            sweep_idx = int(sweep_match.group(1))
            sweep_energy = sweep_match.group(2)
            if sweep_idx > self._last_sweep:
                self._sweep_bar.update(sweep_idx - self._last_sweep)
                self._last_sweep = sweep_idx
            self._sweep_bar.set_postfix({"e/site": sweep_energy})
            return

        env_match = self._ENV_PATTERN.search(line)
        if env_match is None:
            return
        env_idx = int(env_match.group(1))
        env_total = int(env_match.group(2))
        if self._env_bar is None:
            tqdm = _get_tqdm(True)
            if tqdm is not None:
                self._env_bar = tqdm(
                    total=env_total,
                    desc="Tenax iDMRG env warmup",
                    unit="step",
                    dynamic_ncols=True,
                    leave=False,
                )
        if self._env_bar is not None:
            self._env_bar.n = min(env_idx, env_total)
            self._env_bar.refresh()
            if env_idx >= env_total:
                self._env_bar.close()
                self._env_bar = None


def _build_dense_bulk_mpo_tensor(data: np.ndarray) -> Any:
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor
    import jax.numpy as jnp

    if data.ndim != 4:
        raise ValueError(f"bulk MPO tensor must have rank-4, got shape {data.shape}")
    d_w_l, d_top, d_bot, d_w_r = data.shape
    if d_w_l != d_w_r:
        raise ValueError(f"bulk MPO must have equal virtual dimensions, got left={d_w_l}, right={d_w_r}")
    if d_top != d_bot:
        raise ValueError(f"bulk MPO physical bra/ket dims must match, got {d_top} and {d_bot}")

    sym = U1Symmetry()
    indices = (
        TensorIndex(sym, np.zeros(d_w_l, dtype=np.int32), FlowDirection.IN, label="w_l"),
        TensorIndex(sym, np.zeros(d_top, dtype=np.int32), FlowDirection.IN, label="mpo_top"),
        TensorIndex(sym, np.zeros(d_bot, dtype=np.int32), FlowDirection.OUT, label="mpo_bot"),
        TensorIndex(sym, np.zeros(d_w_r, dtype=np.int32), FlowDirection.OUT, label="w_r"),
    )
    return DenseTensor(jnp.asarray(data), indices)


def build_idmrg_bulk_mpo_from_finite_mpo(
    mpo: Any,
    model_spec: ModelSpec,
    bulk_kind: str = "auto",
    max_local_dim: int = 256,
    show_progress: bool = True,
) -> Tuple[Any, int, int, Dict[str, Any]]:
    if bulk_kind not in ("auto", "pair", "single"):
        raise ValueError("bulk_kind must be one of: auto, pair, single.")

    local_dim = int(model_spec.physical_dim)
    n_sites = int(mpo.n_nodes()) if hasattr(mpo, "n_nodes") else int(len(getattr(mpo, "tensors", [])))
    if n_sites < 2:
        raise RuntimeError(f"Need at least 2 MPO sites to build iDMRG bulk tensor, got n_sites={n_sites}.")

    tensors_dense: List[np.ndarray] = []
    scan_bar = _make_progress_bar(
        enabled=show_progress,
        total=n_sites,
        desc="iDMRG bulk scan",
        unit="site",
        leave=False,
    )
    for site in range(n_sites):
        tensor_array = np.asarray(mpo.get_tensor(site).todense())
        tensors_dense.append(tensor_array)
        if scan_bar is not None:
            scan_bar.update(1)
    if scan_bar is not None:
        scan_bar.close()

    center = 0.5 * (n_sites - 1)
    single_candidates: List[Tuple[float, int]] = []
    pair_candidates: List[Tuple[float, int]] = []

    for idx, tensor_array in enumerate(tensors_dense):
        d_w_l, d_top, d_bot, d_w_r = tensor_array.shape
        if d_top == local_dim and d_bot == local_dim and d_w_l == d_w_r:
            single_candidates.append((abs(idx - center), idx))

    for idx in range(n_sites - 1):
        left_tensor = tensors_dense[idx]
        right_tensor = tensors_dense[idx + 1]
        if (
            left_tensor.shape[1] == local_dim
            and left_tensor.shape[2] == local_dim
            and right_tensor.shape[1] == local_dim
            and right_tensor.shape[2] == local_dim
            and left_tensor.shape[3] == right_tensor.shape[0]
            and left_tensor.shape[0] == right_tensor.shape[3]
        ):
            pair_center = idx + 0.5
            pair_candidates.append((abs(pair_center - center), idx))

    diagnostics: Dict[str, Any] = {
        "bulk_kind_requested": bulk_kind,
        "n_sites_finite_mpo": n_sites,
        "single_candidates": [index for _, index in sorted(single_candidates, key=lambda item: item[0])],
        "pair_candidates": [index for _, index in sorted(pair_candidates, key=lambda item: item[0])],
        "local_dim_original_site": local_dim,
    }

    preferred_modes: List[str]
    if bulk_kind == "auto":
        preferred_modes = ["pair", "single"]
    else:
        preferred_modes = [bulk_kind]

    for mode in preferred_modes:
        if mode == "pair":
            if len(pair_candidates) == 0:
                continue
            effective_local_dim = local_dim * local_dim
            if effective_local_dim > max_local_dim:
                diagnostics["pair_rejected_reason"] = (
                    f"effective local dim {effective_local_dim} exceeds limit {max_local_dim}"
                )
                continue
            _, pair_index = min(pair_candidates, key=lambda item: item[0])
            left_tensor = tensors_dense[pair_index]
            right_tensor = tensors_dense[pair_index + 1]
            pair_data = np.einsum("asub,btvc->astuvc", left_tensor, right_tensor, optimize=True)
            d_w_l = left_tensor.shape[0]
            d_w_r = right_tensor.shape[3]
            pair_data = pair_data.reshape(d_w_l, effective_local_dim, effective_local_dim, d_w_r)
            bulk_mpo = _build_dense_bulk_mpo_tensor(pair_data)
            diagnostics.update(
                {
                    "bulk_kind_used": "pair",
                    "bulk_pair_start_site": int(pair_index),
                    "sites_per_idmrg_site": 2,
                    "effective_local_dim": int(effective_local_dim),
                    "bulk_virtual_dim": int(d_w_l),
                }
            )
            return bulk_mpo, effective_local_dim, 2, diagnostics

        if mode == "single":
            if len(single_candidates) == 0:
                continue
            if local_dim > max_local_dim:
                diagnostics["single_rejected_reason"] = (
                    f"effective local dim {local_dim} exceeds limit {max_local_dim}"
                )
                continue
            _, single_index = min(single_candidates, key=lambda item: item[0])
            single_data = tensors_dense[single_index]
            bulk_mpo = _build_dense_bulk_mpo_tensor(single_data)
            diagnostics.update(
                {
                    "bulk_kind_used": "single",
                    "bulk_single_site": int(single_index),
                    "sites_per_idmrg_site": 1,
                    "effective_local_dim": int(local_dim),
                    "bulk_virtual_dim": int(single_data.shape[0]),
                }
            )
            return bulk_mpo, local_dim, 1, diagnostics

    raise RuntimeError(
        "Could not construct a valid bulk MPO tensor for iDMRG from finite MPO. "
        f"Diagnostics: {diagnostics}"
    )


def run_tenax_idmrg_x_from_finite_mpo(
    mpo: Any,
    model_spec: ModelSpec,
    max_bond_dimension: int,
    max_iterations: int,
    bulk_kind: str = "auto",
    max_local_dim: int = 256,
    show_progress: bool = True,
) -> Dict[str, Any]:
    api = get_tenax_api()
    idmrg_fn = api.get("idmrg", None)
    idmrg_config_cls = api.get("iDMRGConfig", None)
    if not callable(idmrg_fn) or idmrg_config_cls is None:
        raise RuntimeError("Tenax iDMRG API is unavailable in the installed Tenax package.")

    stage_start = _start_stage("Tenax iDMRG-x", show_progress)
    sweep_bar = _make_progress_bar(
        enabled=show_progress,
        total=max_iterations,
        desc="Tenax iDMRG sweeps",
        unit="iter",
        leave=False,
    )
    try:
        bulk_mpo, effective_local_dim, sites_per_idmrg_site, diagnostics = build_idmrg_bulk_mpo_from_finite_mpo(
            mpo=mpo,
            model_spec=model_spec,
            bulk_kind=bulk_kind,
            max_local_dim=max_local_dim,
            show_progress=show_progress,
        )
        config = idmrg_config_cls(
            max_bond_dim=max_bond_dimension,
            max_iterations=max_iterations,
            verbose=bool(show_progress),
        )
        if sweep_bar is not None:
            sweep_stdout_proxy = _TenaxIDMRGSweepProgressStream(sys.stdout, sweep_bar)
            with contextlib.redirect_stdout(sweep_stdout_proxy):
                result = idmrg_fn(bulk_mpo, config, d=effective_local_dim, dtype=np.complex128)
            sweep_stdout_proxy.flush()
            sweep_stdout_proxy.close_progress()
        else:
            result = idmrg_fn(bulk_mpo, config, d=effective_local_dim, dtype=np.complex128)
    except Exception:
        if sweep_bar is not None:
            sweep_bar.close()
        raise

    energies_per_step_native = [float(value) for value in list(getattr(result, "energies_per_step", []))]
    energy_per_idmrg_site = float(getattr(result, "energy_per_site"))
    energy_per_original_site = energy_per_idmrg_site / float(sites_per_idmrg_site)
    energies_per_step_original_site = [
        float(value) / float(sites_per_idmrg_site) for value in energies_per_step_native
    ]
    converged = bool(getattr(result, "converged", False))

    if sweep_bar is not None:
        steps_done = len(energies_per_step_native)
        if steps_done > sweep_bar.n:
            sweep_bar.update(steps_done - sweep_bar.n)
        sweep_bar.set_postfix({"e/site": f"{energy_per_original_site:.10f}"})
        sweep_bar.close()

    _end_stage("Tenax iDMRG-x", stage_start, show_progress)
    entanglement_profile = None
    entanglement_warning = None
    try:
        entanglement_profile = compute_tenax_infinite_mps_entropy_profile(
            mps=getattr(result, "mps"),
            sites_per_idmrg_site=sites_per_idmrg_site,
            orders=ENTROPY_ORDERS,
        )
    except Exception as exc:
        entanglement_warning = f"Failed to compute iDMRG entanglement profile: {exc}"

    output = {
        "status": "completed",
        "method_note": (
            "iDMRG-x uses a bulk MPO extracted from the finite-MPO snake-path representation "
            "(single-site or two-site coarse-grained mapping)."
        ),
        "converged": converged,
        "iterations_done": len(energies_per_step_native),
        "energy_per_idmrg_site": energy_per_idmrg_site,
        "energy_per_original_site": energy_per_original_site,
        "energies_per_step_idmrg_site": energies_per_step_native,
        "energies_per_step_original_site": energies_per_step_original_site,
        "sites_per_idmrg_site": int(sites_per_idmrg_site),
        "effective_local_dim": int(effective_local_dim),
        "bulk_construction": diagnostics,
    }
    if entanglement_profile is not None:
        output["entanglement"] = entanglement_profile
    if entanglement_warning is not None:
        output["entanglement_warning"] = entanglement_warning
    return output


def evaluate_expectation_value(mpo_ij: Any, mps: Any) -> complex:
    api = get_tenax_api()
    exp_fn = api["expectation"]
    if callable(exp_fn):
        return complex(exp_fn(mpo_ij, mps))
    if hasattr(mps, "expectation_value"):
        return complex(mps.expectation_value(mpo_ij))
    if hasattr(mps, "expectation"):
        return complex(mps.expectation(mpo_ij))
    raise RuntimeError(
        "No expectation evaluator found. Tenax must provide expectation(...) or MPS expectation methods."
    )


def collect_correlation_matrices_from_tenax(
    mps: Any,
    geometry: GeometryData,
    model_spec: ModelSpec,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    n_sites = geometry.number_of_sites
    custom_ops = build_site_ops(model_spec)
    api = get_tenax_api()
    corr_fn = api.get("correlation", None)
    op_pairs = [
        ("Sx", "Sx"), ("Sy", "Sy"), ("Sz", "Sz"),
        ("Tx", "Tx"), ("Ty", "Ty"), ("Tz", "Tz"),
        ("STx", "STx"), ("STy", "STy"), ("STz", "STz"),
    ]
    correlations = {f"{op1}_{op2}": np.zeros((n_sites, n_sites), dtype=complex) for op1, op2 in op_pairs}

    pair_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=(n_sites * (n_sites - 1)) * len(op_pairs),
        desc="Tenax correlations",
        unit="pair",
        leave=False,
    )
    row_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=n_sites,
        desc="Tenax corr rows",
        unit="row",
        leave=False,
    )
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            for op1, op2 in op_pairs:
                if callable(corr_fn):
                    correlations[f"{op1}_{op2}"][i, j] = corr_fn(
                        mps,
                        custom_ops[op1],
                        i,
                        custom_ops[op2],
                        j,
                    )
                else:
                    mpo_ij = _build_auto_mpo_from_terms([(1.0, op1, i, op2, j)], n_sites, custom_ops)
                    correlations[f"{op1}_{op2}"][i, j] = evaluate_expectation_value(mpo_ij, mps)
                if pair_progress_bar is not None:
                    pair_progress_bar.update(1)
        if row_progress_bar is not None:
            row_progress_bar.update(1)

    if pair_progress_bar is not None:
        pair_progress_bar.close()
    if row_progress_bar is not None:
        row_progress_bar.close()

    return correlations


# ----------------------------------------------------------------------
# Exact diagonalization
# ----------------------------------------------------------------------

def kron_all(op_list: List[sparse.spmatrix]) -> sparse.spmatrix:
    out = op_list[0]
    for op in op_list[1:]:
        out = sparse.kron(out, op, format="csr")
    return out


def build_global_operator_cache() -> Dict[str, sparse.spmatrix]:
    default_spec = build_model_spec(
        spin_rep=SPIN_REP,
        orbital_rep=ORBITAL_REP,
        model_family=MODEL_FAMILY,
        ising_axis=ISING_AXIS,
    )
    ops = build_site_ops(default_spec)
    return {name: sparse.csr_matrix(mat) for name, mat in ops.items()}


def build_global_operator_cache_for_model(model_spec: ModelSpec) -> Dict[str, sparse.spmatrix]:
    ops = build_site_ops(model_spec)
    return {name: sparse.csr_matrix(mat) for name, mat in ops.items()}


def build_exact_hamiltonian(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    show_progress: bool = True,
) -> sparse.spmatrix:
    n_sites = geometry.number_of_sites
    op_cache = build_global_operator_cache_for_model(model_spec)
    ident = op_cache["Id"]
    local_dim = int(ident.shape[0])
    h_exact = sparse.csr_matrix((local_dim ** n_sites, local_dim ** n_sites), dtype=complex)

    bond_terms: List[Tuple[Any, List[Tuple[complex, str]]]] = []
    total_terms = 0
    for bond in geometry.bond_list:
        terms = list(model_terms_for_bond(bond.gamma.lower(), model_spec, alpha, beta, coupling_j))
        bond_terms.append((bond, terms))
        total_terms += len(terms)

    bond_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(bond_terms),
        desc="ED H bonds",
        unit="bond",
        leave=False,
    )
    term_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=total_terms,
        desc="ED H terms",
        unit="term",
        leave=False,
    )
    for bond, terms in bond_terms:
        i, j = bond.i, bond.j
        for coeff, op_name in terms:
            op_list = [ident] * n_sites
            op_list[i] = op_cache[op_name]
            op_list[j] = op_cache[op_name]
            h_exact = h_exact + coeff * kron_all(op_list)
            if term_progress_bar is not None:
                term_progress_bar.update(1)
        if bond_progress_bar is not None:
            bond_progress_bar.update(1)

    if term_progress_bar is not None:
        term_progress_bar.close()
    if bond_progress_bar is not None:
        bond_progress_bar.close()

    return h_exact


def run_small_cluster_exact_diagonalization(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    show_progress: bool = True,
) -> Tuple[float, np.ndarray]:
    stage_start = _start_stage("ED diagonalization", show_progress)
    hamiltonian = build_exact_hamiltonian(
        geometry,
        model_spec,
        alpha,
        beta,
        coupling_j,
        show_progress=show_progress,
    )
    if show_progress:
        print(
            f"[ed] sparse eigensolve started: dim={hamiltonian.shape[0]}, nnz={hamiltonian.nnz}, target=ground_state"
        )
    eigenvalues, eigenvectors = sparse_linalg.eigsh(hamiltonian, k=1, which="SA")
    _end_stage("ED diagonalization", stage_start, show_progress)
    return float(np.real(eigenvalues[0])), eigenvectors[:, 0]


def two_point_expectation_from_state(
    state: np.ndarray,
    op1_name: str,
    i: int,
    op2_name: str,
    j: int,
    n_sites: int,
    op_cache: Dict[str, sparse.spmatrix],
    ident: sparse.spmatrix,
) -> complex:
    op_list = [ident] * n_sites
    op_list[i] = op_cache[op1_name]
    op_list[j] = op_cache[op2_name]
    op_ij = kron_all(op_list)
    return complex(np.vdot(state, op_ij.dot(state)))


def collect_correlation_matrices_from_ed(
    geometry: GeometryData,
    state: np.ndarray,
    model_spec: ModelSpec,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    n_sites = geometry.number_of_sites
    op_cache = build_global_operator_cache_for_model(model_spec)
    ident = op_cache["Id"]
    op_pairs = [
        ("Sx", "Sx"), ("Sy", "Sy"), ("Sz", "Sz"),
        ("Tx", "Tx"), ("Ty", "Ty"), ("Tz", "Tz"),
        ("STx", "STx"), ("STy", "STy"), ("STz", "STz"),
    ]
    correlations = {f"{op1}_{op2}": np.zeros((n_sites, n_sites), dtype=complex) for op1, op2 in op_pairs}

    pair_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=(n_sites * (n_sites - 1)) * len(op_pairs),
        desc="ED correlations",
        unit="pair",
        leave=False,
    )
    row_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=n_sites,
        desc="ED corr rows",
        unit="row",
        leave=False,
    )
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            for op1, op2 in op_pairs:
                correlations[f"{op1}_{op2}"][i, j] = two_point_expectation_from_state(
                    state, op1, i, op2, j, n_sites, op_cache, ident
                )
                if pair_progress_bar is not None:
                    pair_progress_bar.update(1)
        if row_progress_bar is not None:
            row_progress_bar.update(1)

    if pair_progress_bar is not None:
        pair_progress_bar.close()
    if row_progress_bar is not None:
        row_progress_bar.close()

    return correlations


# ----------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------

def build_spin_orbital_scalar_correlations(correlations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    scalar = {
        "S": np.zeros_like(correlations["Sx_Sx"]),
        "T": np.zeros_like(correlations["Tx_Tx"]),
        "ST": np.zeros_like(correlations["STx_STx"]),
    }
    for gamma in ("x", "y", "z"):
        scalar["S"] = scalar["S"] + correlations[f"S{gamma}_S{gamma}"]
        scalar["T"] = scalar["T"] + correlations[f"T{gamma}_T{gamma}"]
        scalar["ST"] = scalar["ST"] + correlations[f"ST{gamma}_ST{gamma}"]
    return scalar


def bond_energy_from_correlations(
    i: int,
    j: int,
    gamma: str,
    correlations: Dict[str, np.ndarray],
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
) -> float:
    e_bond = 0.0j
    for coeff, op_name in model_terms_for_bond(gamma, model_spec, alpha, beta, coupling_j):
        key = f"{op_name}_{op_name}"
        if key not in correlations:
            raise KeyError(f"Missing correlation channel '{key}' required by current model.")
        e_bond = e_bond + coeff * correlations[key][i, j]
    return float(np.real(e_bond))


def all_bond_energies(
    geometry: GeometryData,
    correlations: Dict[str, np.ndarray],
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    show_progress: bool = False,
    progress_desc: str = "Bond energies",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(geometry.bond_list),
        desc=progress_desc,
        unit="bond",
        leave=False,
    )
    for bond in geometry.bond_list:
        rows.append(
            {
                "i": bond.i,
                "j": bond.j,
                "gamma": bond.gamma,
                "O_ij_gamma": bond_energy_from_correlations(
                    bond.i, bond.j, bond.gamma, correlations, model_spec, alpha, beta, coupling_j
                ),
            }
        )
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return rows


def mps_path_quality(geometry: GeometryData) -> Dict[str, Any]:
    def bond_indices_and_gamma(bond_obj: Any) -> Tuple[int, int, str]:
        if hasattr(bond_obj, "i") and hasattr(bond_obj, "j") and hasattr(bond_obj, "gamma"):
            return int(bond_obj.i), int(bond_obj.j), str(bond_obj.gamma)
        if hasattr(bond_obj, "site_i") and hasattr(bond_obj, "site_j") and hasattr(bond_obj, "bond_type"):
            return int(bond_obj.site_i), int(bond_obj.site_j), str(bond_obj.bond_type)
        raise AttributeError("Bond object must provide (i,j,gamma) or (site_i,site_j,bond_type).")

    if len(geometry.bond_list) == 0:
        return {
            "ordering": "snake_x_alternating_y",
            "bond_count": 0,
            "mean_index_span": 0.0,
            "median_index_span": 0.0,
            "p90_index_span": 0.0,
            "max_index_span": 0,
            "by_gamma": {},
        }

    bond_triplets = [bond_indices_and_gamma(bond) for bond in geometry.bond_list]
    spans = np.asarray([abs(i - j) for i, j, _ in bond_triplets], dtype=float)
    gamma_stats: Dict[str, Dict[str, Any]] = {}
    for gamma_label in sorted({gamma for _, _, gamma in bond_triplets}):
        gamma_spans = np.asarray(
            [abs(i - j) for i, j, gamma in bond_triplets if gamma == gamma_label],
            dtype=float,
        )
        gamma_stats[gamma_label] = {
            "bond_count": int(gamma_spans.size),
            "mean_index_span": float(np.mean(gamma_spans)),
            "max_index_span": int(np.max(gamma_spans)),
        }

    return {
        "ordering": "snake_x_alternating_y",
        "bond_count": int(spans.size),
        "mean_index_span": float(np.mean(spans)),
        "median_index_span": float(np.median(spans)),
        "p90_index_span": float(np.percentile(spans, 90.0)),
        "max_index_span": int(np.max(spans)),
        "by_gamma": gamma_stats,
    }


def lattice_display_name(lattice: str) -> str:
    mapping = {
        "honeycomb": "Honeycomb",
        "square": "Square",
        "triangular": "Triangular",
    }
    return mapping.get(lattice.lower(), lattice.title())


def _safe_filename_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    token = re.sub(r"_+", "_", token).strip("_.-")
    return token or "run"


def _rep_filename_token(rep: str) -> str:
    return str(rep).replace("/", "").replace(".", "p")


def model_simplified_name(model_spec: ModelSpec) -> str:
    family_map = {
        "yao_lee": "YL",
        "ising_like": "IsingLike",
        "heisenberg": "Heisenberg",
        "xy": "XY",
        "xxz": "XXZ",
        "xyz": "XYZ",
    }
    family = family_map.get(model_spec.model_family, model_spec.model_family)
    parts = [
        family,
        f"S{_rep_filename_token(model_spec.spin_rep)}",
        f"T{_rep_filename_token(model_spec.orbital_rep)}",
    ]
    if model_spec.model_family == "ising_like" or is_trivial_orbital(model_spec):
        parts.append(f"{model_spec.ising_axis.upper()}axis")
    return _safe_filename_token("_".join(parts))


def model_display_short_name(model_spec: ModelSpec) -> str:
    family_map = {
        "yao_lee": "YL",
        "ising_like": "Ising-like",
        "heisenberg": "Heisenberg",
        "xy": "XY",
        "xxz": "XXZ",
        "xyz": "XYZ",
    }
    family = family_map.get(model_spec.model_family, model_spec.model_family)
    return f"{family} S={model_spec.spin_rep}, T={model_spec.orbital_rep}, axis={model_spec.ising_axis}"


def geometry_size_filename_label(
    geometry: GeometryData,
    lattice: str,
    length_x: int,
    circumference_y: int,
    periodic_y: bool,
) -> str:
    lattice_short = {
        "honeycomb": "hc",
        "square": "sq",
        "triangular": "tri",
    }.get(lattice.lower(), lattice.lower())
    boundary = "pbcY" if periodic_y else "obcY"
    return _safe_filename_token(
        f"{lattice_short}_Lx{int(length_x)}_Cy{int(circumference_y)}_N{int(geometry.number_of_sites)}_{boundary}"
    )


def geometry_size_display_label(
    geometry: GeometryData,
    lattice: str,
    length_x: int,
    circumference_y: int,
    periodic_y: bool,
) -> str:
    boundary = "PBC-y" if periodic_y else "OBC-y"
    return (
        f"{lattice_display_name(lattice)} Lx={int(length_x)}, "
        f"Cy={int(circumference_y)}, N={int(geometry.number_of_sites)}, {boundary}"
    )


def run_output_prefix(
    model_spec: ModelSpec,
    geometry: GeometryData,
    lattice: str,
    length_x: int,
    circumference_y: int,
    periodic_y: bool,
) -> str:
    return _safe_filename_token(
        f"{model_simplified_name(model_spec)}_{geometry_size_filename_label(geometry, lattice, length_x, circumference_y, periodic_y)}"
    )


def run_title_label(
    model_spec: ModelSpec,
    geometry: GeometryData,
    lattice: str,
    length_x: int,
    circumference_y: int,
    periodic_y: bool,
) -> str:
    return (
        f"{model_display_short_name(model_spec)} | "
        f"{geometry_size_display_label(geometry, lattice, length_x, circumference_y, periodic_y)}"
    )


def labeled_output_filename(run_prefix: str, base_filename: str) -> str:
    stem, extension = os.path.splitext(base_filename)
    return f"{_safe_filename_token(run_prefix)}_{_safe_filename_token(stem)}{extension}"


def titled_for_run(base_title: str, title_label: str | None) -> str:
    if title_label:
        return f"{base_title}\n{title_label}"
    return base_title


def reciprocal_lattice_vectors(lattice: str) -> Tuple[np.ndarray, np.ndarray]:
    lattice_name = lattice.lower()
    if lattice_name == "square":
        return (
            np.array([2.0 * np.pi, 0.0], dtype=float),
            np.array([0.0, 2.0 * np.pi], dtype=float),
        )
    if lattice_name in ("honeycomb", "triangular"):
        a = 1.0
        b1 = np.array([2.0 * np.pi / (np.sqrt(3.0) * a), -2.0 * np.pi / (3.0 * a)], dtype=float)
        b2 = np.array([0.0, 4.0 * np.pi / (3.0 * a)], dtype=float)
        return b1, b2
    raise ValueError(f"Unsupported lattice '{lattice}' for reciprocal vectors.")


def default_high_symmetry_momenta(lattice: str) -> Dict[str, np.ndarray]:
    b1, b2 = reciprocal_lattice_vectors(lattice)
    lattice_name = lattice.lower()
    if lattice_name == "square":
        return {
            "Gamma": np.array([0.0, 0.0], dtype=float),
            "X": 0.5 * b1,
            "M": 0.5 * (b1 + b2),
            "Y": 0.5 * b2,
        }
    if lattice_name in ("honeycomb", "triangular"):
        return {
            "Gamma": np.array([0.0, 0.0], dtype=float),
            "M1": 0.5 * b1,
            "M2": 0.5 * b2,
            "M3": 0.5 * (b1 + b2),
            "K1": (2.0 * b1 + b2) / 3.0,
            "K2": (b1 + 2.0 * b2) / 3.0,
        }
    raise ValueError(f"Unsupported lattice '{lattice}' for high-symmetry momenta.")


def structure_factor_from_scalar_correlation(
    q: np.ndarray,
    scalar_corr_matrix: np.ndarray,
    geometry: GeometryData,
) -> float:
    s_q = 0.0j
    n_sites = geometry.number_of_sites
    for i in range(n_sites):
        for j in range(n_sites):
            r_ij = geometry.positions[i] - geometry.positions[j]
            s_q = s_q + np.exp(1.0j * np.dot(q, r_ij)) * scalar_corr_matrix[i, j]
    return float(np.real(s_q) / n_sites)


def all_high_symmetry_structure_factors(
    scalar_correlations: Dict[str, np.ndarray],
    geometry: GeometryData,
    lattice: str,
    show_progress: bool = False,
    progress_desc: str = "Structure factors",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    momenta = list(default_high_symmetry_momenta(lattice).items())
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(momenta),
        desc=progress_desc,
        unit="Q",
        leave=False,
    )
    for label, q in momenta:
        rows.append(
            {
                "Q_label": label,
                "Qx": float(q[0]),
                "Qy": float(q[1]),
                "S(Q)": structure_factor_from_scalar_correlation(q, scalar_correlations["S"], geometry),
                "T(Q)": structure_factor_from_scalar_correlation(q, scalar_correlations["T"], geometry),
                "ST(Q)": structure_factor_from_scalar_correlation(q, scalar_correlations["ST"], geometry),
            }
        )
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return rows


# ----------------------------------------------------------------------
# Plotting (PNG outputs)
# ----------------------------------------------------------------------

def ensure_folder_exists(folder_path: str) -> None:
    os.makedirs(folder_path, exist_ok=True)


def _geometry_positions(geometry: Any) -> np.ndarray:
    if hasattr(geometry, "positions"):
        return np.asarray(geometry.positions, dtype=float)
    if hasattr(geometry, "coordinates"):
        return np.asarray(geometry.coordinates, dtype=float)
    raise AttributeError("Geometry object must provide positions or coordinates.")


def _bond_i_j_gamma(bond: Any) -> Tuple[int, int, str]:
    if hasattr(bond, "i") and hasattr(bond, "j") and hasattr(bond, "gamma"):
        return int(bond.i), int(bond.j), str(bond.gamma)
    if hasattr(bond, "site_i") and hasattr(bond, "site_j") and hasattr(bond, "bond_type"):
        return int(bond.site_i), int(bond.site_j), str(bond.bond_type)
    raise AttributeError("Bond object must have (i,j,gamma) or (site_i,site_j,bond_type).")


def _load_tenpy_backend_module() -> Any:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    return importlib.import_module("yaoleemodel")


def save_geometry_diagram(
    geometry: GeometryData,
    filepath: str,
    lattice: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
    positions = _geometry_positions(geometry)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for bond in geometry.bond_list:
        site_i, site_j, gamma = _bond_i_j_gamma(bond)
        p_i = positions[site_i]
        p_j = positions[site_j]
        ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], color=colors.get(gamma, "#666666"), linewidth=1.5, alpha=0.9)

    if hasattr(geometry, "sublattice_indices"):
        sublattice = np.asarray(geometry.sublattice_indices)
        if np.any(sublattice == 1):
            a_idx = np.where(sublattice == 0)[0]
            b_idx = np.where(sublattice == 1)[0]
            ax.scatter(positions[a_idx, 0], positions[a_idx, 1], s=20, c="#111111", label="A")
            ax.scatter(positions[b_idx, 0], positions[b_idx, 1], s=20, c="#ff7f0e", label="B")
        else:
            ax.scatter(positions[:, 0], positions[:, 1], s=16, c="#111111", label="sites")
    else:
        ax.scatter(positions[:, 0], positions[:, 1], s=16, c="#111111", label="sites")
    ax.set_title(titled_for_run(f"{lattice_display_name(lattice)} Cylinder Geometry", title_label))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_bond_energy_diagram(
    geometry: GeometryData,
    bond_rows: List[Dict[str, Any]],
    filepath: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    positions = _geometry_positions(geometry)
    segments = []
    values = []
    for row in bond_rows:
        i, j = int(row["i"]), int(row["j"])
        segments.append([positions[i], positions[j]])
        values.append(float(row["O_ij_gamma"]))

    values_arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    collection = LineCollection(segments, cmap="coolwarm", linewidths=3.0)
    collection.set_array(values_arr)
    ax.add_collection(collection)
    ax.scatter(positions[:, 0], positions[:, 1], c="black", s=10, zorder=3)
    ax.autoscale()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    cbar = fig.colorbar(collection, ax=ax, shrink=0.9)
    cbar.set_label("Bond energy O_ij_gamma")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_structure_factor_plot(rows: List[Dict[str, Any]], filepath: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["Q_label"] for row in rows]
    s_vals = [float(row["S(Q)"]) for row in rows]
    t_vals = [float(row["T(Q)"]) for row in rows]
    st_vals = [float(row["ST(Q)"]) for row in rows]

    x = np.arange(len(labels), dtype=float)
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.bar(x - w, s_vals, width=w, label="S(Q)")
    ax.bar(x, t_vals, width=w, label="T(Q)")
    ax.bar(x + w, st_vals, width=w, label="ST(Q)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("High-symmetry momentum")
    ax.set_ylabel("Structure factor")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_scalar_correlation_heatmaps(
    scalar_correlations: Dict[str, np.ndarray],
    filepath: str,
    title_prefix: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    keys = ("S", "T", "ST")
    for ax, key in zip(axes, keys):
        matrix = np.real(scalar_correlations[key])
        image = ax.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(f"{title_prefix} {key}_ij")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_multi_method_energy_comparison(
    method_to_energy: Dict[str, float],
    filepath: str,
    title: str = "Ground-State Energy Per Site Comparison",
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred_order = ["DMRG", "ED", "iDMRG-x"]
    labels = [label for label in preferred_order if label in method_to_energy]
    labels += [label for label in method_to_energy.keys() if label not in labels]
    values = [float(method_to_energy[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    color_map = {
        "DMRG": "#1f77b4",
        "ED": "#ff7f0e",
        "iDMRG-x": "#2ca02c",
    }
    colors = [color_map.get(label, "#666666") for label in labels]
    ax.bar(labels, values, color=colors)
    ax.set_title(titled_for_run(title, title_label))
    ax.set_ylabel("Energy per site")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_entropy_profiles_comparison(
    entropy_profiles: Dict[str, Dict[str, Any]],
    filepath: str,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, sharex=True)
    axes_flat = list(axes.flatten())
    method_order = ["DMRG", "ED", "iDMRG-x"]
    colors = {
        "DMRG": "#1f77b4",
        "ED": "#ff7f0e",
        "iDMRG-x": "#2ca02c",
    }
    markers = {
        "DMRG": "o",
        "ED": "s",
        "iDMRG-x": "^",
    }
    linestyles = {
        "DMRG": "-",
        "ED": "--",
        "iDMRG-x": ":",
    }

    for axis, order_n in zip(axes_flat, orders):
        key = f"S{order_n}"
        plotted = False
        for method in method_order:
            profile = entropy_profiles.get(method, None)
            if profile is None:
                continue
            values = profile.get("entropies", {}).get(key, [])
            x_values = profile.get("cuts_normalized", [])
            if len(values) == 0 or len(x_values) != len(values):
                continue
            axis.plot(
                x_values,
                values,
                linestyle=linestyles.get(method, "-"),
                linewidth=1.8,
                markersize=3.5,
                marker=markers.get(method, "o"),
                color=colors.get(method, None),
                label=method,
            )
            plotted = True
        axis.set_title(f"Renyi Entropy n={order_n}")
        axis.set_xlabel("Normalized cut position")
        axis.set_ylabel("Entropy")
        axis.grid(alpha=0.25)
        if not plotted:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if len(handles) > 0:
        axes_flat[0].legend(loc="best")
    fig.suptitle(titled_for_run("Entanglement Entropy Profiles by Method", title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_entropy_method_means_comparison(
    entropy_profiles: Dict[str, Dict[str, Any]],
    filepath: str,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = [method for method in ["DMRG", "ED", "iDMRG-x"] if method in entropy_profiles]
    if len(method_order) == 0:
        raise RuntimeError("No entropy profiles available for method-mean comparison.")

    x = np.arange(len(orders), dtype=float)
    width = 0.8 / float(len(method_order))
    offsets = np.linspace(-0.4 + width / 2.0, 0.4 - width / 2.0, len(method_order))
    color_map = {
        "DMRG": "#1f77b4",
        "ED": "#ff7f0e",
        "iDMRG-x": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for idx, method in enumerate(method_order):
        profile = entropy_profiles[method]
        summary = profile.get("summary", {})
        means = [float(summary.get(f"S{order_n}_mean", np.nan)) for order_n in orders]
        ax.bar(
            x + offsets[idx],
            means,
            width=width,
            label=method,
            color=color_map.get(method, "#666666"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={order_n}" for order_n in orders])
    ax.set_xlabel("Renyi order")
    ax.set_ylabel("Mean entropy across cuts")
    ax.set_title(titled_for_run("Method Comparison: Mean Entanglement Entropies", title_label))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_dmrg_ed_energy_comparison(
    dmrg_energy: float,
    ed_energy: float,
    filepath: str,
    title_label: str | None = None,
) -> None:
    save_multi_method_energy_comparison(
        method_to_energy={"DMRG": float(dmrg_energy), "ED": float(ed_energy)},
        filepath=filepath,
        title="Ground-State Energy Comparison",
        title_label=title_label,
    )


def save_dmrg_ed_structure_comparison(
    dmrg_rows: List[Dict[str, Any]],
    ed_rows: List[Dict[str, Any]],
    filepath: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dmrg_map = {row["Q_label"]: row for row in dmrg_rows}
    ed_map = {row["Q_label"]: row for row in ed_rows}
    labels = [label for label in dmrg_map.keys() if label in ed_map]
    x = np.arange(len(labels), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150, sharex=True)
    channels = ("S(Q)", "T(Q)", "ST(Q)")
    for ax, channel in zip(axes, channels):
        dmrg_values = [dmrg_map[label][channel] for label in labels]
        ed_values = [ed_map[label][channel] for label in labels]
        ax.plot(x, dmrg_values, marker="o", linestyle="-", linewidth=1.8, label="DMRG")
        ax.plot(x, ed_values, marker="s", linestyle="--", linewidth=1.8, label="ED")
        ax.set_title(channel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")
    fig.suptitle(titled_for_run("DMRG vs ED Structure Factors", title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI + main
# ----------------------------------------------------------------------

def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tenax/ED benchmarking for the Yao-Lee model.")
    parser.add_argument(
        "--spin-rep",
        "--spin_rep",
        dest="spin_rep",
        type=str,
        choices=list(SPIN_REP_OPTIONS),
        default=SPIN_REP,
        help="Spin representation: 1/2 or 3/2.",
    )
    parser.add_argument(
        "--orbital-rep",
        "--orbital_rep",
        dest="orbital_rep",
        type=str,
        choices=list(ORBITAL_REP_OPTIONS) + ["1"],
        default=ORBITAL_REP,
        help="Orbital representation: 0 (trivial) or 1/2. (Legacy alias: 1 -> 0)",
    )
    parser.add_argument(
        "--model-family",
        "--model_family",
        dest="model_family",
        type=str,
        choices=list(MODEL_FAMILY_OPTIONS),
        default=MODEL_FAMILY,
        help="Hamiltonian family. With orbital_rep=0, yao_lee automatically reduces to spin-only Ising-like couplings.",
    )
    parser.add_argument(
        "--ising-axis",
        "--ising_axis",
        dest="ising_axis",
        type=str,
        choices=list(AXIS_OPTIONS),
        default=ISING_AXIS,
        help="Spin/orbital axis used by ising_like couplings (and orbital_rep=0 fallback).",
    )
    parser.add_argument(
        "--symmetry-mode",
        "--symmetry_mode",
        dest="symmetry_mode",
        type=str,
        choices=list(SYMMETRY_MODE_OPTIONS),
        default=SYMMETRY_MODE,
        help=(
            "Tensor symmetry mode: none (dense), u1 (strict total Sz/Tz conservation), "
            "or z2 (parity; use this for x/y Yao-Lee flip terms)."
        ),
    )
    parser.add_argument(
        "--u1-target-sz2",
        "--u1_target_sz2",
        dest="u1_target_sz2",
        type=int,
        default=U1_TARGET_TOTAL_SZ2,
        help="Target total 2*Sz sector for symmetry_mode=u1.",
    )
    parser.add_argument(
        "--u1-target-tz2",
        "--u1_target_tz2",
        dest="u1_target_tz2",
        type=int,
        default=U1_TARGET_TOTAL_TZ2,
        help="Target total 2*Tz sector for symmetry_mode=u1.",
    )
    parser.add_argument(
        "--z2-target-parity",
        "--z2_target_parity",
        dest="z2_target_parity",
        type=int,
        choices=list(Z2_PARITY_OPTIONS),
        default=Z2_TARGET_PARITY,
        help="Target global parity sector for symmetry_mode=z2 (0=even, 1=odd).",
    )
    parser.add_argument(
        "--strict-symmetry-selection-rules",
        "--strict_symmetry_selection_rules",
        dest="strict_symmetry_selection_rules",
        action=argparse.BooleanOptionalAction,
        default=STRICT_SYMMETRY_SELECTION_RULES,
        help="If enabled, reject MPO terms that violate the active U1/Z2 selection rule.",
    )
    parser.add_argument(
        "--lattice",
        type=str,
        choices=list(LATTICE_OPTIONS),
        default=LATTICE_TYPE,
        help="Lattice type used to build the Hamiltonian bonds.",
    )
    parser.add_argument(
        "--length-x",
        "--length_x",
        dest="length_x",
        type=int,
        default=LENGTH_X,
        help="Number of unit cells along the open cylinder axis.",
    )
    parser.add_argument(
        "--circumference-y",
        "--circumference_y",
        dest="circumference_y",
        type=int,
        default=CIRCUMFERENCE_Y,
        help="Number of unit cells around the cylinder circumference.",
    )
    parser.add_argument(
        "--open-around-cylinder",
        "--open_around_cylinder",
        dest="open_around_cylinder",
        action="store_true",
        help="Use open boundary conditions around the cylinder instead of periodic.",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Model alpha parameter.")
    parser.add_argument("--beta", type=float, default=BETA, help="Model beta parameter.")
    parser.add_argument("--coupling-j", "--coupling_j", dest="coupling_j", type=float, default=COUPLING_J)
    parser.add_argument(
        "--max-bond-dimension",
        "--max_bond_dimension",
        dest="max_bond_dimension",
        type=int,
        default=MAX_BOND_DIMENSION,
    )
    parser.add_argument("--max-sweeps", "--max_sweeps", dest="max_sweeps", type=int, default=MAX_SWEEPS)
    parser.add_argument(
        "--truncation-cutoff",
        "--truncation_cutoff",
        dest="truncation_cutoff",
        type=float,
        default=TRUNCATION_CUTOFF,
        help="Accepted for compatibility with yaoleemodel.py; Tenax backend may ignore it.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--initial-state",
        "--initial_state",
        dest="initial_state",
        type=str,
        choices=list(INITIAL_STATE_OPTIONS),
        default=INITIAL_STATE_STYLE,
        help="Accepted for compatibility with yaoleemodel.py; Tenax backend currently uses random MPS init.",
    )
    parser.add_argument(
        "--run-ed",
        "--run_ed",
        dest="run_ed",
        action=argparse.BooleanOptionalAction,
        default=RUN_ED,
        help="Run exact diagonalization comparison when the Hilbert-space size is within the configured limits.",
    )
    parser.add_argument(
        "--run-idmrg",
        "--run_idmrg",
        dest="run_idmrg",
        action=argparse.BooleanOptionalAction,
        default=RUN_IDMRG,
        help="Run additional Tenax iDMRG-x workflow and compare with finite DMRG/ED.",
    )
    parser.add_argument(
        "--idmrg-max-iterations",
        "--idmrg_max_iterations",
        dest="idmrg_max_iterations",
        type=int,
        default=IDMRG_MAX_ITERATIONS,
        help="Maximum iDMRG iterations.",
    )
    parser.add_argument(
        "--idmrg-max-local-dim",
        "--idmrg_max_local_dim",
        dest="idmrg_max_local_dim",
        type=int,
        default=IDMRG_MAX_LOCAL_DIM,
        help="Safety limit for effective local dimension used by iDMRG bulk mapping.",
    )
    parser.add_argument(
        "--idmrg-bulk-kind",
        "--idmrg_bulk_kind",
        dest="idmrg_bulk_kind",
        type=str,
        choices=list(IDMRG_BULK_KIND_OPTIONS),
        default=IDMRG_BULK_KIND,
        help="How to extract iDMRG bulk MPO from finite MPO.",
    )
    parser.add_argument("--output-folder", "--output_folder", dest="output_folder", type=str, default=OUTPUT_FOLDER)
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(BACKEND_OPTIONS),
        default=BACKEND,
        help="Select backend. auto tries Tenax first, then falls back to yaoleemodel (TeNPy).",
    )
    parser.add_argument(
        "--overwrite-plots",
        action="store_true",
        default=OVERWRITE_EXISTING_PLOTS,
        help="Regenerate PNG outputs even if files already exist.",
    )
    parser.add_argument(
        "--strict-plot-errors",
        action="store_true",
        default=STRICT_PLOT_ERRORS,
        help="Stop immediately when one plot save fails.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help="Show tqdm progress bars and stage timing logs.",
    )
    parser.set_defaults(open_around_cylinder=(not PERIODIC_AROUND_CYLINDER))
    return parser.parse_args()


def to_json_compatible(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_json_compatible(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_compatible(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
        return float(obj)
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    return obj


def write_json(filepath: str, data_dict: Dict[str, Any]) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(to_json_compatible(data_dict), file, indent=2, sort_keys=True)


def _save_summary_checkpoint(output_folder: str, summary: Dict[str, Any]) -> None:
    outputs = summary.get("outputs", {})
    summary_filename = "run_summary.json"
    if isinstance(outputs, dict):
        summary_filename = str(outputs.get("run_summary_json", summary_filename))

    run_summary_path = os.path.join(output_folder, summary_filename)
    write_json(run_summary_path, summary)


def _load_previous_summary(output_folder: str) -> Dict[str, Any] | None:
    candidate_paths = [os.path.join(output_folder, "run_summary.json")]
    try:
        if os.path.isdir(output_folder):
            labeled_summaries = [
                os.path.join(output_folder, filename)
                for filename in os.listdir(output_folder)
                if filename.endswith("_run_summary.json")
            ]
            labeled_summaries.sort(key=lambda path: os.path.getmtime(path), reverse=True)
            candidate_paths.extend(labeled_summaries)
    except Exception:
        pass

    for run_summary_path in candidate_paths:
        if not os.path.exists(run_summary_path):
            continue
        try:
            with open(run_summary_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _extract_consistency_signature(parameters: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "lattice",
        "length_x",
        "circumference_y",
        "open_around_cylinder",
        "alpha",
        "beta",
        "coupling_j",
        "spin_rep",
        "orbital_rep",
        "model_family",
        "ising_axis",
        "symmetry_mode",
        "u1_target_sz2",
        "u1_target_tz2",
        "z2_target_parity",
        "strict_symmetry_selection_rules",
        "backend",
        "max_bond_dimension",
        "max_sweeps",
        "run_idmrg",
        "idmrg_max_iterations",
        "idmrg_max_local_dim",
        "idmrg_bulk_kind",
        "seed",
        "run_ed",
    ]
    return {key: parameters.get(key) for key in keys}


def _record_output_status(
    summary: Dict[str, Any],
    key: str,
    filename: str,
    status: str,
    error: str | None = None,
) -> None:
    outputs = summary.setdefault("outputs", {})
    outputs[key] = filename
    output_status = summary.setdefault("output_status", {})
    output_status[key] = {"status": status}
    if error is not None:
        output_status[key]["error"] = error


def _save_plot_step(
    summary: Dict[str, Any],
    output_folder: str,
    key: str,
    filename: str,
    save_callable: Callable[[str], None],
    overwrite_existing: bool,
    continue_on_plot_error: bool,
) -> None:
    filepath = os.path.join(output_folder, filename)
    if os.path.exists(filepath) and not overwrite_existing:
        _record_output_status(summary, key, filename, "skipped_exists")
        _save_summary_checkpoint(output_folder, summary)
        print(f"[output] skip existing: {filename}")
        return

    try:
        save_callable(filepath)
        _record_output_status(summary, key, filename, "saved")
        _save_summary_checkpoint(output_folder, summary)
        print(f"[output] saved: {filename}")
    except Exception as exc:
        _record_output_status(summary, key, filename, "failed", str(exc))
        _save_summary_checkpoint(output_folder, summary)
        print(f"[output] failed: {filename} :: {exc}")
        if not continue_on_plot_error:
            raise


def main() -> None:
    args = parse_command_line()
    ensure_folder_exists(args.output_folder)
    show_progress = bool(args.progress)
    overwrite_existing = bool(args.overwrite_plots)
    continue_on_plot_error = not bool(args.strict_plot_errors)
    lattice_name = str(args.lattice).lower()
    periodic_y = not args.open_around_cylinder
    args.symmetry_mode = _normalize_symmetry_mode(args.symmetry_mode)
    model_spec = build_model_spec(
        spin_rep=args.spin_rep,
        orbital_rep=args.orbital_rep,
        model_family=args.model_family,
        ising_axis=args.ising_axis,
    )
    # Normalize legacy alias "1" -> "0" in recorded parameters.
    args.orbital_rep = model_spec.orbital_rep

    # Keep outputs consistent with current settings: if the output folder already
    # contains results from a different parameter set, auto-overwrite stale plots.
    previous_summary = _load_previous_summary(args.output_folder)
    if (
        (not overwrite_existing)
        and previous_summary is not None
        and isinstance(previous_summary.get("parameters"), dict)
    ):
        previous_signature = _extract_consistency_signature(previous_summary["parameters"])
        current_signature = _extract_consistency_signature(vars(args))
        if previous_signature != current_signature:
            overwrite_existing = True
            print(
                "[output] Detected existing plots from a different parameter set; "
                "auto-enabling overwrite for consistency."
            )

    if args.backend == "tenpy" and lattice_name != "honeycomb":
        raise ValueError(
            "TeNPy backend currently supports only --lattice honeycomb. "
            "Use --backend tenax for square/triangular lattices."
        )
    if args.backend == "tenpy" and args.symmetry_mode != "none":
        raise ValueError(
            "TeNPy backend in this script does not support symmetry_mode != none. "
            "Use --backend tenax for U1/Z2 block-sparse runs."
        )
    if args.backend == "tenpy":
        if not (
            model_spec.spin_rep == "1/2"
            and model_spec.orbital_rep == "1/2"
            and model_spec.model_family == "yao_lee"
            and model_spec.ising_axis == "z"
        ):
            raise ValueError(
                "TeNPy backend currently supports only the legacy default model "
                "(spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z). "
                "Use --backend tenax for other constructions."
            )

    geometry = None
    dmrg_info: Dict[str, Any] = {}
    dmrg_energy = 0.0
    dmrg_scalar_correlations: Dict[str, np.ndarray] = {}
    dmrg_bond_rows: List[Dict[str, Any]] = []
    dmrg_structure_factor_rows: List[Dict[str, Any]] = []
    dmrg_state_obj: Any = None
    tenax_mpo = None
    backend_used = None
    backend_warning = None
    entanglement_warning: str | None = None
    entropy_profiles: Dict[str, Dict[str, Any]] = {}
    geometry_plot_status = "not_attempted"
    geometry_plot_error: str | None = None
    run_file_prefix: str | None = None
    run_plot_title_label: str | None = None
    run_summary_filename = "run_summary.json"

    def configure_run_output_names(geometry_obj: Any) -> None:
        nonlocal run_file_prefix, run_plot_title_label, run_summary_filename
        if run_file_prefix is not None and run_plot_title_label is not None:
            return
        run_file_prefix = run_output_prefix(
            model_spec=model_spec,
            geometry=geometry_obj,
            lattice=lattice_name,
            length_x=args.length_x,
            circumference_y=args.circumference_y,
            periodic_y=periodic_y,
        )
        run_plot_title_label = run_title_label(
            model_spec=model_spec,
            geometry=geometry_obj,
            lattice=lattice_name,
            length_x=args.length_x,
            circumference_y=args.circumference_y,
            periodic_y=periodic_y,
        )
        run_summary_filename = labeled_output_filename(run_file_prefix, "run_summary.json")

    def output_filename(base_filename: str) -> str:
        if run_file_prefix is None:
            return base_filename
        return labeled_output_filename(run_file_prefix, base_filename)

    def plot_title(base_title: str) -> str:
        return titled_for_run(base_title, run_plot_title_label)

    def save_geometry_before_sweep(geometry_obj: Any) -> None:
        nonlocal geometry_plot_status, geometry_plot_error
        configure_run_output_names(geometry_obj)
        filename = output_filename("geometry_diagram.png")
        filepath = os.path.join(args.output_folder, filename)
        if os.path.exists(filepath) and not overwrite_existing:
            geometry_plot_status = "skipped_exists"
            print(f"[output] skip existing: {filename}")
            return
        try:
            save_geometry_diagram(geometry_obj, filepath, lattice_name, title_label=run_plot_title_label)
            geometry_plot_status = "saved"
            print(f"[output] saved: {filename}")
        except Exception as exc:
            geometry_plot_status = "failed"
            geometry_plot_error = str(exc)
            print(f"[output] failed: {filename} :: {exc}")
            if not continue_on_plot_error:
                raise

    # Try Tenax first unless user forces tenpy.
    if args.backend in ("auto", "tenax"):
        try:
            geometry = build_lattice_geometry(
                lattice=lattice_name,
                length_x=args.length_x,
                circumference_y=args.circumference_y,
                periodic_y=periodic_y,
            )
            if geometry_plot_status == "not_attempted":
                save_geometry_before_sweep(geometry)
            tenax_mps, tenax_mpo, dmrg_info = run_tenax_cylindrical_dmrg(
                geometry=geometry,
                model_spec=model_spec,
                alpha=args.alpha,
                beta=args.beta,
                coupling_j=args.coupling_j,
                max_bond_dimension=args.max_bond_dimension,
                max_sweeps=args.max_sweeps,
                random_seed=args.seed,
                symmetry_mode=args.symmetry_mode,
                u1_target_total_sz2=args.u1_target_sz2,
                u1_target_total_tz2=args.u1_target_tz2,
                z2_target_parity=args.z2_target_parity,
                strict_symmetry_selection_rules=args.strict_symmetry_selection_rules,
                show_progress=show_progress,
            )
            dmrg_energy = float(dmrg_info["E"])
            dmrg_state_obj = tenax_mps
            dmrg_correlations = collect_correlation_matrices_from_tenax(
                tenax_mps,
                geometry,
                model_spec=model_spec,
                show_progress=show_progress,
            )
            dmrg_scalar_correlations = build_spin_orbital_scalar_correlations(dmrg_correlations)
            dmrg_bond_rows = all_bond_energies(
                geometry,
                dmrg_correlations,
                model_spec,
                args.alpha,
                args.beta,
                args.coupling_j,
                show_progress=show_progress,
                progress_desc="DMRG bond energies",
            )
            dmrg_structure_factor_rows = all_high_symmetry_structure_factors(
                dmrg_scalar_correlations,
                geometry,
                lattice=lattice_name,
                show_progress=show_progress,
                progress_desc="DMRG structure factors",
            )
            backend_used = "tenax"
        except Exception as tenax_exc:
            if args.backend == "tenax":
                raise
            if args.symmetry_mode != "none":
                raise RuntimeError(
                    "Tenax failed while symmetry_mode is enabled, and TeNPy fallback "
                    "cannot preserve U1/Z2 block-sparse symmetry sectors in this script. "
                    f"Original Tenax error: {tenax_exc}"
                ) from tenax_exc
            if lattice_name != "honeycomb":
                raise RuntimeError(
                    f"Tenax backend failed on lattice='{lattice_name}', and TeNPy fallback only supports honeycomb. "
                    f"Original Tenax error: {tenax_exc}"
                ) from tenax_exc
            if show_progress:
                print(f"[backend] Tenax failed; switching to TeNPy fallback. Reason: {tenax_exc}")
            backend_warning = f"Tenax backend failed, fallback to TeNPy: {tenax_exc}"

    # TeNPy fallback via sibling module yaoleemodel.py
    if backend_used is None:
        if lattice_name != "honeycomb":
            raise RuntimeError(
                f"TeNPy fallback does not support lattice='{lattice_name}'. "
                "Only honeycomb is supported in yaoleemodel.py."
            )
        if not (
            model_spec.spin_rep == "1/2"
            and model_spec.orbital_rep == "1/2"
            and model_spec.model_family == "yao_lee"
            and model_spec.ising_axis == "z"
        ):
            raise RuntimeError(
                "TeNPy fallback only supports the legacy default model "
                "(spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z)."
            )
        yl = _load_tenpy_backend_module()
        stage_start = _start_stage("TeNPy fallback DMRG", show_progress)
        geometry = yl.build_honeycomb_cylinder_geometry(
            length_x=args.length_x,
            circumference_y=args.circumference_y,
            periodic_y=periodic_y,
        )
        if geometry_plot_status == "not_attempted":
            save_geometry_before_sweep(geometry)
        psi, _, dmrg_info = yl.run_cylindrical_dmrg(
            geometry=geometry,
            alpha=args.alpha,
            beta=args.beta,
            coupling_j=args.coupling_j,
            max_bond_dimension=args.max_bond_dimension,
            max_sweeps=args.max_sweeps,
            truncation_cutoff=args.truncation_cutoff,
            random_seed=args.seed,
            product_state_style=args.initial_state,
        )
        dmrg_energy = float(dmrg_info["E"])
        dmrg_state_obj = psi
        dmrg_correlations = yl.collect_correlation_matrices_from_dmrg(psi)
        scalar_native = yl.build_spin_orbital_scalar_correlations(dmrg_correlations)
        dmrg_scalar_correlations = {
            "S": scalar_native["spin_scalar"],
            "T": scalar_native["orbital_scalar"],
            "ST": scalar_native["mixed_scalar"],
        }
        dmrg_bond_rows = yl.all_bond_energies(geometry, dmrg_correlations, args.alpha, args.beta, args.coupling_j)
        dmrg_structure_factor_rows = yl.all_high_symmetry_structure_factors(scalar_native, geometry)
        backend_used = "tenpy_fallback"
        _end_stage("TeNPy fallback DMRG", stage_start, show_progress)

    try:
        if dmrg_state_obj is not None:
            if backend_used == "tenax":
                dmrg_entropy_profile = compute_tenax_finite_mps_entropy_profile(
                    dmrg_state_obj,
                    orders=ENTROPY_ORDERS,
                    show_progress=show_progress,
                )
            else:
                dmrg_entropy_profile = compute_tenpy_finite_mps_entropy_profile(
                    dmrg_state_obj,
                    orders=ENTROPY_ORDERS,
                    show_progress=show_progress,
                )
            entropy_profiles["DMRG"] = dmrg_entropy_profile
    except Exception as exc:
        entanglement_warning = f"Failed to compute DMRG entanglement profile: {exc}"

    configure_run_output_names(geometry)
    lattice_label = lattice_display_name(lattice_name)
    model_short_label = model_simplified_name(model_spec)
    size_short_label = geometry_size_filename_label(
        geometry,
        lattice_name,
        args.length_x,
        args.circumference_y,
        periodic_y,
    )
    size_display_label = geometry_size_display_label(
        geometry,
        lattice_name,
        args.length_x,
        args.circumference_y,
        periodic_y,
    )
    model_label = (
        f"{model_spec.model_family}, spin={model_spec.spin_rep}, orbital={model_spec.orbital_rep}, axis={model_spec.ising_axis}"
    )

    summary: Dict[str, Any] = {
        "model_name": f"{lattice_label} spin-orbital model ({model_label})",
        "model_simplified_name": model_short_label,
        "model_size_name": size_short_label,
        "run_output_prefix": run_file_prefix,
        "monitor_data_name": run_summary_filename,
        "plot_title_label": run_plot_title_label,
        "run_status": "running",
        "parameters": vars(args),
        "model_spec": {
            "spin_rep": model_spec.spin_rep,
            "orbital_rep": model_spec.orbital_rep,
            "model_family": model_spec.model_family,
            "ising_axis": model_spec.ising_axis,
            "spin_value": model_spec.spin_value,
            "orbital_value": model_spec.orbital_value,
            "physical_dim": model_spec.physical_dim,
            "orbital_is_trivial": is_trivial_orbital(model_spec),
            "effective_interaction": (
                "spin_only_ising_like_fallback"
                if (model_spec.model_family == "yao_lee" and is_trivial_orbital(model_spec))
                else model_spec.model_family
            ),
        },
        "backend_used": backend_used,
        "geometry": {
            "lattice": lattice_name,
            "number_of_sites": geometry.number_of_sites,
            "number_of_bonds": len(geometry.bond_list),
            "size_label": size_display_label,
            "mps_path": mps_path_quality(geometry),
        },
        "dmrg": {
            "ground_state_energy": dmrg_energy,
            "energy_per_site": dmrg_energy / geometry.number_of_sites,
            "info": dmrg_info,
            "entanglement": entropy_profiles.get("DMRG"),
            "structure_factors": dmrg_structure_factor_rows,
        },
        "stages": {
            "dmrg": "completed",
            "dmrg_plots": "running",
            "idmrg": "pending" if args.run_idmrg else "not_requested",
            "ed": "pending" if args.run_ed else "not_requested",
        },
        "outputs": {
            "run_summary_json": run_summary_filename,
            "monitor_data_json": run_summary_filename,
        },
    }
    if backend_warning:
        summary["backend_warning"] = backend_warning
    if entanglement_warning:
        summary["entanglement_warning"] = entanglement_warning
    _save_summary_checkpoint(args.output_folder, summary)

    # Save DMRG plots immediately, one by one.
    _record_output_status(
        summary,
        "geometry_diagram_png",
        output_filename("geometry_diagram.png"),
        geometry_plot_status,
        geometry_plot_error,
    )
    _save_summary_checkpoint(args.output_folder, summary)
    _save_plot_step(
        summary,
        args.output_folder,
        "dmrg_bond_energy_diagram_png",
        output_filename("dmrg_bond_energy_diagram.png"),
        lambda path: save_bond_energy_diagram(geometry, dmrg_bond_rows, path, plot_title("DMRG Bond-Energy Diagram")),
        overwrite_existing,
        continue_on_plot_error,
    )
    _save_plot_step(
        summary,
        args.output_folder,
        "dmrg_structure_factors_png",
        output_filename("dmrg_structure_factors.png"),
        lambda path: save_structure_factor_plot(dmrg_structure_factor_rows, path, plot_title("DMRG Structure Factors")),
        overwrite_existing,
        continue_on_plot_error,
    )
    _save_plot_step(
        summary,
        args.output_folder,
        "dmrg_scalar_correlation_heatmaps_png",
        output_filename("dmrg_scalar_correlation_heatmaps.png"),
        lambda path: save_scalar_correlation_heatmaps(dmrg_scalar_correlations, path, f"DMRG | {run_plot_title_label}"),
        overwrite_existing,
        continue_on_plot_error,
    )
    summary["stages"]["dmrg_plots"] = "completed"
    _save_summary_checkpoint(args.output_folder, summary)

    # Optional iDMRG workflow (runs after finite DMRG outputs are saved).
    if args.run_idmrg:
        summary["stages"]["idmrg"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        if backend_used != "tenax":
            summary["idmrg"] = {
                "status": "skipped",
                "reason": "iDMRG-x currently requires Tenax backend output.",
            }
            summary["stages"]["idmrg"] = "skipped"
            _save_summary_checkpoint(args.output_folder, summary)
        elif tenax_mpo is None:
            summary["idmrg"] = {
                "status": "failed",
                "error": "Tenax MPO object unavailable after DMRG.",
            }
            summary["stages"]["idmrg"] = "failed"
            _save_summary_checkpoint(args.output_folder, summary)
        else:
            try:
                idmrg_info = run_tenax_idmrg_x_from_finite_mpo(
                    mpo=tenax_mpo,
                    model_spec=model_spec,
                    max_bond_dimension=args.max_bond_dimension,
                    max_iterations=args.idmrg_max_iterations,
                    bulk_kind=args.idmrg_bulk_kind,
                    max_local_dim=args.idmrg_max_local_dim,
                    show_progress=show_progress,
                )
                summary["idmrg"] = idmrg_info
                if isinstance(idmrg_info.get("entanglement"), dict):
                    entropy_profiles["iDMRG-x"] = idmrg_info["entanglement"]
                summary["stages"]["idmrg"] = "completed"
                _save_summary_checkpoint(args.output_folder, summary)
                _save_plot_step(
                    summary,
                    args.output_folder,
                    "dmrg_vs_idmrg_energy_png",
                    output_filename("dmrg_vs_idmrg_energy.png"),
                    lambda path: save_multi_method_energy_comparison(
                        method_to_energy={
                            "DMRG": float(summary["dmrg"]["energy_per_site"]),
                            "iDMRG-x": float(idmrg_info["energy_per_original_site"]),
                        },
                        filepath=path,
                        title="Finite DMRG vs iDMRG-x Energy Per Site",
                        title_label=run_plot_title_label,
                    ),
                    overwrite_existing,
                    continue_on_plot_error,
                )
            except Exception as exc:
                summary["idmrg"] = {"status": "failed", "error": str(exc)}
                summary["stages"]["idmrg"] = "failed"
                if not continue_on_plot_error:
                    raise
                _save_summary_checkpoint(args.output_folder, summary)

    # Optional ED workflow (runs after all DMRG outputs are already saved).
    if args.run_ed:
        summary["stages"]["ed"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        local_dim = int(model_spec.physical_dim)
        hilbert_dim = int(local_dim ** geometry.number_of_sites)
        if geometry.number_of_sites > MAX_ED_SITES:
            summary["ed"] = {
                "status": "skipped",
                "reason": f"ED is limited to {MAX_ED_SITES} sites or fewer.",
            }
            summary["stages"]["ed"] = "skipped"
        elif hilbert_dim > MAX_ED_HILBERT_DIM:
            summary["ed"] = {
                "status": "skipped",
                "reason": (
                    f"ED Hilbert-space dimension {hilbert_dim} exceeds limit {MAX_ED_HILBERT_DIM} "
                    f"(local_dim={local_dim}, sites={geometry.number_of_sites})."
                ),
            }
            summary["stages"]["ed"] = "skipped"
        else:
            try:
                if backend_used == "tenax":
                    ed_energy, ed_state = run_small_cluster_exact_diagonalization(
                        geometry=geometry,
                        model_spec=model_spec,
                        alpha=args.alpha,
                        beta=args.beta,
                        coupling_j=args.coupling_j,
                        show_progress=show_progress,
                    )
                    ed_correlations = collect_correlation_matrices_from_ed(
                        geometry,
                        ed_state,
                        model_spec=model_spec,
                        show_progress=show_progress,
                    )
                    ed_scalar_correlations = build_spin_orbital_scalar_correlations(ed_correlations)
                    ed_bond_rows = all_bond_energies(
                        geometry,
                        ed_correlations,
                        model_spec,
                        args.alpha,
                        args.beta,
                        args.coupling_j,
                        show_progress=show_progress,
                        progress_desc="ED bond energies",
                    )
                    ed_structure_factor_rows = all_high_symmetry_structure_factors(
                        ed_scalar_correlations,
                        geometry,
                        lattice=lattice_name,
                        show_progress=show_progress,
                        progress_desc="ED structure factors",
                    )
                else:
                    yl = _load_tenpy_backend_module()
                    stage_start = _start_stage("TeNPy fallback ED", show_progress)
                    ed_energy, ed_state = yl.run_small_cluster_exact_diagonalization(
                        geometry=geometry,
                        alpha=args.alpha,
                        beta=args.beta,
                        coupling_j=args.coupling_j,
                    )
                    ed_correlations = yl.collect_correlation_matrices_from_ed(geometry, ed_state)
                    ed_scalar_native = yl.build_spin_orbital_scalar_correlations(ed_correlations)
                    ed_scalar_correlations = {
                        "S": ed_scalar_native["spin_scalar"],
                        "T": ed_scalar_native["orbital_scalar"],
                        "ST": ed_scalar_native["mixed_scalar"],
                    }
                    ed_bond_rows = yl.all_bond_energies(geometry, ed_correlations, args.alpha, args.beta, args.coupling_j)
                    ed_structure_factor_rows = yl.all_high_symmetry_structure_factors(ed_scalar_native, geometry)
                    _end_stage("TeNPy fallback ED", stage_start, show_progress)

                ed_entropy_warning: str | None = None
                ed_entropy_profile: Dict[str, Any] | None = None
                try:
                    ed_entropy_profile = compute_ed_entropy_profile_from_state(
                        state=ed_state,
                        n_sites=geometry.number_of_sites,
                        local_dim=local_dim,
                        orders=ENTROPY_ORDERS,
                        show_progress=show_progress,
                    )
                    entropy_profiles["ED"] = ed_entropy_profile
                except Exception as exc:
                    ed_entropy_warning = f"Failed to compute ED entanglement profile: {exc}"

                summary["ed"] = {
                    "status": "completed",
                    "ground_state_energy": ed_energy,
                    "energy_per_site": ed_energy / geometry.number_of_sites,
                    "absolute_energy_difference_dmrg_minus_ed": abs(dmrg_energy - ed_energy),
                    "structure_factors": ed_structure_factor_rows,
                }
                if ed_entropy_profile is not None:
                    summary["ed"]["entanglement"] = ed_entropy_profile
                if ed_entropy_warning is not None:
                    summary["ed"]["entanglement_warning"] = ed_entropy_warning
                _save_summary_checkpoint(args.output_folder, summary)

                _save_plot_step(
                    summary,
                    args.output_folder,
                    "ed_bond_energy_diagram_png",
                    output_filename("ed_bond_energy_diagram.png"),
                    lambda path: save_bond_energy_diagram(geometry, ed_bond_rows, path, plot_title("ED Bond-Energy Diagram")),
                    overwrite_existing,
                    continue_on_plot_error,
                )
                _save_plot_step(
                    summary,
                    args.output_folder,
                    "ed_structure_factors_png",
                    output_filename("ed_structure_factors.png"),
                    lambda path: save_structure_factor_plot(ed_structure_factor_rows, path, plot_title("ED Structure Factors")),
                    overwrite_existing,
                    continue_on_plot_error,
                )
                _save_plot_step(
                    summary,
                    args.output_folder,
                    "ed_scalar_correlation_heatmaps_png",
                    output_filename("ed_scalar_correlation_heatmaps.png"),
                    lambda path: save_scalar_correlation_heatmaps(ed_scalar_correlations, path, f"ED | {run_plot_title_label}"),
                    overwrite_existing,
                    continue_on_plot_error,
                )
                _save_plot_step(
                    summary,
                    args.output_folder,
                    "dmrg_vs_ed_energy_png",
                    output_filename("dmrg_vs_ed_energy.png"),
                    lambda path: save_multi_method_energy_comparison(
                        method_to_energy={
                            "DMRG": float(summary["dmrg"]["energy_per_site"]),
                            "ED": float(summary["ed"]["energy_per_site"]),
                        },
                        filepath=path,
                        title="Finite DMRG vs ED Energy Per Site",
                        title_label=run_plot_title_label,
                    ),
                    overwrite_existing,
                    continue_on_plot_error,
                )
                if (
                    isinstance(summary.get("idmrg"), dict)
                    and summary["idmrg"].get("status") == "completed"
                    and "energy_per_original_site" in summary["idmrg"]
                ):
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "dmrg_vs_ed_vs_idmrg_energy_png",
                        output_filename("dmrg_vs_ed_vs_idmrg_energy.png"),
                        lambda path: save_multi_method_energy_comparison(
                            method_to_energy={
                                "DMRG": float(summary["dmrg"]["energy_per_site"]),
                                "ED": float(summary["ed"]["energy_per_site"]),
                                "iDMRG-x": float(summary["idmrg"]["energy_per_original_site"]),
                            },
                            filepath=path,
                            title="Finite DMRG vs ED vs iDMRG-x Energy Per Site",
                            title_label=run_plot_title_label,
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                _save_plot_step(
                    summary,
                    args.output_folder,
                    "dmrg_vs_ed_structure_factors_png",
                    output_filename("dmrg_vs_ed_structure_factors.png"),
                    lambda path: save_dmrg_ed_structure_comparison(
                        dmrg_structure_factor_rows,
                        ed_structure_factor_rows,
                        path,
                        title_label=run_plot_title_label,
                    ),
                    overwrite_existing,
                    continue_on_plot_error,
                )
            except Exception as exc:
                summary["ed"] = {"status": "failed", "error": str(exc)}
                summary["stages"]["ed"] = "failed"
                if not continue_on_plot_error:
                    raise
            if summary["stages"]["ed"] != "failed":
                summary["stages"]["ed"] = "completed"
        _save_summary_checkpoint(args.output_folder, summary)

    if len(entropy_profiles) > 0:
        _save_plot_step(
            summary,
            args.output_folder,
            "entanglement_entropy_profiles_png",
            output_filename("entanglement_entropy_profiles.png"),
            lambda path: save_entropy_profiles_comparison(
                entropy_profiles=entropy_profiles,
                filepath=path,
                orders=ENTROPY_ORDERS,
                title_label=run_plot_title_label,
            ),
            overwrite_existing,
            continue_on_plot_error,
        )
        _save_plot_step(
            summary,
            args.output_folder,
            "entanglement_entropy_method_means_png",
            output_filename("entanglement_entropy_method_means.png"),
            lambda path: save_entropy_method_means_comparison(
                entropy_profiles=entropy_profiles,
                filepath=path,
                orders=ENTROPY_ORDERS,
                title_label=run_plot_title_label,
            ),
            overwrite_existing,
            continue_on_plot_error,
        )

    failed_outputs = [
        key
        for key, item in summary.get("output_status", {}).items()
        if isinstance(item, dict) and item.get("status") == "failed"
    ]
    if (
        failed_outputs
        or (isinstance(summary.get("ed"), dict) and summary["ed"].get("status") == "failed")
        or (isinstance(summary.get("idmrg"), dict) and summary["idmrg"].get("status") == "failed")
    ):
        summary["run_status"] = "completed_with_warnings"
    else:
        summary["run_status"] = "completed"
    _save_summary_checkpoint(args.output_folder, summary)
    print(json.dumps(to_json_compatible(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
