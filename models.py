#!/usr/bin/env python3
"""Shared physics layer for the Yao-Lee driver.

This module owns model specifications, local operators, external-field term
construction, lattice geometry, correlation post-processing, and structure
factors. ED lives in ``ed_backend.py``; Tenax-specific MPO/DMRG code lives in
``tenax_backend.py``; TeNPy code lives in ``tenpy_backend.py``; scan analysis
belongs in ``analysis.py``; PNG output code belongs in ``plot_outputs.py``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from analysis import _make_progress_bar

AXIS_OPTIONS = ("x", "y", "z")
AXES = AXIS_OPTIONS
SPIN_REP_VALUES = {"1/2": 0.5, "3/2": 1.5}
# "1" is kept as a legacy alias and normalized to "0".
ORBITAL_REP_VALUES = {"0": 0.0, "1": 0.0, "1/2": 0.5}
SPIN_ONLY_MODEL_FAMILIES = ("heisenberg", "xy", "xxz", "xyz")
U1_SYMMETRY_MODES = ("u1", "u1_sz", "u1_tz")
SYMMETRY_MODE_OPTIONS = ("none", "auto") + U1_SYMMETRY_MODES + ("z2",)
U1_CHARGE_TZ_STRIDE = 4096
SPIN_REP_DEFAULT = "1/2"
ORBITAL_REP_DEFAULT = "1/2"
MODEL_FAMILY_DEFAULT = "yao_lee"
MODEL_FAMILY_OPTIONS = ("yao_lee", "ising_like") + tuple(SPIN_ONLY_MODEL_FAMILIES)
ISING_AXIS_DEFAULT = "z"
SPIN_REP = SPIN_REP_DEFAULT
ORBITAL_REP = ORBITAL_REP_DEFAULT
MODEL_FAMILY = MODEL_FAMILY_DEFAULT
ISING_AXIS = ISING_AXIS_DEFAULT
EXTERNAL_FIELD_TREATMENT_OPTIONS = ("off", "perturbation", "hamiltonian")
EXTERNAL_FIELD_AXIS_OPTIONS = ("custom", "111")


def build_spin_only_bond_terms(
    model_family: str,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
) -> List[Tuple[float, str]]:
    """Return spin-only bond terms for simple benchmark models."""
    family = str(model_family).strip().lower()
    j_scale = float(coupling_j)
    if family == "heisenberg":
        return [
            (j_scale, "Sx"),
            (j_scale, "Sy"),
            (j_scale, "Sz"),
        ]
    if family == "xy":
        return [
            (j_scale * float(jx), "Sx"),
            (j_scale * float(jy), "Sy"),
        ]
    if family == "xxz":
        return [
            (j_scale * float(jx), "Sx"),
            (j_scale * float(jy), "Sy"),
            (j_scale * float(jz), "Sz"),
        ]
    if family == "xyz":
        return [
            (j_scale * float(jx), "Sx"),
            (j_scale * float(jy), "Sy"),
            (j_scale * float(jz), "Sz"),
        ]
    raise ValueError(
        f"Unsupported spin-only model family '{model_family}'. "
        f"Choose from: {', '.join(SPIN_ONLY_MODEL_FAMILIES)}."
    )

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
    if family not in MODEL_FAMILY_OPTIONS:
        raise ValueError(f"model_family must be one of: {', '.join(MODEL_FAMILY_OPTIONS)}.")

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
    if text in ("auto", "best", "detect"):
        return "auto"
    if text in ("u1", "u(1)", "u1_pair", "u1x", "u1xu1", "u1_sz_tz", "u1_sztz"):
        return "u1"
    if text in ("u1_sz", "u1-spin", "u1_spin", "sz", "spin_u1", "u1s"):
        return "u1_sz"
    if text in ("u1_tz", "u1_tau", "u1_orbital", "tz", "tau_z", "orbital_u1", "u1t"):
        return "u1_tz"
    if text in ("z2", "z_2", "z(2)", "parity"):
        return "z2"
    raise ValueError(f"Unsupported symmetry mode '{mode}'. Choose from: {', '.join(SYMMETRY_MODE_OPTIONS)}.")


def _is_u1_symmetry_mode(mode: str | None) -> bool:
    return _normalize_symmetry_mode(mode) in U1_SYMMETRY_MODES


def _m2_values_from_spin_value(spin_value: float) -> List[int]:
    two_s = int(round(2.0 * spin_value))
    # Local basis convention used throughout this project:
    # lowest m first.  For spin/orbital 1/2 this is down, up with charges
    # [-1, +1], matching the bitwise ED convention 0=down, 1=up.
    return [-two_s + 2 * index for index in range(two_s + 1)]


def _get_z2_symmetry_object() -> Any:
    """Return a Z2 symmetry object from Tenax (direct class or ZnSymmetry(2) fallback)."""
    try:
        from tenax.core.symmetry import Z2Symmetry  # type: ignore
        return Z2Symmetry()
    except Exception:
        from tenax.core.symmetry import ZnSymmetry
        return ZnSymmetry(2)


def _encode_u1_charge_pair(sz2: int, tz2: int) -> int:
    """Additively encode (2*Sz, 2*Tz) as one Tenax U1 integer charge."""
    return int(sz2) * int(U1_CHARGE_TZ_STRIDE) + int(tz2)


def _u1_charge_encoding_summary() -> Dict[str, Any]:
    return {
        "scheme": "additive_integer_pair",
        "formula": "q = U1_CHARGE_TZ_STRIDE * (2*Sz) + (2*Tz)",
        "U1_CHARGE_TZ_STRIDE": int(U1_CHARGE_TZ_STRIDE),
        "reason": (
            "Tenax AutoMPO computes U1 operator charge by raw integer differences; "
            "packed ProductSymmetry charges are not additive under that operation."
        ),
    }


def _u1_encoded_phys_charges_for_model(model_spec: ModelSpec) -> np.ndarray:
    """Encode (2*Sz, 2*Tz) into one additive integer charge for Tenax U(1)."""
    spin_m2_values = _m2_values_from_spin_value(model_spec.spin_value)
    orb_m2_values = _m2_values_from_spin_value(model_spec.orbital_value)
    encoded: List[int] = []
    for sz2 in spin_m2_values:
        for tz2 in orb_m2_values:
            encoded.append(_encode_u1_charge_pair(int(sz2), int(tz2)))
    return np.asarray(encoded, dtype=np.int32)


def _u1_sz_phys_charges_for_model(model_spec: ModelSpec) -> np.ndarray:
    """Physical U1 charge q = 2*Sz, independent of orbital tau_z."""
    spin_m2_values = _m2_values_from_spin_value(model_spec.spin_value)
    orb_m2_values = _m2_values_from_spin_value(model_spec.orbital_value)
    return np.asarray(
        [int(sz2) for sz2 in spin_m2_values for _tz2 in orb_m2_values],
        dtype=np.int32,
    )


def _u1_tz_phys_charges_for_model(model_spec: ModelSpec) -> np.ndarray:
    """Physical U1 charge q = 2*tau_z/Tz, independent of spin Sz."""
    spin_m2_values = _m2_values_from_spin_value(model_spec.spin_value)
    orb_m2_values = _m2_values_from_spin_value(model_spec.orbital_value)
    return np.asarray(
        [int(tz2) for _sz2 in spin_m2_values for tz2 in orb_m2_values],
        dtype=np.int32,
    )


def _u1_phys_charges_for_model(model_spec: ModelSpec, symmetry_mode: str) -> np.ndarray:
    mode = _normalize_symmetry_mode(symmetry_mode)
    if mode == "u1":
        return _u1_encoded_phys_charges_for_model(model_spec)
    if mode == "u1_sz":
        return _u1_sz_phys_charges_for_model(model_spec)
    if mode == "u1_tz":
        return _u1_tz_phys_charges_for_model(model_spec)
    raise ValueError(f"Mode '{mode}' is not a U1 symmetry mode.")


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
    sz_charges = _u1_sz_phys_charges_for_model(model_spec)
    tz_charges = _u1_tz_phys_charges_for_model(model_spec)
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
                    "u1_sz_charge": int(sz_charges[idx]),
                    "u1_tz_charge": int(tz_charges[idx]),
                    "u1_charge_encoding": "q = 4096*(2*Sz) + (2*Tz)",
                }
            )
            idx += 1
    return table


def _u1_target_charge_for_mode(
    total_sz2: int,
    total_tz2: int,
    symmetry_mode: str,
) -> int:
    mode = _normalize_symmetry_mode(symmetry_mode)
    if mode == "u1":
        return _u1_encoded_target_charge(int(total_sz2), int(total_tz2))
    if mode == "u1_sz":
        return int(total_sz2)
    if mode == "u1_tz":
        return int(total_tz2)
    raise ValueError(f"Mode '{mode}' is not a U1 symmetry mode.")


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
    return _encode_u1_charge_pair(int(total_sz2), int(total_tz2))


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
    for spin_op in ("Sx", "Sy", "Sz", "Sp", "Sm"):
        for orbital_op in ("Tx", "Ty", "Tz", "Tp", "Tm"):
            ops[f"{spin_op}{orbital_op}"] = ops[spin_op] @ ops[orbital_op]

    # Legacy aliases for same-axis spin-orbital operators.
    ops["STx"] = ops["SxTx"]
    ops["STy"] = ops["SyTy"]
    ops["STz"] = ops["SzTz"]
    ops["STp"] = ops["SpTp"]
    ops["STm"] = ops["SmTm"]
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
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
) -> List[Tuple[float, str]]:
    """Legacy same-operator bond terms.

    New Hamiltonian builders should call ``two_site_operator_terms_for_bond`` so
    U(1)-symmetric ladder-operator terms such as ``Sp_i Sm_j`` can be represented
    exactly.  This helper is kept for spin-only benchmarks and old diagnostics.
    """
    axis_gamma = str(gamma).lower()
    if axis_gamma not in AXES:
        raise ValueError(f"Unknown bond axis '{gamma}'.")

    family = str(model_spec.model_family).strip().lower()
    if family in SPIN_ONLY_MODEL_FAMILIES:
        if not is_trivial_orbital(model_spec):
            raise ValueError(
                f"model_family='{family}' is spin-only and requires orbital_rep=0 (legacy alias 1 also accepted)."
            )
        return build_spin_only_bond_terms(family, coupling_j=coupling_j, jx=jx, jy=jy, jz=jz)

    # Requested behavior:
    # orbital_rep == "1" means no orbital DOF, and Yao-Lee orbital-dependent
    # terms reduce to spin-only Ising-like couplings.
    if is_trivial_orbital(model_spec):
        axis = model_spec.ising_axis
        return [(coupling_j * (1.0 + beta), f"S{axis}")]

    if family == "ising_like":
        axis = model_spec.ising_axis
        return [
            (coupling_j * (1.0 + beta), f"S{axis}"),
            (coupling_j * (1.0 - beta), f"T{axis}"),
            (coupling_j * alpha, f"ST{axis}"),
        ]
    if family != "yao_lee":
        raise ValueError(f"Unsupported model family '{model_spec.model_family}'.")

    # Bond-dependent Yao-Lee-like channel.
    return [
        (coupling_j * (1.0 + beta), f"S{axis_gamma}"),
        (coupling_j * (1.0 - beta), f"T{axis_gamma}"),
        (coupling_j * alpha, f"ST{axis_gamma}"),
    ]


def _combine_spin_orbital_operator_names(spin_op: str, orbital_op: str) -> str:
    """Return the one-site product-operator name used by ``build_site_ops``."""
    spin_name = str(spin_op)
    orbital_name = str(orbital_op)
    if spin_name == "Id":
        return orbital_name
    if orbital_name == "Id":
        return spin_name
    return f"{spin_name}{orbital_name}"


def _spin_dot_two_site_terms(coefficient: complex) -> List[Tuple[complex, str, str]]:
    """Expansion of ``coefficient * S_i.S_j`` in U(1)-compatible operators."""
    coeff = complex(coefficient)
    return [
        (0.5 * coeff, "Sp", "Sm"),
        (0.5 * coeff, "Sm", "Sp"),
        (coeff, "Sz", "Sz"),
    ]


def _orbital_axis_two_site_terms(
    gamma: str,
    coefficient: complex,
    *,
    real_ladder_y: bool = True,
) -> List[Tuple[complex, str, str]]:
    """Expansion of ``coefficient * T_i^gamma T_j^gamma``.

    The orbital sector is not charge-conserved.  The y-channel is nevertheless
    expanded in ``Tp/Tm`` by default so Tenax's U(1) AutoMPO path can stay real
    while representing exactly the same ``Ty_i Ty_j`` operator.
    """
    axis = str(gamma).strip().lower()
    coeff = complex(coefficient)
    if axis == "x":
        return [(coeff, "Tx", "Tx")]
    if axis == "z":
        return [(coeff, "Tz", "Tz")]
    if axis != "y":
        raise ValueError(f"Unknown bond axis '{gamma}'.")
    if not real_ladder_y:
        return [(coeff, "Ty", "Ty")]
    return [
        (-0.25 * coeff, "Tp", "Tp"),
        (0.25 * coeff, "Tp", "Tm"),
        (0.25 * coeff, "Tm", "Tp"),
        (-0.25 * coeff, "Tm", "Tm"),
    ]


def yao_lee_u1_two_site_terms_for_bond(
    gamma: str,
    alpha: float,
    beta: float,
    coupling_j: float,
) -> List[Tuple[complex, str, str]]:
    """Canonical spin/orbital Yao-Lee bond terms preserving total spin Sz.

    Formula:
        J * [(1+beta) S_i.S_j
             + (1-beta) T_i^gamma T_j^gamma
             + alpha (S_i.S_j)(T_i^gamma T_j^gamma)]

    The spin part is written with ``Sp/Sm/Sz`` so every term has net spin-U(1)
    charge zero.  The orbital part is unrestricted; ``Ty Ty`` is exactly
    represented through the real ``Tp/Tm`` expansion.
    """
    axis = str(gamma).strip().lower()
    if axis not in AXES:
        raise ValueError(f"Unknown bond axis '{gamma}'.")

    spin_terms = _spin_dot_two_site_terms(float(coupling_j) * (1.0 + float(beta)))
    orbital_terms = _orbital_axis_two_site_terms(
        axis,
        float(coupling_j) * (1.0 - float(beta)),
        real_ladder_y=True,
    )
    mixed_spin_terms = _spin_dot_two_site_terms(1.0)
    mixed_orbital_terms = _orbital_axis_two_site_terms(axis, 1.0, real_ladder_y=True)

    terms: List[Tuple[complex, str, str]] = []
    terms.extend(spin_terms)
    terms.extend(orbital_terms)
    for spin_coeff, spin_i, spin_j in mixed_spin_terms:
        for orbital_coeff, orbital_i, orbital_j in mixed_orbital_terms:
            terms.append(
                (
                    complex(float(coupling_j) * float(alpha)) * spin_coeff * orbital_coeff,
                    _combine_spin_orbital_operator_names(spin_i, orbital_i),
                    _combine_spin_orbital_operator_names(spin_j, orbital_j),
                )
            )
    return [
        (complex(coeff), op_i, op_j)
        for coeff, op_i, op_j in terms
        if not _is_zero_coefficient(coeff)
    ]


def two_site_operator_terms_for_bond(
    gamma: str,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
) -> List[Tuple[complex, str, str]]:
    """Return explicit two-site Hamiltonian terms ``(coeff, op_i, op_j)``.

    Unlike the legacy ``model_terms_for_bond`` helper, this supports different
    one-site operators on the two sites, which is required for U(1)-symmetric
    ladder-operator MPOs.
    """
    family = str(model_spec.model_family).strip().lower()
    if family == "yao_lee" and not is_trivial_orbital(model_spec):
        return yao_lee_u1_two_site_terms_for_bond(gamma, alpha, beta, coupling_j)
    return [
        (complex(coefficient), str(op_name), str(op_name))
        for coefficient, op_name in model_terms_for_bond(
            gamma,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx=jx,
            jy=jy,
            jz=jz,
        )
        if not _is_zero_coefficient(coefficient)
    ]


def auto_mpo_terms_for_bond(
    gamma: str,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    i: int,
    j: int,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
) -> List[Tuple[Any, str, int, str, int]]:
    """Return Tenax/AutoMPO-ready terms for one geometry bond."""
    terms: List[Tuple[Any, str, int, str, int]] = []
    for coefficient, op_i, op_j in two_site_operator_terms_for_bond(
        gamma,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
    ):
        coeff = _real_scalar_if_close(coefficient)
        if int(i) <= int(j):
            terms.append((coeff, str(op_i), int(i), str(op_j), int(j)))
        else:
            terms.append((coeff, str(op_j), int(j), str(op_i), int(i)))
    return terms


def _normalize_external_field_treatment(treatment: str | None) -> str:
    text = str(treatment if treatment is not None else "off").strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "record": "perturbation",
        "annotate": "perturbation",
        "perturbative": "perturbation",
        "exact": "hamiltonian",
        "solve": "hamiltonian",
        "on": "hamiltonian",
    }
    text = aliases.get(text, text)
    if text not in EXTERNAL_FIELD_TREATMENT_OPTIONS:
        raise ValueError(
            f"Unsupported external_field_treatment '{treatment}'. "
            f"Choose from: {', '.join(EXTERNAL_FIELD_TREATMENT_OPTIONS)}."
        )
    return text


def _normalize_external_field_axis(axis: str | None) -> str:
    text = str(axis if axis is not None else "custom").strip().lower().replace("[", "").replace("]", "")
    if text in ("custom", "xyz", "component", "components"):
        return "custom"
    if text in ("111", "1,1,1", "1 1 1"):
        return "111"
    raise ValueError(f"Unsupported external_field_axis '{axis}'. Choose from: custom, 111.")


def external_field_vector(
    axis: str,
    strength: float,
    hx: float,
    hy: float,
    hz: float,
) -> Tuple[float, float, float]:
    axis_mode = _normalize_external_field_axis(axis)
    if axis_mode == "111":
        component = float(strength) / float(np.sqrt(3.0))
        return component, component, component
    return float(hx), float(hy), float(hz)


def external_field_is_active(treatment: str, field_vector: Tuple[float, float, float]) -> bool:
    if _normalize_external_field_treatment(treatment) == "off":
        return False
    return any(abs(float(component)) > 1e-14 for component in field_vector)


def external_field_terms_for_model(
    field_vector: Tuple[float, float, float],
    mu_b: float,
    field_sign: float,
    sigma_factor: float,
) -> List[Tuple[float, str]]:
    hx, hy, hz = [float(component) for component in field_vector]
    prefactor = float(field_sign) * float(mu_b) * float(sigma_factor)
    terms = [
        (prefactor * hx, "Sx"),
        (prefactor * hy, "Sy"),
        (prefactor * hz, "Sz"),
    ]
    return [(coefficient, op_name) for coefficient, op_name in terms if abs(float(coefficient)) > 1e-14]


def validate_external_field_symmetry_compatibility(
    field_terms: List[Tuple[float, str]],
    symmetry_mode: str,
) -> None:
    mode = _normalize_symmetry_mode(symmetry_mode)
    if mode in ("none", "auto") or len(field_terms) == 0:
        return
    if mode == "u1_tz":
        return
    breaking_terms = [op_name for _, op_name in field_terms if op_name in ("Sx", "Sy")]
    if breaking_terms:
        raise ValueError(
            "An external field with hx/hy components breaks the strict U1/Z2 sectors used by this script. "
            "Use external_field_treatment=perturbation to annotate it without changing the symmetric solve, "
            "or set symmetry_mode=none when external_field_treatment=hamiltonian."
        )


def _external_field_float_filename_token(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace("+", "")
    return text.replace(".", "p")


def _safe_external_field_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "run"


def external_field_filename_label(
    treatment: str,
    axis: str,
    field_vector: Tuple[float, float, float],
) -> str | None:
    if not external_field_is_active(treatment, field_vector):
        return None
    hx, hy, hz = field_vector
    axis_mode = _normalize_external_field_axis(axis)
    if axis_mode == "111":
        magnitude = float(np.sqrt(hx * hx + hy * hy + hz * hz))
        return _safe_external_field_token(
            f"H111_{_external_field_float_filename_token(magnitude)}_{treatment}"
        )
    return _safe_external_field_token(
        "Hxyz_"
        f"hx{_external_field_float_filename_token(hx)}_"
        f"hy{_external_field_float_filename_token(hy)}_"
        f"hz{_external_field_float_filename_token(hz)}_"
        f"{treatment}"
    )


def external_field_display_label(
    treatment: str,
    axis: str,
    field_vector: Tuple[float, float, float],
) -> str | None:
    if not external_field_is_active(treatment, field_vector):
        return None
    hx, hy, hz = field_vector
    axis_text = "[111]" if _normalize_external_field_axis(axis) == "111" else "custom"
    return f"Hz field {treatment}, axis={axis_text}, H=({hx:.4g}, {hy:.4g}, {hz:.4g})"


def external_field_construction_summary(
    treatment: str,
    axis: str,
    field_vector: Tuple[float, float, float],
    mu_b: float,
    field_sign: float,
    sigma_factor: float,
    field_terms: List[Tuple[float, str]],
) -> Dict[str, Any]:
    active = external_field_is_active(treatment, field_vector)
    return {
        "treatment": _normalize_external_field_treatment(treatment),
        "axis": _normalize_external_field_axis(axis),
        "field_vector_hx_hy_hz": [float(value) for value in field_vector],
        "active": bool(active),
        "mu_B": float(mu_b),
        "field_sign": float(field_sign),
        "sigma_factor": float(sigma_factor),
        "formula": (
            "H_Z = field_sign * mu_B * sigma_factor * sum_i "
            "(hx*Sx_i + hy*Sy_i + hz*Sz_i); orbital Zeeman coupling is omitted because eg L=0."
        ),
        "model_insertion": (
            "not inserted; recorded as perturbation only"
            if _normalize_external_field_treatment(treatment) == "perturbation"
            else ("inserted as one-site spin terms" if field_terms else "off or zero field")
        ),
        "hamiltonian_field_terms": [
            {"coefficient": float(coefficient), "operator": op_name}
            for coefficient, op_name in field_terms
        ],
        "perturbative_assumption": (
            "vison gap finite; field treated outside the unperturbed bond Hamiltonian unless treatment=hamiltonian"
        ),
    }


def _is_zero_coefficient(value: complex, tol: float = 1e-12) -> bool:
    return abs(complex(value)) <= tol


def _real_scalar_if_close(value: complex, tol: float = 1e-12) -> float | complex:
    coeff = complex(value)
    if abs(coeff.imag) <= tol:
        return float(coeff.real)
    return coeff


def nonzero_bond_terms(
    bond_terms: List[Tuple[float, str]],
    tol: float = 1e-12,
) -> List[Tuple[float, str]]:
    """Drop exactly inactive Hamiltonian channels before MPO construction."""
    return [
        (coefficient, op_name)
        for coefficient, op_name in bond_terms
        if not _is_zero_coefficient(coefficient, tol=tol)
    ]


def nonzero_auto_mpo_terms(
    terms: List[Tuple[Any, ...]],
    tol: float = 1e-12,
) -> List[Tuple[Any, ...]]:
    """Return AutoMPO terms with non-zero scalar coefficients.

    Tenax raises a low-level "No terms" error if the physical Hamiltonian is
    empty. Keeping this filter in the physics layer makes dense, Z2, and U1 MPO
    builds agree about which channels are actually active.
    """
    active_terms: List[Tuple[Any, ...]] = []
    for term in terms:
        if len(term) == 0:
            continue
        try:
            if _is_zero_coefficient(term[0], tol=tol):
                continue
        except (TypeError, ValueError):
            # Non-numeric coefficients are left for the backend/API to validate.
            pass
        active_terms.append(term)
    return active_terms


def _u1_pair_terms_for_bond_terms(
    bond_terms: List[Tuple[float, str]],
    i: int,
    j: int,
    symmetry_mode: str,
) -> List[Tuple[complex, str, int, str, int]]:
    """Convert U1-conserving x/y pairs into raising/lowering AutoMPO terms."""
    mode = _normalize_symmetry_mode(symmetry_mode)
    if not _is_u1_symmetry_mode(mode):
        raise ValueError(f"Mode '{mode}' is not a U1 symmetry mode.")

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
    charged_families = {
        "u1": {"S", "T"},
        "u1_sz": {"S"},
        "u1_tz": {"T"},
    }[mode]

    def append_y_axis_expansion(coeff: complex, plus_op: str, minus_op: str) -> None:
        terms.extend(
            [
                (_real_scalar_if_close(-0.25 * coeff), plus_op, i, plus_op, j),
                (_real_scalar_if_close(0.25 * coeff), plus_op, i, minus_op, j),
                (_real_scalar_if_close(0.25 * coeff), minus_op, i, plus_op, j),
                (_real_scalar_if_close(-0.25 * coeff), minus_op, i, minus_op, j),
            ]
        )

    for family, x_op, y_op, plus_op, minus_op in (
        ("S", "Sx", "Sy", "Sp", "Sm"),
        ("T", "Tx", "Ty", "Tp", "Tm"),
    ):
        coeff_x = combined.pop(x_op, 0.0j)
        coeff_y = combined.pop(y_op, 0.0j)
        if _is_zero_coefficient(coeff_x) and _is_zero_coefficient(coeff_y):
            continue
        if family in charged_families:
            if _is_zero_coefficient(coeff_x - coeff_y):
                coeff = 0.5 * coeff_x
                terms.append((_real_scalar_if_close(coeff), plus_op, i, minus_op, j))
                terms.append((_real_scalar_if_close(coeff), minus_op, i, plus_op, j))
                continue
            raise ValueError(
                f"A {mode.upper()} symmetric MPO requires charged transverse pair terms "
                f"to appear as {x_op}_{i}{x_op}_{j} + {y_op}_{i}{y_op}_{j} "
                f"with equal coefficients. Got {x_op} coefficient {coeff_x:g} "
                f"and {y_op} coefficient {coeff_y:g}."
            )
        if not _is_zero_coefficient(coeff_x):
            terms.append((_real_scalar_if_close(coeff_x), x_op, i, x_op, j))
        if not _is_zero_coefficient(coeff_y):
            append_y_axis_expansion(coeff_y, plus_op, minus_op)
        continue

    for op_name in ("STx", "STy"):
        coeff = combined.get(op_name, 0.0j)
        if not _is_zero_coefficient(coeff):
            raise ValueError(
                f"{mode.upper()} symmetry cannot preserve the current {op_name}_{i}{op_name}_{j} "
                "term without changing the Hamiltonian. Use symmetry_mode=z2 or none for "
                "Yao-Lee x/y spin-orbital channels."
            )

    for op_name in order:
        coeff = combined.get(op_name, 0.0j)
        if _is_zero_coefficient(coeff):
            continue
        terms.append((_real_scalar_if_close(coeff), op_name, i, op_name, j))
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
    active_bond_terms = nonzero_bond_terms(bond_terms)
    if _is_u1_symmetry_mode(mode):
        return list(_u1_pair_terms_for_bond_terms(active_bond_terms, i, j, mode))
    return [(coefficient, op_name, i, op_name, j) for coefficient, op_name in active_bond_terms]


def _reachable_charge_sums(single_site_values: List[int], n_sites: int) -> List[int]:
    reachable = {0}
    values = [int(value) for value in single_site_values]
    for _site in range(max(0, int(n_sites))):
        reachable = {int(total + value) for total in reachable for value in values}
    return sorted(reachable)


def _reachable_z2_sums(single_site_values: List[int], n_sites: int) -> List[int]:
    reachable = {0}
    values = [int(value) % 2 for value in single_site_values]
    for _site in range(max(0, int(n_sites))):
        reachable = {int((total + value) % 2) for total in reachable for value in values}
    return sorted(reachable)


def _append_limited_issue(
    issues: List[Dict[str, Any]],
    issue: Dict[str, Any],
    *,
    max_issues: int = 12,
) -> None:
    if len(issues) < int(max_issues):
        issues.append(issue)


def _auto_mpo_terms_for_symmetry_check(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    *,
    jx: float,
    jy: float,
    jz: float,
    external_field_terms: List[Tuple[float, str]],
    symmetry_mode: str,
) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
    mode = _normalize_symmetry_mode(symmetry_mode)
    terms: List[Tuple[Any, ...]] = []
    issues: List[Dict[str, Any]] = []
    for bond in geometry.bond_list:
        try:
            bond_auto_terms = auto_mpo_terms_for_bond(
                bond.gamma.lower(),
                model_spec,
                alpha,
                beta,
                coupling_j,
                bond.i,
                bond.j,
                jx=jx,
                jy=jy,
                jz=jz,
            )
            terms.extend(bond_auto_terms)
        except Exception as exc:
            _append_limited_issue(
                issues,
                {
                    "kind": "bond_term_conversion_failed",
                    "bond": {"i": int(bond.i), "j": int(bond.j), "gamma": str(bond.gamma)},
                    "formula": "canonical Sz-conserving Yao-Lee expansion"
                    if str(model_spec.model_family) == "yao_lee"
                    else "legacy same-operator two-site expansion",
                    "error": str(exc),
                },
            )

    for site in range(int(geometry.number_of_sites)):
        for coefficient, op_name in external_field_terms:
            terms.append((coefficient, op_name, site))

    return nonzero_auto_mpo_terms(terms), issues


def _check_terms_conserve_symmetry(
    terms: List[Tuple[Any, ...]],
    model_spec: ModelSpec,
    symmetry_mode: str,
) -> Tuple[bool, List[Dict[str, Any]]]:
    mode = _normalize_symmetry_mode(symmetry_mode)
    if _is_u1_symmetry_mode(mode):
        phys_charges = _u1_phys_charges_for_model(model_spec, mode)
    elif mode == "z2":
        phys_charges = _z2_phys_charges_for_model(model_spec)
    else:
        return True, []

    site_ops = build_site_ops(model_spec)
    issues: List[Dict[str, Any]] = []
    for term in terms:
        try:
            _validate_symmetry_conserving_terms(
                [term],
                site_ops,
                phys_charges,
                mode,
            )
        except Exception as exc:
            _append_limited_issue(
                issues,
                {
                    "kind": "term_not_conserved",
                    "term": [str(item) for item in term],
                    "error": str(exc),
                },
            )
    return len(issues) == 0, issues


def analyze_hamiltonian_symmetries(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    *,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    requested_symmetry_mode: str = "none",
    u1_target_total_sz2: int = 0,
    u1_target_total_tz2: int = 0,
    z2_target_parity: int = 0,
) -> Dict[str, Any]:
    """Strictly report which requested symmetry sectors match the Hamiltonian.

    This checks the actual finite-cluster terms after the same symmetry-aware
    ladder-operator conversion used by the Tenax MPO path.
    """

    requested_mode = _normalize_symmetry_mode(requested_symmetry_mode)
    n_sites = int(geometry.number_of_sites)
    field_terms = list(external_field_terms or [])
    spin_m2_values = _m2_values_from_spin_value(model_spec.spin_value)
    orbital_m2_values = _m2_values_from_spin_value(model_spec.orbital_value)
    reachable_sz2 = _reachable_charge_sums(spin_m2_values, n_sites)
    reachable_tz2 = _reachable_charge_sums(orbital_m2_values, n_sites)
    target_sz2 = int(u1_target_total_sz2)
    target_tz2 = int(u1_target_total_tz2)
    reachable_sz2_set = set(reachable_sz2)
    reachable_tz2_set = set(reachable_tz2)

    def u1_target_reachable_for_mode(mode: str) -> bool:
        if mode == "u1":
            return target_sz2 in reachable_sz2_set and target_tz2 in reachable_tz2_set
        if mode == "u1_sz":
            return target_sz2 in reachable_sz2_set
        if mode == "u1_tz":
            return target_tz2 in reachable_tz2_set
        return False

    def u1_target_sector_for_mode(mode: str) -> Dict[str, Any]:
        target_charge = int(_u1_target_charge_for_mode(target_sz2, target_tz2, mode))
        sector: Dict[str, Any] = {
            "mode": mode,
            "target_charge": target_charge,
            "reachable": bool(u1_target_reachable_for_mode(mode)),
        }
        if mode in ("u1", "u1_sz"):
            sector["total_Sz_times_2"] = target_sz2
        if mode in ("u1", "u1_tz"):
            sector["total_Tz_times_2"] = target_tz2
        if mode == "u1":
            sector["encoded_charge"] = target_charge
        return sector

    def build_u1_report(mode: str) -> Dict[str, Any]:
        mode = _normalize_symmetry_mode(mode)
        u1_terms, generation_issues = _auto_mpo_terms_for_symmetry_check(
            geometry,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx=jx,
            jy=jy,
            jz=jz,
            external_field_terms=field_terms,
            symmetry_mode=mode,
        )
        terms_conserved, term_issues = _check_terms_conserve_symmetry(
            u1_terms,
            model_spec,
            mode,
        )
        issues = list(generation_issues) + list(term_issues)
        conserved = len(generation_issues) == 0 and bool(terms_conserved)
        target_reachable = u1_target_reachable_for_mode(mode)
        if mode == "u1":
            conserved_key = "conserved_total_Sz_and_total_Tz"
            note = (
                "U1 here means simultaneous conservation of total Sz and orbital tau_z/Tz. "
                "Both spin and orbital transverse terms must be U1-paired."
            )
        elif mode == "u1_sz":
            conserved_key = "conserved_total_Sz"
            note = (
                "Spin-only U1 conserves total Sz while allowing orbital-only terms that do "
                "not change Sz. Spin transverse terms must still appear as conserving pairs."
            )
        else:
            conserved_key = "conserved_total_Tz"
            note = (
                "Orbital-only U1 conserves total tau_z/Tz while allowing spin-only terms that "
                "do not change Tz. Orbital transverse terms must still appear as conserving pairs."
            )
        return {
            "mode": mode,
            conserved_key: bool(conserved),
            "conserved": bool(conserved),
            "backend_supported_for_simplification": bool(conserved and target_reachable),
            "target_sector": u1_target_sector_for_mode(mode),
            "reachable_total_Sz_times_2": reachable_sz2,
            "reachable_total_Tz_times_2": reachable_tz2,
            "charge_encoding": _u1_charge_encoding_summary() if mode == "u1" else {
                "scheme": "single_integer_charge",
                "formula": "q = 2*Sz" if mode == "u1_sz" else "q = 2*Tz",
            },
            "basis_charge_table": _u1_basis_charge_table_for_model(model_spec),
            "checked_term_count": int(len(u1_terms)),
            "issues": issues,
            "note": note,
        }

    u1_report = build_u1_report("u1")
    u1_sz_report = build_u1_report("u1_sz")
    u1_tz_report = build_u1_report("u1_tz")

    z2_terms, z2_generation_issues = _auto_mpo_terms_for_symmetry_check(
        geometry,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
        external_field_terms=field_terms,
        symmetry_mode="z2",
    )
    z2_terms_conserved, z2_term_issues = _check_terms_conserve_symmetry(
        z2_terms,
        model_spec,
        "z2",
    )
    z2_issues = list(z2_generation_issues) + list(z2_term_issues)
    reachable_z2 = _reachable_z2_sums(
        [int(value) for value in _z2_phys_charges_for_model(model_spec)],
        n_sites,
    )
    target_z2 = int(z2_target_parity) % 2
    z2_target_reachable = target_z2 in set(reachable_z2)
    z2_conserved = len(z2_generation_issues) == 0 and bool(z2_terms_conserved)

    u1_reports_by_mode = {
        "u1": u1_report,
        "u1_sz": u1_sz_report,
        "u1_tz": u1_tz_report,
    }
    supported_u1_modes = [
        mode
        for mode in ("u1", "u1_sz", "u1_tz")
        if bool(u1_reports_by_mode[mode].get("backend_supported_for_simplification", False))
    ]
    if supported_u1_modes:
        recommended_mode = supported_u1_modes[0]
        recommendation_reason = (
            f"{recommended_mode} is conserved, its requested target sector is reachable, "
            "and Tenax can use it for U1 block-sparse simplification."
        )
    elif bool(z2_conserved) and bool(z2_target_reachable):
        recommended_mode = "none"
        recommendation_reason = (
            "Only Z2/parity is available, but this Tenax AutoMPO path cannot build "
            "a true Z2 block-sparse MPO; use dense mode or a backend with Z2 MPO support."
        )
    else:
        recommended_mode = "none"
        recommendation_reason = "No supported nontrivial symmetry sector passed the strict checks."

    def requested_u1_report() -> Dict[str, Any] | None:
        if requested_mode in u1_reports_by_mode:
            return u1_reports_by_mode[requested_mode]
        return None

    requested_report = requested_u1_report()
    requested_physical_symmetry_valid = (
        requested_mode in ("none", "auto")
        or (requested_report is not None and bool(requested_report.get("conserved", False))
            and bool((requested_report.get("target_sector") or {}).get("reachable", False)))
        or (
            requested_mode == "z2"
            and bool(z2_conserved)
            and bool(z2_target_reachable)
        )
    )
    requested_backend_supported = (
        requested_mode in ("none", "auto")
        or (requested_report is not None and bool(requested_report.get("backend_supported_for_simplification", False)))
    )

    return {
        "requested_mode": requested_mode,
        "requested_physical_symmetry_valid": bool(requested_physical_symmetry_valid),
        "requested_backend_supported_for_simplification": bool(requested_backend_supported),
        "recommended_mode_for_tenax": recommended_mode,
        "recommendation_reason": recommendation_reason,
        "u1": u1_report,
        "u1_sz": u1_sz_report,
        "u1_tz": u1_tz_report,
        "z2": {
            "conserved_global_parity": bool(z2_conserved),
            "backend_supported_for_simplification": False,
            "backend_support_note": (
                "Tenax 0.2 AutoMPO symmetric construction used here is U1-only; "
                "Z2 is reported as a physical conservation law but not used for block-sparse MPO speedup."
            ),
            "target_sector": {
                "global_parity": target_z2,
                "reachable": bool(z2_target_reachable),
            },
            "reachable_global_parities": reachable_z2,
            "basis_charge_table": _z2_basis_charge_table_for_model(model_spec),
            "checked_term_count": int(len(z2_terms)),
            "issues": z2_issues,
            "note": "Parity charge is (spin basis index + orbital basis index) mod 2.",
        },
    }


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
    length_x: int | None = None
    length_y: int | None = None
    circumference_x: bool = False
    circumference_y: bool = True


PLAQUETTE_FLUX_AXES = ("x", "y", "z")
PLAQUETTE_FLUX_NORMALIZATION = 2.0 ** 6


def _canonical_cycle_key(cycle: List[int]) -> Tuple[int, ...]:
    """Return an orientation-independent key for a simple cycle."""
    values = [int(site) for site in cycle]
    rotations: List[Tuple[int, ...]] = []
    for seq in (values, list(reversed(values))):
        for offset in range(len(seq)):
            rotations.append(tuple(seq[offset:] + seq[:offset]))
    return min(rotations)


def _honeycomb_bond_axis_map(geometry: GeometryData) -> Dict[Tuple[int, int], str]:
    axes: Dict[Tuple[int, int], str] = {}
    for bond in geometry.bond_list:
        i = int(bond.i)
        j = int(bond.j)
        axes[(min(i, j), max(i, j))] = str(bond.gamma).strip().lower()
    return axes


def _honeycomb_adjacency(geometry: GeometryData) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {site: [] for site in range(int(geometry.number_of_sites))}
    for bond in geometry.bond_list:
        i = int(bond.i)
        j = int(bond.j)
        adjacency[i].append(j)
        adjacency[j].append(i)
    for neighbors in adjacency.values():
        neighbors.sort()
    return adjacency


def _simple_cycles_of_length_six(geometry: GeometryData) -> List[List[int]]:
    """Find unique simple length-six cycles in the bond graph."""
    adjacency = _honeycomb_adjacency(geometry)
    cycles: Dict[Tuple[int, ...], List[int]] = {}

    def dfs(start: int, current: int, path: List[int]) -> None:
        if len(path) == 6:
            if start in adjacency[current]:
                key = _canonical_cycle_key(path)
                cycles.setdefault(key, list(key))
            return
        for neighbor in adjacency[current]:
            if neighbor == start:
                continue
            if neighbor in path:
                continue
            if neighbor < start:
                continue
            dfs(start, neighbor, path + [neighbor])

    for start in range(int(geometry.number_of_sites)):
        dfs(start, start, [start])
    return [list(cycle) for cycle in cycles.values()]


def _elementary_honeycomb_plaquette_cycles(geometry: GeometryData) -> List[List[int]] | None:
    """Build contractible elementary honeycomb hexagons from cell indices.

    On small periodic cylinders, the bond graph can contain non-contractible
    length-six loops around the circumference.  Those are useful paths, but
    they are not local plaquettes.  The elementary hexagon anchored at
    ``(x, y)`` uses the two neighboring x-cells and the next y-row:

        A(x,y), B(x,y+1), A(x,y+1), B(x+1,y+1), A(x+1,y), B(x+1,y).

    Missing sites or missing bonds are rejected by the caller, so open-y
    boundaries naturally drop incomplete plaquettes.
    """
    if len(getattr(geometry, "cell_indices", [])) != int(geometry.number_of_sites):
        return None
    if len(getattr(geometry, "sublattice_indices", [])) != int(geometry.number_of_sites):
        return None

    site_by_cell: Dict[Tuple[int, int, int], int] = {}
    x_values: set[int] = set()
    y_values: set[int] = set()
    for site, (cell, sublattice) in enumerate(zip(geometry.cell_indices, geometry.sublattice_indices)):
        if len(cell) != 2:
            return None
        x_cell = int(cell[0])
        y_cell = int(cell[1])
        sub = int(sublattice)
        site_by_cell[(x_cell, y_cell, sub)] = int(site)
        x_values.add(x_cell)
        y_values.add(y_cell)

    sorted_x = sorted(x_values)
    sorted_y = sorted(y_values)
    if len(sorted_x) < 2 or len(sorted_y) < 1:
        return []
    circumference_x = bool(getattr(geometry, "circumference_x", False))
    circumference_y = bool(getattr(geometry, "circumference_y", True))
    min_x = int(sorted_x[0])
    max_x = int(sorted_x[-1])
    min_y = int(sorted_y[0])
    max_y = int(sorted_y[-1])

    cycles: List[List[int]] = []
    for x_cell in sorted_x:
        x_next = int(x_cell) + 1
        if x_next not in x_values:
            if circumference_x and int(x_cell) == max_x:
                x_next = min_x
            else:
                continue
        if x_next == int(x_cell):
            continue
        for y_cell in sorted_y:
            y_next = int(y_cell) + 1
            if y_next not in y_values:
                if circumference_y and int(y_cell) == max_y:
                    y_next = min_y
                else:
                    continue
            if y_next == int(y_cell):
                continue
            plaquette_keys = [
                (x_cell, y_cell, 0),
                (x_cell, y_next, 1),
                (x_cell, y_next, 0),
                (x_next, y_next, 1),
                (x_next, y_cell, 0),
                (x_next, y_cell, 1),
            ]
            try:
                cycle = [site_by_cell[key] for key in plaquette_keys]
            except KeyError:
                continue
            if len(set(cycle)) == 6:
                cycles.append([int(site) for site in cycle])
    return cycles


def _plaquette_axes_for_cycle(
    cycle: List[int],
    bond_axes: Dict[Tuple[int, int], str],
) -> List[str] | None:
    axes: List[str] = []
    for index, site in enumerate(cycle):
        previous_site = cycle[(index - 1) % len(cycle)]
        next_site = cycle[(index + 1) % len(cycle)]
        edge_axes = {
            bond_axes.get((min(site, previous_site), max(site, previous_site))),
            bond_axes.get((min(site, next_site), max(site, next_site))),
        }
        if None in edge_axes:
            return None
        missing_axes = [axis for axis in PLAQUETTE_FLUX_AXES if axis not in edge_axes]
        if len(missing_axes) != 1:
            return None
        axes.append(missing_axes[0])
    return axes


def honeycomb_plaquette_flux_operators(geometry: GeometryData) -> List[Dict[str, Any]]:
    """Return normalized orbital plaquette-flux operators for honeycomb hexagons.

    The local orbital matrices in this project are spin-1/2 operators
    ``tau_a = sigma_a / 2``.  The flux diagnostic is normalized as
    ``W_p = prod_l (2 tau_l^{gamma_l})`` so the conserved values are near
    ``+/-1`` rather than ``+/-1/64``.

    Only elementary six-site honeycomb plaquettes are returned.  In
    particular, non-contractible length-six loops caused by small periodic
    circumferences are filtered out.
    """
    if int(geometry.number_of_sites) < 6:
        return []
    bond_axes = _honeycomb_bond_axis_map(geometry)
    plaquettes: List[Dict[str, Any]] = []
    elementary_cycles = _elementary_honeycomb_plaquette_cycles(geometry)
    candidate_cycles = (
        _simple_cycles_of_length_six(geometry)
        if elementary_cycles is None
        else elementary_cycles
    )
    if len(candidate_cycles) == 0:
        return []
    seen_cycles: set[Tuple[int, ...]] = set()
    for cycle in candidate_cycles:
        key = _canonical_cycle_key(cycle)
        if key in seen_cycles:
            continue
        seen_cycles.add(key)
        axes = _plaquette_axes_for_cycle(cycle, bond_axes)
        if axes is None:
            continue
        positions = np.asarray([geometry.positions[site] for site in cycle], dtype=float)
        center = np.mean(positions, axis=0)
        plaquettes.append(
            {
                "plaquette_index": 0,
                "sites": [int(site) for site in cycle],
                "axes": [str(axis) for axis in axes],
                "operator_names": [f"T{axis}" for axis in axes],
                "tenpy_operator_names": [f"tau_{axis}" for axis in axes],
                "center": [float(center[0]), float(center[1])],
                "normalization": float(PLAQUETTE_FLUX_NORMALIZATION),
                "definition": "W_p = product_l (2 tau_l^{gamma_l}) around one honeycomb hexagon",
            }
        )
    plaquettes.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1]), item["sites"]))
    for index, item in enumerate(plaquettes):
        item["plaquette_index"] = int(index)
    return plaquettes


def select_honeycomb_plaquette_flux_operator(
    geometry: GeometryData,
    plaquette_center_idx: int | None = None,
) -> Dict[str, Any]:
    """Select one plaquette-flux operator by sorted plaquette index.

    ``plaquette_center_idx=None`` chooses the central plaquette in the sorted
    list, which is useful for finite cylinders where boundary plaquettes may be
    less representative.
    """
    plaquettes = honeycomb_plaquette_flux_operators(geometry)
    if len(plaquettes) == 0:
        raise ValueError("No honeycomb length-six plaquette was found in this geometry.")
    index = len(plaquettes) // 2 if plaquette_center_idx is None else int(plaquette_center_idx)
    if index < 0:
        index = len(plaquettes) + index
    if index < 0 or index >= len(plaquettes):
        raise IndexError(
            f"plaquette_center_idx={plaquette_center_idx} is outside the available "
            f"plaquette range [0, {len(plaquettes) - 1}]."
        )
    return dict(plaquettes[index])


def plaquette_flux_close_to_target(
    value: Any,
    *,
    target: float = 1.0,
    tolerance: float = 0.15,
) -> bool:
    """Check whether a normalized plaquette flux is near its conserved value."""
    try:
        flux = float(np.real(value))
    except (TypeError, ValueError):
        return False
    if not np.isfinite(flux):
        return False
    target_value = float(target)
    tol = abs(float(tolerance))
    return bool(
        abs(flux - target_value) <= tol
        or abs(abs(flux) - abs(target_value)) <= tol
    )


def honeycomb_real_space_position(x_cell: int, y_cell: int, sublattice: int) -> np.ndarray:
    a = 1.0
    a1 = np.array([np.sqrt(3.0) * a, 0.0], dtype=float)
    a2 = np.array([np.sqrt(3.0) * a / 2.0, 3.0 * a / 2.0], dtype=float)
    delta_b = np.array([0.0, a], dtype=float)

    position = x_cell * a1 + y_cell * a2
    if sublattice == 1:
        position = position + delta_b
    return position


def _validate_lattice_length(name: str, value: int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _validate_length_y(length_y: int) -> int:
    value = int(length_y)
    if value <= 0:
        raise ValueError("length_y must be positive.")
    return value


def snake_y_values(x_cell: int, length_y: int):
    if x_cell % 2 == 0:
        return range(length_y)
    return range(length_y - 1, -1, -1)


def build_honeycomb_cylinder_geometry(
    length_x: int,
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> GeometryData:
    length_x = _validate_lattice_length("length_x", length_x)
    length_y_value = _validate_length_y(length_y)
    bond_list: List[Bond] = []
    positions: List[np.ndarray] = []
    cell_indices: List[Tuple[int, int]] = []
    sublattice_indices: List[int] = []
    n_sites = length_x * length_y_value * 2
    site_to_index: Dict[Tuple[int, int, int], int] = {}

    for x in range(length_x):
        for y in snake_y_values(x, length_y_value):
            for sub in (0, 1):
                site_to_index[(x, y, sub)] = len(positions)
                positions.append(honeycomb_real_space_position(x, y, sub))
                cell_indices.append((x, y))
                sublattice_indices.append(sub)
    if len(positions) != n_sites:
        raise RuntimeError("Internal geometry error: honeycomb snake ordering generated wrong site count.")

    for x in range(length_x):
        for y in range(length_y_value):
            i_a = site_to_index[(x, y, 0)]
            i_b = site_to_index[(x, y, 1)]
            bond_list.append(Bond(i_a, i_b, "z"))

            y_plus_1 = (y + 1) % length_y_value
            if circumference_y or (y + 1 < length_y_value):
                j_b_y = site_to_index[(x, y_plus_1, 1)]
                bond_list.append(Bond(i_a, j_b_y, "y"))

            x_plus_1 = (x + 1) % length_x
            if circumference_x or (x + 1 < length_x):
                j_b_x = site_to_index[(x_plus_1, y, 1)]
                bond_list.append(Bond(i_a, j_b_x, "x"))

    return GeometryData(
        number_of_sites=n_sites,
        bond_list=bond_list,
        positions=np.asarray(positions, dtype=float),
        cell_indices=cell_indices,
        sublattice_indices=sublattice_indices,
        length_x=length_x,
        length_y=length_y_value,
        circumference_x=bool(circumference_x),
        circumference_y=bool(circumference_y),
    )


def square_real_space_position(x_cell: int, y_cell: int) -> np.ndarray:
    return np.asarray([float(x_cell), float(y_cell)], dtype=float)


def build_square_cylinder_geometry(
    length_x: int,
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> GeometryData:
    length_x = _validate_lattice_length("length_x", length_x)
    length_y_value = _validate_length_y(length_y)
    bond_list: List[Bond] = []
    positions: List[np.ndarray] = []
    cell_indices: List[Tuple[int, int]] = []
    sublattice_indices: List[int] = []
    n_sites = length_x * length_y_value
    site_to_index: Dict[Tuple[int, int], int] = {}

    for x in range(length_x):
        for y in snake_y_values(x, length_y_value):
            site_to_index[(x, y)] = len(positions)
            positions.append(square_real_space_position(x, y))
            cell_indices.append((x, y))
            sublattice_indices.append(0)
    if len(positions) != n_sites:
        raise RuntimeError("Internal geometry error: square snake ordering generated wrong site count.")

    for x in range(length_x):
        for y in range(length_y_value):
            i_site = site_to_index[(x, y)]
            x_plus_1 = (x + 1) % length_x
            if circumference_x or (x + 1 < length_x):
                j_x = site_to_index[(x_plus_1, y)]
                if j_x != i_site:
                    bond_list.append(Bond(i_site, j_x, "x"))

            y_plus_1 = (y + 1) % length_y_value
            if circumference_y or (y + 1 < length_y_value):
                j_y = site_to_index[(x, y_plus_1)]
                if j_y != i_site:
                    bond_list.append(Bond(i_site, j_y, "y"))

    return GeometryData(
        number_of_sites=n_sites,
        bond_list=bond_list,
        positions=np.asarray(positions, dtype=float),
        cell_indices=cell_indices,
        sublattice_indices=sublattice_indices,
        length_x=length_x,
        length_y=length_y_value,
        circumference_x=bool(circumference_x),
        circumference_y=bool(circumference_y),
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
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> GeometryData:
    length_x = _validate_lattice_length("length_x", length_x)
    length_y_value = _validate_length_y(length_y)
    bond_list: List[Bond] = []
    positions: List[np.ndarray] = []
    cell_indices: List[Tuple[int, int]] = []
    sublattice_indices: List[int] = []
    n_sites = length_x * length_y_value
    site_to_index: Dict[Tuple[int, int], int] = {}

    for x in range(length_x):
        for y in snake_y_values(x, length_y_value):
            site_to_index[(x, y)] = len(positions)
            positions.append(triangular_real_space_position(x, y))
            cell_indices.append((x, y))
            sublattice_indices.append(0)
    if len(positions) != n_sites:
        raise RuntimeError("Internal geometry error: triangular snake ordering generated wrong site count.")

    for x in range(length_x):
        for y in range(length_y_value):
            i_site = site_to_index[(x, y)]

            x_plus_1 = (x + 1) % length_x
            if circumference_x or (x + 1 < length_x):
                j_x = site_to_index[(x_plus_1, y)]
                if j_x != i_site:
                    bond_list.append(Bond(i_site, j_x, "x"))

            y_plus_1 = (y + 1) % length_y_value
            if circumference_y or (y + 1 < length_y_value):
                j_y = site_to_index[(x, y_plus_1)]
                if j_y != i_site:
                    bond_list.append(Bond(i_site, j_y, "y"))

            if circumference_x or (x + 1 < length_x):
                y_minus_1 = (y - 1) % length_y_value
                if circumference_y or (y - 1 >= 0):
                    j_z = site_to_index[(x_plus_1, y_minus_1)]
                    if j_z != i_site:
                        bond_list.append(Bond(i_site, j_z, "z"))

    return GeometryData(
        number_of_sites=n_sites,
        bond_list=bond_list,
        positions=np.asarray(positions, dtype=float),
        cell_indices=cell_indices,
        sublattice_indices=sublattice_indices,
        length_x=length_x,
        length_y=length_y_value,
        circumference_x=bool(circumference_x),
        circumference_y=bool(circumference_y),
    )


def build_lattice_geometry(
    lattice: str,
    length_x: int,
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> GeometryData:
    lattice_name = lattice.lower()
    if lattice_name == "honeycomb":
        return build_honeycomb_cylinder_geometry(
            length_x,
            length_y=length_y,
            circumference_x=circumference_x,
            circumference_y=circumference_y,
        )
    if lattice_name == "square":
        return build_square_cylinder_geometry(
            length_x,
            length_y=length_y,
            circumference_x=circumference_x,
            circumference_y=circumference_y,
        )
    if lattice_name == "triangular":
        return build_triangular_cylinder_geometry(
            length_x,
            length_y=length_y,
            circumference_x=circumference_x,
            circumference_y=circumference_y,
        )
    raise ValueError(f"Unsupported lattice '{lattice}'. Choose from: honeycomb, square, triangular.")


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
            "boundary": {
                "circumference_x": bool(getattr(geometry, "circumference_x", False)),
                "circumference_y": bool(getattr(geometry, "circumference_y", True)),
            },
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
        "boundary": {
            "circumference_x": bool(getattr(geometry, "circumference_x", False)),
            "circumference_y": bool(getattr(geometry, "circumference_y", True)),
        },
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
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> str:
    length_y_value = _validate_length_y(length_y)
    lattice_short = {
        "honeycomb": "hc",
        "square": "sq",
        "triangular": "tri",
    }.get(lattice.lower(), lattice.lower())
    boundary = f"{'pbcX' if circumference_x else 'obcX'}_{'pbcY' if circumference_y else 'obcY'}"
    return _safe_filename_token(
        f"{lattice_short}_Lx{int(length_x)}_Ly{int(length_y_value)}_N{int(geometry.number_of_sites)}_{boundary}"
    )


def geometry_size_display_label(
    geometry: GeometryData,
    lattice: str,
    length_x: int,
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> str:
    length_y_value = _validate_length_y(length_y)
    boundary = f"{'PBC-x' if circumference_x else 'OBC-x'}, {'PBC-y' if circumference_y else 'OBC-y'}"
    return (
        f"{lattice_display_name(lattice)} Lx={int(length_x)}, "
        f"Ly={int(length_y_value)}, N={int(geometry.number_of_sites)}, {boundary}"
    )


def run_output_prefix(
    model_spec: ModelSpec,
    geometry: GeometryData,
    lattice: str,
    length_x: int,
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> str:
    return _safe_filename_token(
        f"{model_simplified_name(model_spec)}_"
        f"{geometry_size_filename_label(geometry, lattice, length_x, length_y=length_y, circumference_x=circumference_x, circumference_y=circumference_y)}"
    )


def run_title_label(
    model_spec: ModelSpec,
    geometry: GeometryData,
    lattice: str,
    length_x: int,
    *,
    length_y: int,
    circumference_x: bool = False,
    circumference_y: bool = True,
) -> str:
    return (
        f"{model_display_short_name(model_spec)} | "
        f"{geometry_size_display_label(geometry, lattice, length_x, length_y=length_y, circumference_x=circumference_x, circumference_y=circumference_y)}"
    )


def labeled_output_filename(run_prefix: str, base_filename: str) -> str:
    stem, extension = os.path.splitext(base_filename)
    return f"{_safe_filename_token(run_prefix)}_{_safe_filename_token(stem)}{extension}"


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
