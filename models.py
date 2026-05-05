#!/usr/bin/env python3
"""Shared physics layer for the Yao-Lee driver.

This module owns model specifications, local operators, external-field term
construction, lattice geometry, exact diagonalization, correlation
post-processing, and structure factors. Scan analysis belongs in
``analysis.py``; Tenax-specific MPO/DMRG code belongs in ``backend.py``; PNG
output code belongs in ``plot_outputs.py``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from analysis import _end_stage, _make_progress_bar, _start_stage, resolve_low_energy_spectrum

AXIS_OPTIONS = ("x", "y", "z")
AXES = AXIS_OPTIONS
SPIN_REP_VALUES = {"1/2": 0.5, "3/2": 1.5}
# "1" is kept as a legacy alias and normalized to "0".
ORBITAL_REP_VALUES = {"0": 0.0, "1": 0.0, "1/2": 0.5}
SPIN_ONLY_MODEL_FAMILIES = ("heisenberg", "xy", "xxz", "xyz")
SYMMETRY_MODE_OPTIONS = ("none", "u1", "z2")
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
                    "u1_charge_encoding": "q = 4096*(2*Sz) + (2*Tz)",
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
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
) -> List[Tuple[float, str]]:
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
    if mode == "none" or len(field_terms) == 0:
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
            terms.append((_real_scalar_if_close(coeff), plus_op, i, minus_op, j))
            terms.append((_real_scalar_if_close(coeff), minus_op, i, plus_op, j))
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
    if mode == "u1":
        return list(_u1_pair_terms_for_bond_terms(active_bond_terms, i, j))
    return [(coefficient, op_name, i, op_name, j) for coefficient, op_name in active_bond_terms]


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
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
) -> sparse.spmatrix:
    n_sites = geometry.number_of_sites
    op_cache = build_global_operator_cache_for_model(model_spec)
    ident = op_cache["Id"]
    local_dim = int(ident.shape[0])
    h_exact = sparse.csr_matrix((local_dim ** n_sites, local_dim ** n_sites), dtype=complex)

    bond_terms: List[Tuple[Any, List[Tuple[complex, str]]]] = []
    total_terms = 0
    field_terms = list(external_field_terms or [])
    for bond in geometry.bond_list:
        terms = list(
            model_terms_for_bond(
                bond.gamma.lower(),
                model_spec,
                alpha,
                beta,
                coupling_j,
                jx=jx,
                jy=jy,
                jz=jz,
            )
        )
        bond_terms.append((bond, terms))
        total_terms += len(terms)
    total_terms += n_sites * len(field_terms)

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

    for site in range(n_sites):
        for coeff, op_name in field_terms:
            op_list = [ident] * n_sites
            op_list[site] = op_cache[op_name]
            h_exact = h_exact + coeff * kron_all(op_list)
            if term_progress_bar is not None:
                term_progress_bar.update(1)

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
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
) -> Tuple[float, np.ndarray]:
    spectrum, eigenvectors = run_small_cluster_exact_spectrum(
        geometry=geometry,
        model_spec=model_spec,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        eigenstate_count=1,
        jx=jx,
        jy=jy,
        jz=jz,
        external_field_terms=external_field_terms,
        show_progress=show_progress,
    )
    return float(spectrum["ground_state_energy"]), eigenvectors[:, 0]


def run_small_cluster_exact_spectrum(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    eigenstate_count: int = 2,
    check_ground_state_degeneracy: bool = True,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
    ground_manifold_abs_tol: float = 1e-12,
    ground_manifold_rel_tol: float = 1e-12,
) -> Tuple[Dict[str, Any], np.ndarray]:
    stage_start = _start_stage("ED diagonalization", show_progress)
    hamiltonian = build_exact_hamiltonian(
        geometry,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
        external_field_terms=external_field_terms,
        show_progress=show_progress,
    )
    hilbert_dim = int(hamiltonian.shape[0])
    requested_count = max(1, int(eigenstate_count))
    solve_count = min(requested_count, hilbert_dim)
    if show_progress:
        print(
            f"[ed] eigensolve started: dim={hamiltonian.shape[0]}, nnz={hamiltonian.nnz}, k={solve_count}"
        )
    if solve_count >= hilbert_dim - 1:
        dense_hamiltonian = hamiltonian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
        eigenvalues = eigenvalues[:solve_count]
        eigenvectors = eigenvectors[:, :solve_count]
        solver_mode = "dense"
    else:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(hamiltonian, k=solve_count, which="SA")
        solver_mode = "sparse"
    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.asarray(np.real(eigenvalues[order]), dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    _end_stage("ED diagonalization", stage_start, show_progress)
    low_energy_resolution = resolve_low_energy_spectrum(
        eigenvalues,
        check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
        hilbert_dim=hilbert_dim,
        degeneracy_tolerance_abs=float(ground_manifold_abs_tol),
        degeneracy_tolerance_rel=float(ground_manifold_rel_tol),
    )
    e0 = float(low_energy_resolution["ground_state_energy"])
    spectrum: Dict[str, Any] = {
        "solver_mode": solver_mode,
        "hilbert_dim": hilbert_dim,
        "eigenstates_requested": requested_count,
        "eigenstates_returned": int(eigenvalues.size),
        "energies": [float(value) for value in eigenvalues],
        "ground_state_energy": e0,
        **low_energy_resolution,
    }
    return spectrum, eigenvectors


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


def one_point_expectation_from_state(
    state: np.ndarray,
    op_name: str,
    i: int,
    n_sites: int,
    op_cache: Dict[str, sparse.spmatrix],
    ident: sparse.spmatrix,
) -> complex:
    op_list = [ident] * n_sites
    op_list[i] = op_cache[op_name]
    op_i = kron_all(op_list)
    return complex(np.vdot(state, op_i.dot(state)))


def collect_uniform_z_observables_from_ed_state(
    geometry: GeometryData,
    state: np.ndarray,
    model_spec: ModelSpec,
) -> Dict[str, float]:
    n_sites = geometry.number_of_sites
    op_cache = build_global_operator_cache_for_model(model_spec)
    ident = op_cache["Id"]
    spin_z = 0.0j
    orbital_z = 0.0j
    for site in range(n_sites):
        spin_z += one_point_expectation_from_state(state, "Sz", site, n_sites, op_cache, ident)
        orbital_z += one_point_expectation_from_state(state, "Tz", site, n_sites, op_cache, ident)
    return {
        "spin_z_per_site": float(np.real(spin_z) / n_sites),
        "orbital_z_per_site": float(np.real(orbital_z) / n_sites),
    }


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
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
) -> float:
    e_bond = 0.0j
    for coeff, op_name in model_terms_for_bond(
        gamma,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
    ):
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
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
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
                    bond.i,
                    bond.j,
                    bond.gamma,
                    correlations,
                    model_spec,
                    alpha,
                    beta,
                    coupling_j,
                    jx=jx,
                    jy=jy,
                    jz=jz,
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


def finite_temperature_grid(
    temperature_min: float,
    temperature_max: float,
    temperature_points: int,
    temperature_scale: str = "log",
) -> np.ndarray:
    if float(temperature_min) <= 0.0:
        raise ValueError("temperature_min must be positive for canonical finite-temperature ED.")
    if float(temperature_max) < float(temperature_min):
        raise ValueError("temperature_max must be >= temperature_min.")
    points = int(temperature_points)
    if points < 2:
        raise ValueError("temperature_points must be at least 2.")
    scale = str(temperature_scale).strip().lower()
    if scale == "linear":
        return np.linspace(float(temperature_min), float(temperature_max), points)
    if scale == "log":
        return np.geomspace(float(temperature_min), float(temperature_max), points)
    raise ValueError("temperature_scale must be 'linear' or 'log'.")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.real(np.dot(weights, np.asarray(values, dtype=np.complex128))))


def run_finite_temperature_ed(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    *,
    lattice: str,
    temperature_min: float,
    temperature_max: float,
    temperature_points: int,
    temperature_scale: str = "log",
    max_eigenstates: int = 16,
    full_spectrum_max_dim: int = 512,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
    ground_manifold_abs_tol: float = 1e-12,
    ground_manifold_rel_tol: float = 1e-12,
) -> Dict[str, Any]:
    """Compute finite-temperature ED observables from a full or low-energy spectrum.

    Full spectra are exact but only feasible for tiny Hilbert spaces. Larger
    allowed ED clusters use the lowest ``max_eigenstates`` eigenpairs, which is
    useful for low-temperature trends and is recorded as truncated in metadata.
    """
    n_sites = int(geometry.number_of_sites)
    temperatures = finite_temperature_grid(
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        temperature_points=temperature_points,
        temperature_scale=temperature_scale,
    )

    stage_start = _start_stage("Finite-temperature ED", show_progress)
    hamiltonian = build_exact_hamiltonian(
        geometry,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
        external_field_terms=external_field_terms,
        show_progress=show_progress,
    )
    hilbert_dim = int(hamiltonian.shape[0])
    if hilbert_dim <= int(full_spectrum_max_dim):
        if show_progress:
            print(f"[thermal-ed] dense full diagonalization started: dim={hilbert_dim}")
        dense_hamiltonian = hamiltonian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
        spectrum_mode = "full"
        full_spectrum = True
    else:
        eigenstate_count = max(1, min(int(max_eigenstates), hilbert_dim - 2))
        if show_progress:
            print(
                "[thermal-ed] sparse low-energy eigensolve started: "
                f"dim={hilbert_dim}, nnz={hamiltonian.nnz}, k={eigenstate_count}"
            )
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            hamiltonian,
            k=eigenstate_count,
            which="SA",
        )
        spectrum_mode = "low_energy_truncated"
        full_spectrum = False

    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.asarray(np.real(eigenvalues[order]), dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    eigenstate_count = int(eigenvalues.size)

    state_rows: List[Dict[str, Any]] = []
    state_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=eigenstate_count,
        desc="Thermal ED states",
        unit="state",
        leave=False,
    )
    for state_index in range(eigenstate_count):
        state = eigenvectors[:, state_index]
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
            lattice=lattice,
            show_progress=False,
        )
        bond_rows = all_bond_energies(
            geometry,
            correlations,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx=jx,
            jy=jy,
            jz=jz,
            show_progress=False,
        )
        if len(geometry.bond_list) > 0:
            nn_s = float(
                np.mean([np.real(scalar_correlations["S"][bond.i, bond.j]) for bond in geometry.bond_list])
            )
            nn_t = float(
                np.mean([np.real(scalar_correlations["T"][bond.i, bond.j]) for bond in geometry.bond_list])
            )
            nn_st = float(
                np.mean([np.real(scalar_correlations["ST"][bond.i, bond.j]) for bond in geometry.bond_list])
            )
        else:
            nn_s = 0.0
            nn_t = 0.0
            nn_st = 0.0
        uniform_z = collect_uniform_z_observables_from_ed_state(geometry, state, model_spec)
        state_rows.append(
            {
                "state_index": int(state_index),
                "energy": float(eigenvalues[state_index]),
                "spin_z_per_site": float(uniform_z["spin_z_per_site"]),
                "orbital_z_per_site": float(uniform_z["orbital_z_per_site"]),
                "nearest_neighbor_S": nn_s,
                "nearest_neighbor_T": nn_t,
                "nearest_neighbor_ST": nn_st,
                "bond_energy_per_site": float(
                    np.sum([float(row["O_ij_gamma"]) for row in bond_rows]) / float(max(1, n_sites))
                ),
                "structure_factors": structure_rows,
            }
        )
        if state_progress_bar is not None:
            state_progress_bar.update(1)
    if state_progress_bar is not None:
        state_progress_bar.close()

    state_arrays = {
        key: np.asarray([float(row[key]) for row in state_rows], dtype=float)
        for key in (
            "spin_z_per_site",
            "orbital_z_per_site",
            "nearest_neighbor_S",
            "nearest_neighbor_T",
            "nearest_neighbor_ST",
            "bond_energy_per_site",
        )
    }
    sf_labels = [row["Q_label"] for row in state_rows[0]["structure_factors"]] if state_rows else []
    sf_channels = ("S(Q)", "T(Q)", "ST(Q)")

    observables: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    structure_rows_by_temperature: List[Dict[str, Any]] = []
    e0 = float(eigenvalues[0])
    shifted_energies = eigenvalues - e0
    for temperature in temperatures:
        t_value = float(temperature)
        weights_raw = np.exp(-shifted_energies / t_value)
        partition_shifted = float(np.sum(weights_raw))
        weights = weights_raw / partition_shifted
        energy = _weighted_mean(eigenvalues, weights)
        energy2 = _weighted_mean(eigenvalues * eigenvalues, weights)
        heat_capacity = max(0.0, (energy2 - energy * energy) / (t_value * t_value * n_sites))
        entropy = float(np.log(partition_shifted) + (energy - e0) / t_value)
        free_energy = float(e0 - t_value * np.log(partition_shifted))

        observables.append(
            {
                "T": t_value,
                "energy": energy,
                "energy_per_site": float(energy / n_sites),
                "free_energy_per_site": float(free_energy / n_sites),
                "entropy_per_site": float(entropy / n_sites),
                "specific_heat_per_site": float(heat_capacity),
                "spin_z_per_site": _weighted_mean(state_arrays["spin_z_per_site"], weights),
                "orbital_z_per_site": _weighted_mean(state_arrays["orbital_z_per_site"], weights),
                "partition_function_shifted": partition_shifted,
            }
        )
        correlation_rows.append(
            {
                "T": t_value,
                "nearest_neighbor_S": _weighted_mean(state_arrays["nearest_neighbor_S"], weights),
                "nearest_neighbor_T": _weighted_mean(state_arrays["nearest_neighbor_T"], weights),
                "nearest_neighbor_ST": _weighted_mean(state_arrays["nearest_neighbor_ST"], weights),
                "bond_energy_per_site": _weighted_mean(state_arrays["bond_energy_per_site"], weights),
            }
        )
        for q_index, q_label in enumerate(sf_labels):
            row: Dict[str, Any] = {"T": t_value, "Q_label": q_label}
            for channel in sf_channels:
                values = np.asarray(
                    [float(state_row["structure_factors"][q_index][channel]) for state_row in state_rows],
                    dtype=float,
                )
                row[channel] = _weighted_mean(values, weights)
            structure_rows_by_temperature.append(row)

    low_energy_resolution = resolve_low_energy_spectrum(
        eigenvalues,
        check_ground_state_degeneracy=True,
        hilbert_dim=hilbert_dim,
        degeneracy_tolerance_abs=float(ground_manifold_abs_tol),
        degeneracy_tolerance_rel=float(ground_manifold_rel_tol),
    )
    degeneracy_tolerance = float(low_energy_resolution["ground_state_degeneracy_tolerance"])
    ground_indices = np.asarray(
        low_energy_resolution.get("ground_state_indices", [0]),
        dtype=int,
    )
    if ground_indices.size == 0:
        ground_indices = np.asarray([0], dtype=int)
    first_excited_energy = low_energy_resolution.get("first_excited_energy")
    spectral_gap = low_energy_resolution.get("spectral_gap")
    gap_above_ground_manifold = low_energy_resolution.get("gap_above_ground_manifold")

    def _ground_average(key: str) -> float:
        return float(np.mean([float(state_rows[int(index)][key]) for index in ground_indices]))

    ground_observables = {
        "T": 0.0,
        "energy": e0,
        "energy_per_site": float(e0 / n_sites),
        "spin_z_per_site": _ground_average("spin_z_per_site"),
        "orbital_z_per_site": _ground_average("orbital_z_per_site"),
    }
    ground_correlations = {
        "T": 0.0,
        "nearest_neighbor_S": _ground_average("nearest_neighbor_S"),
        "nearest_neighbor_T": _ground_average("nearest_neighbor_T"),
        "nearest_neighbor_ST": _ground_average("nearest_neighbor_ST"),
        "bond_energy_per_site": _ground_average("bond_energy_per_site"),
    }
    ground_structure_rows: List[Dict[str, Any]] = []
    if state_rows:
        for q_index, q_label in enumerate(sf_labels):
            row: Dict[str, Any] = {"T": 0.0, "Q_label": q_label}
            for channel in sf_channels:
                row[channel] = float(
                    np.mean(
                        [
                            float(state_rows[int(index)]["structure_factors"][q_index][channel])
                            for index in ground_indices
                        ]
                    )
                )
            first_state_row = state_rows[0]["structure_factors"][q_index]
            if "Qx" in first_state_row:
                row["Qx"] = float(first_state_row["Qx"])
            if "Qy" in first_state_row:
                row["Qy"] = float(first_state_row["Qy"])
            ground_structure_rows.append(row)

    t_min = float(temperatures[0])
    weights_raw_min = np.exp(-shifted_energies / t_min)
    weights_min = weights_raw_min / float(np.sum(weights_raw_min))
    ground_weight_min = float(np.sum(weights_min[ground_indices]))
    first_observable_row = observables[0] if observables else {}
    first_correlation_row = correlation_rows[0] if correlation_rows else {}
    first_structure_rows = {
        str(row.get("Q_label")): row
        for row in structure_rows_by_temperature
        if abs(float(row.get("T", np.nan)) - t_min) <= 1e-14
    }

    observable_diffs: Dict[str, Dict[str, float]] = {}
    for key, target in ground_observables.items():
        if key == "T" or key not in first_observable_row:
            continue
        actual = float(first_observable_row[key])
        observable_diffs[key] = {
            "T_min": actual,
            "ground_limit": float(target),
            "abs_difference": float(abs(actual - float(target))),
        }
    correlation_diffs: Dict[str, Dict[str, float]] = {}
    for key, target in ground_correlations.items():
        if key == "T" or key not in first_correlation_row:
            continue
        actual = float(first_correlation_row[key])
        correlation_diffs[key] = {
            "T_min": actual,
            "ground_limit": float(target),
            "abs_difference": float(abs(actual - float(target))),
        }
    structure_diffs: List[Dict[str, Any]] = []
    for ground_row in ground_structure_rows:
        q_label = str(ground_row["Q_label"])
        t_row = first_structure_rows.get(q_label, {})
        for channel in sf_channels:
            if channel not in t_row:
                continue
            actual = float(t_row[channel])
            target = float(ground_row[channel])
            structure_diffs.append(
                {
                    "Q_label": q_label,
                    "channel": channel,
                    "T_min": actual,
                    "ground_limit": target,
                    "abs_difference": float(abs(actual - target)),
                }
            )
    max_observable_diff = max(
        [item["abs_difference"] for item in observable_diffs.values()] or [0.0]
    )
    max_correlation_diff = max(
        [item["abs_difference"] for item in correlation_diffs.values()] or [0.0]
    )
    max_structure_diff = max(
        [item["abs_difference"] for item in structure_diffs] or [0.0]
    )
    t_min_over_gap = (
        float(t_min / gap_above_ground_manifold)
        if gap_above_ground_manifold is not None and gap_above_ground_manifold > 0.0
        else None
    )
    ground_limit_abs_tolerance = 1e-5
    if (
        ground_weight_min >= 0.99
        and max(max_observable_diff, max_correlation_diff, max_structure_diff) <= ground_limit_abs_tolerance
    ):
        ground_limit_status = "passed"
    elif t_min_over_gap is not None and t_min_over_gap >= 0.1:
        ground_limit_status = "temperature_min_not_below_gap"
    else:
        ground_limit_status = "not_at_ground_limit"

    _end_stage("Finite-temperature ED", stage_start, show_progress)
    return {
        "spectrum": {
            "mode": spectrum_mode,
            "full_spectrum": bool(full_spectrum),
            "hilbert_dim": hilbert_dim,
            "eigenstates_used": eigenstate_count,
            "ground_state_energy": e0,
            "first_excited_energy": first_excited_energy,
            "spectral_gap": spectral_gap,
            "ground_manifold_degeneracy": int(
                low_energy_resolution.get("ground_state_degeneracy", ground_indices.size)
            ),
            "ground_manifold_tolerance": float(degeneracy_tolerance),
            "ground_manifold_absolute_tolerance": float(
                low_energy_resolution.get(
                    "ground_state_degeneracy_absolute_tolerance",
                    ground_manifold_abs_tol,
                )
            ),
            "ground_manifold_relative_tolerance": float(
                low_energy_resolution.get(
                    "ground_state_degeneracy_relative_tolerance",
                    ground_manifold_rel_tol,
                )
            ),
            "gap_above_ground_manifold": gap_above_ground_manifold,
            "low_energy_resolution": low_energy_resolution,
            "highest_included_energy": float(eigenvalues[-1]),
            "included_eigenvalues": [float(value) for value in eigenvalues],
            "truncated_spectrum_may_miss_degenerate_ground_states": bool(
                (not full_spectrum) and abs(float(eigenvalues[-1] - e0)) <= degeneracy_tolerance
            ),
            "note": (
                "Exact canonical trace over the full spectrum."
                if full_spectrum
                else (
                    "Low-energy truncated canonical trace. Treat high-temperature "
                    "values as qualitative unless max_eigenstates spans the relevant spectrum."
                )
            ),
        },
        "ground_state_limit_check": {
            "status": ground_limit_status,
            "temperature_min": t_min,
            "ground_manifold_boltzmann_weight_at_temperature_min": ground_weight_min,
            "temperature_min_over_gap_above_ground_manifold": t_min_over_gap,
            "max_observable_abs_difference": float(max_observable_diff),
            "max_correlation_abs_difference": float(max_correlation_diff),
            "max_structure_factor_abs_difference": float(max_structure_diff),
            "abs_difference_tolerance": float(ground_limit_abs_tolerance),
            "observable_differences": observable_diffs,
            "correlation_differences": correlation_diffs,
            "structure_factor_differences": structure_diffs,
            "structure_factor_definition": {
                "onsite_i_equals_j_terms_included": False,
                "note": (
                    "ED and DMRG structure factors in this driver are built from correlation "
                    "matrices whose diagonal entries are left at zero, so T-independent onsite "
                    "terms are not the source of flat curves here."
                ),
            },
        },
        "zero_temperature_references": {
            "ED-GS": {
                "method": "ED-GS",
                "temperature": 0.0,
                "note": (
                    "Canonical T->0 ED reference. If the ground state is degenerate, this is "
                    "the equal-weight average over the resolved ground manifold."
                ),
                "observables": ground_observables,
                "correlations": ground_correlations,
                "structure_factors": ground_structure_rows,
            }
        },
        "temperature_grid": {
            "min": float(temperature_min),
            "max": float(temperature_max),
            "points": int(temperature_points),
            "scale": str(temperature_scale).strip().lower(),
            "values": [float(value) for value in temperatures],
        },
        "observables": observables,
        "correlations": correlation_rows,
        "structure_factors": structure_rows_by_temperature,
        "state_observables": state_rows,
    }


# ----------------------------------------------------------------------
