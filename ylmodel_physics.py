#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from ylmodel_core_runtime import _end_stage, _make_progress_bar, _start_stage

AXIS_OPTIONS = ("x", "y", "z")
AXES = AXIS_OPTIONS
SPIN_REP_VALUES = {"1/2": 0.5, "3/2": 1.5}
# "1" is kept as a legacy alias and normalized to "0".
ORBITAL_REP_VALUES = {"0": 0.0, "1": 0.0, "1/2": 0.5}
SPIN_ONLY_MODEL_FAMILIES = ("heisenberg", "xy", "xxz", "xyz")
SYMMETRY_MODE_OPTIONS = ("none", "u1", "z2")
SPIN_REP_DEFAULT = "1/2"
ORBITAL_REP_DEFAULT = "1/2"
MODEL_FAMILY_DEFAULT = "yao_lee"
MODEL_FAMILY_OPTIONS = ("yao_lee", "ising_like") + tuple(SPIN_ONLY_MODEL_FAMILIES)
ISING_AXIS_DEFAULT = "z"
SPIN_REP = SPIN_REP_DEFAULT
ORBITAL_REP = ORBITAL_REP_DEFAULT
MODEL_FAMILY = MODEL_FAMILY_DEFAULT
ISING_AXIS = ISING_AXIS_DEFAULT


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
            net_charge += _operator_charge_transfer(site_ops[op_name], phys_charges, mode)
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
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    show_progress: bool = True,
) -> Tuple[float, np.ndarray]:
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
