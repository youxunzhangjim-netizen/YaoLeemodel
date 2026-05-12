#!/usr/bin/env python3
"""QuSpin exact-diagonalization backend for the spin-orbital Yao-Lee model.

This module mirrors the small-cluster ED entry points in ``ed_backend.py`` for
the spin-1/2, orbital-1/2 Yao-Lee Hilbert space.  The local physical dimension
is d=4, represented as a tensor product of a spin chain and an orbital chain.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse
from quspin.basis import spin_basis_1d, spin_basis_general, tensor_basis
from quspin.operators import hamiltonian

try:
    from models import (
        honeycomb_plaquette_flux_operators,
        plaquette_flux_close_to_target,
        select_honeycomb_plaquette_flux_operator,
    )
except Exception:  # pragma: no cover
    from .models import (  # type: ignore
        honeycomb_plaquette_flux_operators,
        plaquette_flux_close_to_target,
        select_honeycomb_plaquette_flux_operator,
    )


def _make_quspin_progress_bar(enabled: bool, total: int, desc: str, unit: str) -> Any | None:
    if not bool(enabled):
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return None
    return tqdm(total=int(total), desc=desc, unit=unit, dynamic_ncols=True, leave=False)


def _nup_from_total_m2(n_sites: int, total_m2: int, label: str) -> int:
    nup_numerator = int(n_sites) + int(total_m2)
    if nup_numerator % 2 != 0:
        raise ValueError(f"{label} target 2*total_z={total_m2} is unreachable for {n_sites} sites.")
    nup = nup_numerator // 2
    if nup < 0 or nup > int(n_sites):
        raise ValueError(f"{label} target 2*total_z={total_m2} is outside the reachable range.")
    return int(nup)


def valid_total_m2_sectors(n_sites: int) -> List[int]:
    """Return all total ``2*Sz`` sectors for ``n_sites`` spin-1/2 sites."""
    n = int(n_sites)
    return [int(2 * nup - n) for nup in range(n + 1)]


def _field_terms(external_field_terms: List[Tuple[float, str]] | None) -> List[Tuple[float, str]]:
    return [
        (float(coefficient), str(op_name))
        for coefficient, op_name in list(external_field_terms or [])
        if abs(float(coefficient)) > 1e-14
    ]


def _has_sz_zeeman_terms(external_field_terms: List[Tuple[float, str]] | None) -> bool:
    return any(op_name == "Sz" for _coefficient, op_name in _field_terms(external_field_terms))


def _has_transverse_spin_field_terms(external_field_terms: List[Tuple[float, str]] | None) -> bool:
    return any(op_name in ("Sx", "Sy") for _coefficient, op_name in _field_terms(external_field_terms))


def _spin_field_breaks_z2(external_field_terms: List[Tuple[float, str]] | None) -> bool:
    return any(op_name in ("Sx", "Sy", "Sz") for _coefficient, op_name in _field_terms(external_field_terms))


def _cell_and_sublattice_lookup(geometry: Any) -> Tuple[Dict[Tuple[int, int, int], int], List[int], List[int]]:
    cell_indices = list(getattr(geometry, "cell_indices", []))
    sublattice_indices = list(getattr(geometry, "sublattice_indices", []))
    n_sites = int(getattr(geometry, "number_of_sites"))
    if len(cell_indices) != n_sites or len(sublattice_indices) != n_sites:
        raise ValueError("2D translation blocks require geometry.cell_indices and geometry.sublattice_indices.")
    x_values = sorted({int(cell[0]) for cell in cell_indices})
    y_values = sorted({int(cell[1]) for cell in cell_indices})
    lookup: Dict[Tuple[int, int, int], int] = {}
    for site, (cell, sublattice) in enumerate(zip(cell_indices, sublattice_indices)):
        key = (int(cell[0]), int(cell[1]), int(sublattice))
        if key in lookup:
            raise ValueError(f"Duplicate honeycomb site label {key}; cannot build translations.")
        lookup[key] = int(site)
    return lookup, x_values, y_values


def build_honeycomb_torus_translation_permutations(geometry: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return T1/T2 site permutations preserving honeycomb bond directions.

    The maps translate the unit-cell labels ``(x, y, sublattice)`` by one cell
    along x/y.  They are valid symmetry reductions only when the Hamiltonian
    is a periodic two-dimensional torus; the validation below rejects open-x
    cylinders because T1 would move an x-boundary bond to a missing wrap bond.
    """
    n_sites = int(getattr(geometry, "number_of_sites"))
    lookup, x_values, y_values = _cell_and_sublattice_lookup(geometry)
    x_index = {value: offset for offset, value in enumerate(x_values)}
    y_index = {value: offset for offset, value in enumerate(y_values)}
    t1 = np.empty(n_sites, dtype=np.int32)
    t2 = np.empty(n_sites, dtype=np.int32)
    for (x_cell, y_cell, sublattice), site in lookup.items():
        next_x = x_values[(x_index[x_cell] + 1) % len(x_values)]
        next_y = y_values[(y_index[y_cell] + 1) % len(y_values)]
        try:
            t1[site] = lookup[(next_x, y_cell, sublattice)]
            t2[site] = lookup[(x_cell, next_y, sublattice)]
        except KeyError as exc:
            raise ValueError("Honeycomb geometry is not a complete rectangular cell torus.") from exc
    return t1, t2


def _translation_preserves_bond_directions(geometry: Any, permutation: np.ndarray) -> bool:
    bond_set = {
        (min(int(i), int(j)), max(int(i), int(j)), str(gamma))
        for i, j, gamma in _bond_triplets(geometry)
    }
    for i, j, gamma in _bond_triplets(geometry):
        mapped_i = int(permutation[int(i)])
        mapped_j = int(permutation[int(j)])
        if (min(mapped_i, mapped_j), max(mapped_i, mapped_j), str(gamma)) not in bond_set:
            return False
    return True


def _validated_translation_blocks(
    geometry: Any,
    use_translation_x_block: bool = True,
    use_translation_y_block: bool = True,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if bool(use_translation_x_block) and hasattr(geometry, "circumference_x") and not bool(geometry.circumference_x):
        raise ValueError(
            "QuSpin x-translation block T1 is forbidden because geometry.circumference_x is false. "
            "Use --circumference-x to close the x direction."
        )
    if bool(use_translation_y_block) and hasattr(geometry, "circumference_y") and not bool(geometry.circumference_y):
        raise ValueError(
            "QuSpin y-translation block T2 is forbidden because geometry.circumference_y is false. "
            "Use --circumference-y to close the y direction."
        )
    t1, t2 = build_honeycomb_torus_translation_permutations(geometry)
    if bool(use_translation_x_block) and not _translation_preserves_bond_directions(geometry, t1):
        raise ValueError(
            "QuSpin x-translation block T1 is forbidden for this geometry: "
            "the bond list is not periodic along x or T1 changes the bond-direction set."
        )
    if bool(use_translation_y_block) and not _translation_preserves_bond_directions(geometry, t2):
        raise ValueError(
            "QuSpin y-translation block T2 is forbidden for this geometry: "
            "the bond list is not periodic along y or T2 changes the bond-direction set."
        )
    return (
        t1 if bool(use_translation_x_block) else None,
        t2 if bool(use_translation_y_block) else None,
    )


def quspin_translation_block_support(geometry: Any) -> Dict[str, Dict[str, Any]]:
    """Check x/y honeycomb translation blocks independently."""
    support: Dict[str, Dict[str, Any]] = {}
    for axis, use_x, use_y in (("x", True, False), ("y", False, True)):
        try:
            _validated_translation_blocks(
                geometry,
                use_translation_x_block=use_x,
                use_translation_y_block=use_y,
            )
        except Exception as exc:
            support[axis] = {"supported": False, "reason": str(exc)}
        else:
            support[axis] = {"supported": True, "reason": None}
    return support


def quspin_translation_blocks_supported(
    geometry: Any,
    use_translation_x_block: bool = True,
    use_translation_y_block: bool = True,
) -> Tuple[bool, str | None]:
    """Check whether all requested translation blocks are valid."""
    try:
        _validated_translation_blocks(
            geometry,
            use_translation_x_block=use_translation_x_block,
            use_translation_y_block=use_translation_y_block,
        )
    except Exception as exc:
        return False, str(exc)
    return True, None


def _spin_flip_permutation(n_sites: int) -> np.ndarray:
    # Negative entries request spin inversion in QuSpin's general-basis maps.
    return -(np.arange(int(n_sites), dtype=np.int32) + 1)


def _general_spin_basis_with_fallbacks(n_sites: int, **kwargs: Any) -> Any:
    """Build spin_basis_general while keeping compatibility with QuSpin variants."""
    attempts: List[Dict[str, Any]] = [dict(kwargs)]
    literal_kwargs = dict(kwargs)
    literal_block_dict: Dict[str, np.ndarray] = {}
    if isinstance(literal_kwargs.get("kblock_1"), tuple):
        t1, q1 = literal_kwargs.pop("kblock_1")
        literal_kwargs["kblock_1"] = int(q1)
        literal_block_dict["T1"] = np.asarray(t1, dtype=np.int32)
    if isinstance(literal_kwargs.get("kblock_2"), tuple):
        t2, q2 = literal_kwargs.pop("kblock_2")
        literal_kwargs["kblock_2"] = int(q2)
        literal_block_dict["T2"] = np.asarray(t2, dtype=np.int32)
    if isinstance(literal_kwargs.get("pblock"), tuple):
        spin_flip, parity = literal_kwargs.pop("pblock")
        literal_kwargs["pblock"] = 1 if int(parity) == 0 else -1
        literal_block_dict["P"] = np.asarray(spin_flip, dtype=np.int32)
    if literal_block_dict:
        literal_kwargs["block_dict"] = literal_block_dict
        # Some local QuSpin variants expose the block_dict interface described
        # in the driver comments: kblock_1/kblock_2 are quantum numbers, while
        # block_dict carries the T1/T2 permutations.
        attempts.append(literal_kwargs)
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return spin_basis_general(int(n_sites), **attempt)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return spin_basis_general(int(n_sites), **kwargs)


def build_quspin_yao_lee_basis(
    n_sites: int,
    geometry: Any | None = None,
    use_sz_block: bool = True,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_target_parity: int = 0,
    use_translation_block: bool = False,
    use_translation_x_block: bool | None = None,
    use_translation_y_block: bool | None = None,
    momentum_block_1: int = 0,
    momentum_block_2: int = 0,
    momentum_x_block: int | None = None,
    momentum_y_block: int | None = None,
    use_reflection_block: bool = False,
    reflection_block: int = 0,
) -> Any:
    """Build the QuSpin tensor basis using the shared symmetry-reduction settings."""
    n_sites = int(n_sites)
    if n_sites <= 0:
        raise ValueError("n_sites must be positive.")
    if bool(use_reflection_block) or int(reflection_block) != 0:
        raise ValueError(
            "QuSpin reflection/C3 spatial blocks are forbidden for the bond-directional Yao-Lee "
            "Hamiltonian unless a gauge transformation that permutes x/y/z bonds is implemented."
        )
    if bool(use_z2_block) and (not bool(use_sz_block) or int(target_sz2) != 0):
        raise ValueError("QuSpin spin-flip Z2 parity is allowed only inside the total Sz=0 sector.")
    translation_x_requested = (
        bool(use_translation_block)
        if use_translation_x_block is None
        else bool(use_translation_x_block)
    )
    translation_y_requested = (
        bool(use_translation_block)
        if use_translation_y_block is None
        else bool(use_translation_y_block)
    )
    any_translation_requested = bool(translation_x_requested or translation_y_requested)
    kx = int(momentum_block_1 if momentum_x_block is None else momentum_x_block)
    ky = int(momentum_block_2 if momentum_y_block is None else momentum_y_block)
    if bool(use_tau_z_block) and (bool(use_z2_block) or any_translation_requested):
        raise ValueError(
            "QuSpin does not combine tau_z with the optimized Sz/Z2/translation reduction path. "
            "Use tau_z alone only when the Hamiltonian really conserves it."
        )

    # pauli=False makes x, y, z represent spin-1/2 operators S and tau rather
    # than Pauli matrices sigma.
    spin_kwargs: Dict[str, Any] = {"pauli": False}
    orbital_kwargs: Dict[str, Any] = {"pauli": False}
    if bool(use_sz_block):
        spin_kwargs["Nup"] = _nup_from_total_m2(n_sites, int(target_sz2), "spin")
    if bool(use_tau_z_block):
        orbital_kwargs["Nup"] = _nup_from_total_m2(n_sites, int(target_tz2), "orbital")

    if any_translation_requested:
        if geometry is None:
            raise ValueError("QuSpin translation blocks require the full geometry object.")
        t1_perm, t2_perm = _validated_translation_blocks(
            geometry,
            use_translation_x_block=translation_x_requested,
            use_translation_y_block=translation_y_requested,
        )
        if translation_x_requested:
            spin_kwargs["kblock_1"] = (t1_perm, kx)
            orbital_kwargs["kblock_1"] = (t1_perm, kx)
        if translation_y_requested:
            spin_kwargs["kblock_2"] = (t2_perm, ky)
            orbital_kwargs["kblock_2"] = (t2_perm, ky)
    if bool(use_z2_block):
        # Spin-flip is a spin-sector symmetry only.  Even parity is the default
        # target and corresponds to the user-facing z2_target_parity=0.
        spin_kwargs["pblock"] = (_spin_flip_permutation(n_sites), int(z2_target_parity) % 2)

    if any_translation_requested or bool(use_z2_block):
        basis_spin = _general_spin_basis_with_fallbacks(n_sites, **spin_kwargs)
        basis_orbital = _general_spin_basis_with_fallbacks(n_sites, **orbital_kwargs)
    else:
        basis_spin = spin_basis_1d(
            L=n_sites,
            **spin_kwargs,
        )
        basis_orbital = spin_basis_1d(
            L=n_sites,
            **orbital_kwargs,
        )
    return tensor_basis(basis_spin, basis_orbital)


def _bond_triplets(geometry: Any) -> List[Tuple[int, int, str]]:
    """Extract ``(i, j, gamma)`` bonds from the project's geometry object."""
    triplets: List[Tuple[int, int, str]] = []
    for bond in getattr(geometry, "bond_list", []):
        if hasattr(bond, "i") and hasattr(bond, "j") and hasattr(bond, "gamma"):
            i, j, gamma = int(bond.i), int(bond.j), str(bond.gamma).lower()
        elif hasattr(bond, "site_i") and hasattr(bond, "site_j") and hasattr(bond, "bond_type"):
            i, j, gamma = int(bond.site_i), int(bond.site_j), str(bond.bond_type).lower()
        else:
            raise AttributeError("Bond object must provide i, j, gamma fields.")
        if gamma not in ("x", "y", "z"):
            raise ValueError(f"Unsupported Yao-Lee bond direction '{gamma}'.")
        triplets.append((i, j, gamma))
    return triplets


def _append_term(
    static: List[List[Any]],
    op_string: str,
    coupling_list: List[float],
) -> None:
    """Append one QuSpin static term if its coefficient is nonzero."""
    coefficient = float(coupling_list[0])
    if coefficient != 0.0:
        static.append([op_string, [coupling_list]])


def build_quspin_yao_lee_static_terms(
    geometry: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    external_field_terms: List[Tuple[float, str]] | None = None,
) -> List[List[Any]]:
    """Build the QuSpin ``static`` list for the Yao-Lee Hamiltonian.

    For a tensor basis, strings have the form ``"op_spin|op_orbital"``.
    The identity side still receives a site index in the coupling list because
    QuSpin counts the ``I`` character as a local operator in the tensor string.
    """
    static: List[List[Any]] = []
    spin_coefficient = float(coupling_j) * (1.0 + float(beta))
    orbital_coefficient = float(coupling_j) * (1.0 - float(beta))
    mixed_coefficient = float(coupling_j) * float(alpha)
    orbital_pair_for_gamma = {"x": "xx", "y": "yy", "z": "zz"}

    for i, j, gamma in _bond_triplets(geometry):
        orbital_pair = orbital_pair_for_gamma[gamma]

        # Spin Heisenberg term: (1+beta) S_i dot S_j.
        for spin_pair in ("xx", "yy", "zz"):
            _append_term(static, f"{spin_pair}|I", [spin_coefficient, i, j, i])

        # Orbital compass/Ising term: (1-beta) tau_i^gamma tau_j^gamma.
        _append_term(static, f"I|{orbital_pair}", [orbital_coefficient, i, i, j])

        # Mixed term: alpha (S_i dot S_j)(tau_i^gamma tau_j^gamma).
        for spin_pair in ("xx", "yy", "zz"):
            _append_term(static, f"{spin_pair}|{orbital_pair}", [mixed_coefficient, i, j, i, j])

    spin_field_ops = {
        "Sx": "x|I",
        "Sy": "y|I",
        "Sz": "z|I",
    }
    for coefficient, op_name in _field_terms(external_field_terms):
        if op_name not in spin_field_ops:
            raise ValueError(f"Unsupported QuSpin external field operator '{op_name}'.")
        for site in range(int(getattr(geometry, "number_of_sites"))):
            _append_term(static, spin_field_ops[op_name], [coefficient, int(site), int(site)])

    return static


def build_quspin_yao_lee_hamiltonian(
    geometry: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    use_sz_block: bool = True,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_target_parity: int = 0,
    use_translation_block: bool = False,
    use_translation_x_block: bool | None = None,
    use_translation_y_block: bool | None = None,
    momentum_block_1: int = 0,
    momentum_block_2: int = 0,
    momentum_x_block: int | None = None,
    momentum_y_block: int | None = None,
    use_reflection_block: bool = False,
    reflection_block: int = 0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    check_symm: bool = False,
    check_herm: bool = False,
    check_pcon: bool = False,
) -> Tuple[Any, Any, List[List[Any]]]:
    """Construct the QuSpin Yao-Lee Hamiltonian and return ``(H, basis, static)``."""
    n_sites = int(getattr(geometry, "number_of_sites"))
    basis = build_quspin_yao_lee_basis(
        n_sites,
        geometry=geometry,
        use_sz_block=use_sz_block,
        target_sz2=target_sz2,
        use_tau_z_block=use_tau_z_block,
        target_tz2=target_tz2,
        use_z2_block=use_z2_block,
        z2_target_parity=z2_target_parity,
        use_translation_block=use_translation_block,
        use_translation_x_block=use_translation_x_block,
        use_translation_y_block=use_translation_y_block,
        momentum_block_1=momentum_block_1,
        momentum_block_2=momentum_block_2,
        momentum_x_block=momentum_x_block,
        momentum_y_block=momentum_y_block,
        use_reflection_block=use_reflection_block,
        reflection_block=reflection_block,
    )
    static = build_quspin_yao_lee_static_terms(
        geometry=geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        external_field_terms=external_field_terms,
    )
    hamiltonian_operator = hamiltonian(
        static,
        [],
        basis=basis,
        dtype=np.complex128,
        # QuSpin does not implement check_symm for tensor_basis; forcing this
        # off avoids a noisy warning while keeping hermiticity/pcon checks.
        check_symm=False,
        check_herm=bool(check_herm),
        check_pcon=bool(check_pcon),
    )
    return hamiltonian_operator, basis, static


def quspin_hamiltonian_as_sparse_matrix(hamiltonian_operator: Any) -> sparse.spmatrix:
    """Return a SciPy sparse matrix view of a QuSpin Hamiltonian."""
    matrix = hamiltonian_operator.tocsr()
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    return matrix


def _solve_lowest_quspin_eigenpairs(
    hamiltonian_operator: Any,
    basis: Any,
    eigenstate_count: int,
    *,
    show_progress: bool,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the lowest QuSpin eigenpairs, including the one-state basis edge case."""
    dimension = int(basis.Ns)
    if dimension <= 0:
        raise ValueError("Cannot diagonalize an empty QuSpin basis.")
    if dimension == 1:
        matrix = quspin_hamiltonian_as_sparse_matrix(hamiltonian_operator)
        return (
            np.asarray([float(np.real(matrix[0, 0]))], dtype=float),
            np.ones((1, 1), dtype=np.complex128),
        )
    requested_count = max(1, int(eigenstate_count))
    if dimension <= 2 or requested_count >= dimension - 1:
        matrix = quspin_hamiltonian_as_sparse_matrix(hamiltonian_operator).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        count = min(requested_count, dimension)
        return (
            np.asarray(np.real(eigenvalues[:count]), dtype=float),
            np.asarray(eigenvectors[:, :count], dtype=np.complex128),
        )
    k = max(1, min(requested_count, dimension - 2))
    if show_progress:
        hamiltonian_matrix = quspin_hamiltonian_as_sparse_matrix(hamiltonian_operator)
        print(
            f"[quspin-ed] {label} eigsh started: "
            f"dim={hamiltonian_matrix.shape[0]}, nnz={hamiltonian_matrix.nnz}, k={k}"
        )
    start = time.perf_counter()
    eigenvalues, eigenvectors = hamiltonian_operator.eigsh(k=k, which="SA")
    if show_progress:
        print(f"[quspin-ed] {label} eigsh finished in {time.perf_counter() - start:.2f}s")
    order = np.argsort(np.real(eigenvalues))
    return (
        np.asarray(np.real(eigenvalues[order]), dtype=float),
        np.asarray(eigenvectors[:, order], dtype=np.complex128),
    )


def _basis_operator_matrix(
    basis: Any,
    op_string: str,
    indices: List[int],
) -> sparse.spmatrix:
    """Build a sparse operator matrix from ``basis.Op``."""
    matrix_elements, rows, cols = basis.Op(op_string, indices, 1.0, np.complex128)
    return sparse.csr_matrix(
        (matrix_elements, (rows, cols)),
        shape=(int(basis.Ns), int(basis.Ns)),
        dtype=np.complex128,
    )


def _expectation_value_from_basis_op(
    basis: Any,
    evec: np.ndarray,
    op_string: str,
    indices: List[int],
) -> complex:
    """Compute ``<evec|O|evec>`` for a QuSpin ``basis.Op`` operator."""
    state = np.asarray(evec, dtype=np.complex128).reshape(-1)
    operator = _basis_operator_matrix(basis, op_string, indices)
    return complex(np.vdot(state, operator.dot(state)))


def _spin_pair_op(axis: str, i: int, j: int) -> Tuple[str, List[int]]:
    axis = str(axis)
    return f"{axis}{axis}|I", [int(i), int(j), int(i)]


def _orbital_pair_op(axis: str, i: int, j: int) -> Tuple[str, List[int]]:
    axis = str(axis)
    return f"I|{axis}{axis}", [int(i), int(i), int(j)]


def _mixed_pair_op(spin_axis: str, orbital_axis: str, i: int, j: int) -> Tuple[str, List[int]]:
    spin_axis = str(spin_axis)
    orbital_axis = str(orbital_axis)
    return (
        f"{spin_axis}{spin_axis}|{orbital_axis}{orbital_axis}",
        [int(i), int(j), int(i), int(j)],
    )


def build_spin_orbital_scalar_correlations(
    basis: Any,
    evec: np.ndarray,
    n_sites: int,
) -> Dict[str, np.ndarray]:
    """Return scalar spin/orbital/mixed correlations from a QuSpin ground state.

    The returned dictionary contains both the ED-style short keys ``S``, ``T``,
    ``ST`` and the TeNPy-style aliases ``spin_scalar``, ``orbital_scalar``,
    ``mixed_scalar``.
    """
    n_sites = int(n_sites)
    state = np.asarray(evec, dtype=np.complex128).reshape(-1)
    if int(state.size) != int(basis.Ns):
        raise ValueError(f"evec length {state.size} does not match basis dimension {basis.Ns}.")

    spin_scalar = np.zeros((n_sites, n_sites), dtype=np.complex128)
    orbital_scalar = np.zeros((n_sites, n_sites), dtype=np.complex128)
    mixed_scalar = np.zeros((n_sites, n_sites), dtype=np.complex128)

    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                # For spin-1/2 operators with pauli=False:
                # S_a^2 = tau_a^2 = 1/4, so sum_a S_a^2 = 3/4 and
                # sum_{a,b} (S_a tau_b)^2 = 9/16.
                spin_scalar[i, j] = 0.75
                orbital_scalar[i, j] = 0.75
                mixed_scalar[i, j] = 9.0 / 16.0
                continue

            spin_value = 0.0j
            orbital_value = 0.0j
            mixed_value = 0.0j
            for axis in ("x", "y", "z"):
                op_string, indices = _spin_pair_op(axis, i, j)
                spin_value += _expectation_value_from_basis_op(basis, state, op_string, indices)

                op_string, indices = _orbital_pair_op(axis, i, j)
                orbital_value += _expectation_value_from_basis_op(basis, state, op_string, indices)

            for spin_axis in ("x", "y", "z"):
                for orbital_axis in ("x", "y", "z"):
                    op_string, indices = _mixed_pair_op(spin_axis, orbital_axis, i, j)
                    mixed_value += _expectation_value_from_basis_op(basis, state, op_string, indices)

            spin_scalar[i, j] = spin_value
            orbital_scalar[i, j] = orbital_value
            mixed_scalar[i, j] = mixed_value

    return {
        "S": spin_scalar,
        "T": orbital_scalar,
        "ST": mixed_scalar,
        "spin_scalar": spin_scalar,
        "orbital_scalar": orbital_scalar,
        "mixed_scalar": mixed_scalar,
    }


def all_bond_energies(
    geometry: Any,
    correlations: Dict[str, np.ndarray],
    alpha: float,
    beta: float,
    coupling_j: float,
) -> List[Dict[str, Any]]:
    """Format scalar correlations as bond-energy rows.

    This lightweight formatter matches the row shape used by the TeNPy backend.
    It uses scalar ``S/T/ST`` correlations as a placeholder until the QuSpin
    backend exposes gamma-resolved bond-energy channels.
    """
    spin_matrix = correlations.get("S", correlations.get("spin_scalar"))
    orbital_matrix = correlations.get("T", correlations.get("orbital_scalar"))
    mixed_matrix = correlations.get("ST", correlations.get("mixed_scalar"))
    if spin_matrix is None or orbital_matrix is None or mixed_matrix is None:
        raise KeyError("correlations must contain S/T/ST or spin_scalar/orbital_scalar/mixed_scalar.")

    spin_coefficient = float(coupling_j) * (1.0 + float(beta))
    orbital_coefficient = float(coupling_j) * (1.0 - float(beta))
    mixed_coefficient = float(coupling_j) * float(alpha)
    rows: List[Dict[str, Any]] = []
    for i, j, gamma in _bond_triplets(geometry):
        spin_corr = complex(spin_matrix[i, j])
        orbital_corr = complex(orbital_matrix[i, j])
        mixed_corr = complex(mixed_matrix[i, j])
        components = [
            {
                "channel": "S",
                "operator": "Sdot",
                "axis": "dot",
                "coefficient": spin_coefficient,
                "correlation": float(np.real(spin_corr)),
                "energy": float(np.real(spin_coefficient * spin_corr)),
            },
            {
                "channel": "T",
                "operator": f"T{gamma}",
                "axis": str(gamma),
                "coefficient": orbital_coefficient,
                "correlation": float(np.real(orbital_corr)),
                "energy": float(np.real(orbital_coefficient * orbital_corr)),
            },
            {
                "channel": "ST",
                "operator": f"SdotT{gamma}",
                "axis": str(gamma),
                "coefficient": mixed_coefficient,
                "correlation": float(np.real(mixed_corr)),
                "energy": float(np.real(mixed_coefficient * mixed_corr)),
            },
        ]
        channel_energies = {
            str(component["channel"]): float(component["energy"])
            for component in components
        }
        rows.append(
            {
                "i": int(i),
                "j": int(j),
                "gamma": str(gamma),
                "O_ij_gamma": float(sum(float(component["energy"]) for component in components)),
                "components": components,
                "channel_energies": channel_energies,
            }
        )
    return rows


def _gamma_structure_factor(matrix: np.ndarray) -> float:
    size = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if size <= 0:
        return 0.0
    return float(np.real(np.sum(matrix)) / float(size))


def all_high_symmetry_structure_factors(
    scalar_correlations: Dict[str, np.ndarray],
    geometry: Any,
) -> List[Dict[str, Any]]:
    """Return a minimal high-symmetry structure-factor row list.

    The output keys match ``tenpy_backend.all_high_symmetry_structure_factors``.
    """
    del geometry
    spin_matrix = scalar_correlations.get("S", scalar_correlations.get("spin_scalar"))
    orbital_matrix = scalar_correlations.get("T", scalar_correlations.get("orbital_scalar"))
    mixed_matrix = scalar_correlations.get("ST", scalar_correlations.get("mixed_scalar"))
    if spin_matrix is None or orbital_matrix is None or mixed_matrix is None:
        raise KeyError("scalar_correlations must contain S/T/ST or spin_scalar/orbital_scalar/mixed_scalar.")
    return [
        {
            "Q_label": "Gamma",
            "Qx": 0.0,
            "Qy": 0.0,
            "S(Q)": _gamma_structure_factor(np.asarray(spin_matrix)),
            "T(Q)": _gamma_structure_factor(np.asarray(orbital_matrix)),
            "ST(Q)": _gamma_structure_factor(np.asarray(mixed_matrix)),
        }
    ]


def compute_plaquette_flux(
    basis: Any,
    evec: np.ndarray,
    geometry: Any,
    plaquette_center_idx: int | None = None,
) -> Dict[str, Any]:
    """Evaluate normalized honeycomb plaquette flux on every valid hexagon."""
    state = np.asarray(evec, dtype=np.complex128).reshape(-1)
    plaquettes = honeycomb_plaquette_flux_operators(geometry)
    if len(plaquettes) == 0:
        raise ValueError("No honeycomb length-six plaquette was found in this geometry.")
    selected = select_honeycomb_plaquette_flux_operator(geometry, plaquette_center_idx)
    selected_index = int(selected["plaquette_index"])
    flux_map: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}
    for plaquette in plaquettes:
        op_string = "I|" + "".join(str(axis) for axis in plaquette["axes"])
        indices = [int(plaquette["sites"][0])] + [int(site) for site in plaquette["sites"]]
        raw_value = _expectation_value_from_basis_op(basis, state, op_string, indices)
        normalized_value = float(np.real(raw_value) * float(plaquette["normalization"]))
        plaquette_index = int(plaquette["plaquette_index"])
        flux_map[plaquette_index] = normalized_value
        details[plaquette_index] = {
            "plaquette_index": plaquette_index,
            "sites": [int(site) for site in plaquette["sites"]],
            "axes": [str(axis) for axis in plaquette["axes"]],
            "operators": [f"I|{axis}" for axis in plaquette["axes"]],
            "raw_tau_product": float(np.real(raw_value)),
            "W_p": normalized_value,
            "value": normalized_value,
            "normalization": float(plaquette["normalization"]),
            "target": 1.0,
            "close_to_target": plaquette_flux_close_to_target(normalized_value),
        }
    selected_detail = details[selected_index]
    values = np.asarray(list(flux_map.values()), dtype=float)
    return {
        "available": True,
        **selected_detail,
        "all_plaquette_fluxes": flux_map,
        "plaquette_flux_map": flux_map,
        "plaquettes": details,
        "mean_W_p": float(np.mean(values)),
        "min_W_p": float(np.min(values)),
        "max_W_p": float(np.max(values)),
        "std_W_p": float(np.std(values)),
        "plaquette_count": int(len(flux_map)),
    }


def run_small_cluster_exact_diagonalization(
    geometry: Any,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
    solver: str = "auto",
    sparse_tol: float = 0.0,
    sparse_maxiter: int | None = None,
    use_sz_block: bool = True,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_target_parity: int = 0,
    use_translation_block: bool = False,
    use_translation_x_block: bool | None = None,
    use_translation_y_block: bool | None = None,
    momentum_block_1: int = 0,
    momentum_block_2: int = 0,
    momentum_x_block: int | None = None,
    momentum_y_block: int | None = None,
    use_reflection_block: bool = False,
    reflection_block: int = 0,
    check_symm: bool = False,
    check_herm: bool = False,
    check_pcon: bool = False,
) -> Tuple[float, np.ndarray]:
    """Run QuSpin sparse ED and return the ground-state energy and vector.

    The extra arguments match ``ed_backend.run_small_cluster_exact_diagonalization``.
    When a longitudinal ``Sz`` Zeeman field is used with a fixed-Sz request,
    every reachable total-Sz sector is checked and the absolute ground state is
    returned.
    """
    del model_spec, jx, jy, jz, solver, sparse_tol, sparse_maxiter

    field_terms = _field_terms(external_field_terms)
    transverse_field = _has_transverse_spin_field_terms(field_terms)
    scan_sz_sectors = bool(use_sz_block and _has_sz_zeeman_terms(field_terms) and not transverse_field)
    if transverse_field and bool(use_sz_block):
        if show_progress:
            print("[quspin-ed] transverse field breaks total Sz; using the full spin basis.")
        use_sz_block = False
        use_z2_block = False
    if _spin_field_breaks_z2(field_terms) and bool(use_z2_block):
        if show_progress:
            print("[quspin-ed] spin field breaks spin-flip Z2; disabling the Z2 block.")
        use_z2_block = False

    sector_targets = valid_total_m2_sectors(int(getattr(geometry, "number_of_sites"))) if scan_sz_sectors else [int(target_sz2)]
    progress_bar = _make_quspin_progress_bar(show_progress, total=len(sector_targets), desc="quspin ed", unit="sector")
    best: Tuple[float, np.ndarray] | None = None
    try:
        for sector_target_sz2 in sector_targets:
            hamiltonian_operator, basis, _static = build_quspin_yao_lee_hamiltonian(
                geometry=geometry,
                alpha=alpha,
                beta=beta,
                coupling_j=coupling_j,
                use_sz_block=use_sz_block,
                target_sz2=int(sector_target_sz2),
                use_tau_z_block=use_tau_z_block,
                target_tz2=target_tz2,
                use_z2_block=False if scan_sz_sectors else use_z2_block,
                z2_target_parity=z2_target_parity,
                use_translation_block=use_translation_block,
                use_translation_x_block=use_translation_x_block,
                use_translation_y_block=use_translation_y_block,
                momentum_block_1=momentum_block_1,
                momentum_block_2=momentum_block_2,
                momentum_x_block=momentum_x_block,
                momentum_y_block=momentum_y_block,
                use_reflection_block=use_reflection_block,
                reflection_block=reflection_block,
                external_field_terms=field_terms,
                check_symm=check_symm,
                check_herm=check_herm,
                check_pcon=check_pcon,
            )
            eigenvalues, eigenvectors = _solve_lowest_quspin_eigenpairs(
                hamiltonian_operator,
                basis,
                1,
                show_progress=show_progress,
                label=f"sector 2Sz={int(sector_target_sz2)}",
            )
            candidate = (float(eigenvalues[0]), np.asarray(eigenvectors[:, 0], dtype=np.complex128))
            if best is None or candidate[0] < best[0]:
                best = candidate
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    if best is None:
        raise RuntimeError("No QuSpin sector produced an eigenpair.")
    ground_energy = float(best[0])
    ground_vector = np.asarray(best[1], dtype=np.complex128)
    return ground_energy, ground_vector


def run_small_cluster_exact_spectrum(
    geometry: Any,
    model_spec: Any,
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
    solver: str = "auto",
    sparse_tol: float = 0.0,
    sparse_maxiter: int | None = None,
    use_sz_block: bool = True,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_target_parity: int = 0,
    use_translation_block: bool = False,
    use_translation_x_block: bool | None = None,
    use_translation_y_block: bool | None = None,
    momentum_block_1: int = 0,
    momentum_block_2: int = 0,
    momentum_x_block: int | None = None,
    momentum_y_block: int | None = None,
    use_reflection_block: bool = False,
    reflection_block: int = 0,
    check_symm: bool = False,
    check_herm: bool = False,
    check_pcon: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """QuSpin low-energy spectrum helper with an ``ed_backend.py``-like shape."""
    del model_spec, check_ground_state_degeneracy, jx, jy, jz
    del ground_manifold_abs_tol, ground_manifold_rel_tol, solver, sparse_tol, sparse_maxiter

    requested_use_sz_block = bool(use_sz_block)
    requested_target_sz2 = int(target_sz2)
    requested_use_z2_block = bool(use_z2_block)
    field_terms = _field_terms(external_field_terms)
    transverse_field = _has_transverse_spin_field_terms(field_terms)
    scan_sz_sectors = bool(use_sz_block and _has_sz_zeeman_terms(field_terms) and not transverse_field)
    if transverse_field and bool(use_sz_block):
        if show_progress:
            print("[quspin-ed] transverse field breaks total Sz; using the full spin basis.")
        use_sz_block = False
        use_z2_block = False
    if _spin_field_breaks_z2(field_terms) and bool(use_z2_block):
        if show_progress:
            print("[quspin-ed] spin field breaks spin-flip Z2; disabling the Z2 block.")
        use_z2_block = False
    if scan_sz_sectors:
        use_z2_block = False

    sector_targets = valid_total_m2_sectors(int(getattr(geometry, "number_of_sites"))) if scan_sz_sectors else [int(target_sz2)]
    progress_bar = _make_quspin_progress_bar(
        show_progress,
        total=len(sector_targets),
        desc="quspin ed spectrum",
        unit="sector",
    )
    best: Dict[str, Any] | None = None
    sector_scan_rows: List[Dict[str, Any]] = []
    try:
        for sector_target_sz2 in sector_targets:
            hamiltonian_operator, sector_basis, sector_static = build_quspin_yao_lee_hamiltonian(
                geometry=geometry,
                alpha=alpha,
                beta=beta,
                coupling_j=coupling_j,
                use_sz_block=use_sz_block,
                target_sz2=int(sector_target_sz2),
                use_tau_z_block=use_tau_z_block,
                target_tz2=target_tz2,
                use_z2_block=use_z2_block,
                z2_target_parity=z2_target_parity,
                use_translation_block=use_translation_block,
                use_translation_x_block=use_translation_x_block,
                use_translation_y_block=use_translation_y_block,
                momentum_block_1=momentum_block_1,
                momentum_block_2=momentum_block_2,
                momentum_x_block=momentum_x_block,
                momentum_y_block=momentum_y_block,
                use_reflection_block=use_reflection_block,
                reflection_block=reflection_block,
                external_field_terms=field_terms,
                check_symm=check_symm,
                check_herm=check_herm,
                check_pcon=check_pcon,
            )
            eigenvalues, eigenvectors = _solve_lowest_quspin_eigenpairs(
                hamiltonian_operator,
                sector_basis,
                eigenstate_count,
                show_progress=show_progress,
                label=f"sector 2Sz={int(sector_target_sz2)}",
            )
            sector_record = {
                "target_sz2": int(sector_target_sz2),
                "hilbert_dimension": int(sector_basis.Ns),
                "ground_state_energy": float(eigenvalues[0]),
                "eigenvalues": [float(value) for value in eigenvalues],
            }
            sector_scan_rows.append(sector_record)
            candidate = {
                "target_sz2": int(sector_target_sz2),
                "basis": sector_basis,
                "static": sector_static,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "dimension": int(sector_basis.Ns),
            }
            if best is None or float(eigenvalues[0]) < float(best["eigenvalues"][0]):
                best = candidate
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    if best is None:
        raise RuntimeError("No QuSpin sector produced an eigenpair.")
    basis = best["basis"]
    static = best["static"]
    dimension = int(best["dimension"])
    target_sz2 = int(best["target_sz2"])
    eigenvalues = np.asarray(best["eigenvalues"], dtype=float)
    eigenvectors = np.asarray(best["eigenvectors"], dtype=np.complex128)
    translation_x_used = bool(use_translation_block) if use_translation_x_block is None else bool(use_translation_x_block)
    translation_y_used = bool(use_translation_block) if use_translation_y_block is None else bool(use_translation_y_block)
    kx = int(momentum_block_1 if momentum_x_block is None else momentum_x_block)
    ky = int(momentum_block_2 if momentum_y_block is None else momentum_y_block)
    spectrum: Dict[str, Any] = {
        "backend": "quspin",
        "basis": (
            "tensor_basis("
            f"spin={'fixed Sz' if bool(use_sz_block) else 'full'}, "
            f"orbital={'fixed tau_z' if bool(use_tau_z_block) else 'full'}"
            ")"
        ),
        "use_sz_block": bool(use_sz_block),
        "target_sz2": int(target_sz2),
        "requested_use_sz_block": bool(requested_use_sz_block),
        "requested_target_sz2": int(requested_target_sz2),
        "use_tau_z_block": bool(use_tau_z_block),
        "target_tz2": int(target_tz2),
        "use_z2_block": bool(use_z2_block),
        "requested_use_z2_block": bool(requested_use_z2_block),
        "z2_target_parity": int(z2_target_parity) % 2,
        "use_translation_block": bool(translation_x_used or translation_y_used),
        "use_translation_x_block": bool(translation_x_used),
        "use_translation_y_block": bool(translation_y_used),
        "momentum_block_1": int(kx),
        "momentum_block_2": int(ky),
        "momentum_x_block": int(kx),
        "momentum_y_block": int(ky),
        "use_reflection_block": bool(use_reflection_block),
        "reflection_block": int(reflection_block),
        "formula": (
            "H = J sum_<ij>_gamma [(1+beta) S_i.S_j + "
            "(1-beta) tau_i^gamma tau_j^gamma + "
            "alpha (S_i.S_j)(tau_i^gamma tau_j^gamma)]"
        ),
        "hilbert_dimension": dimension,
        "static_term_count": len(static),
        "ground_state_energy": float(eigenvalues[0]),
        "eigenvalues": eigenvalues.tolist(),
        "solver": "quspin_eigsh",
        "ground_state_degeneracy_check_enabled": False,
        "ground_state_degeneracy_status": "not_checked",
        "ground_state_degeneracy": None,
        "external_field_terms": field_terms,
        "sz_sector_scan": {
            "enabled": bool(scan_sz_sectors),
            "reason": "longitudinal Sz Zeeman field with fixed-Sz basis"
            if scan_sz_sectors
            else None,
            "sectors": sector_scan_rows if scan_sz_sectors else [],
            "selected_target_sz2": int(target_sz2) if scan_sz_sectors else None,
        },
    }
    try:
        plaquette_flux = compute_plaquette_flux(
            basis,
            eigenvectors[:, 0],
            geometry,
            plaquette_center_idx=None,
        )
        spectrum["plaquette_flux"] = plaquette_flux
        spectrum["all_plaquette_fluxes"] = plaquette_flux.get("all_plaquette_fluxes", {})
        spectrum["plaquette_flux_map"] = plaquette_flux.get("plaquette_flux_map", {})
    except Exception as exc:
        spectrum["plaquette_flux"] = {"available": False, "warning": str(exc)}
        spectrum["all_plaquette_fluxes"] = {}
        spectrum["plaquette_flux_map"] = {}
    global_sector_levels = sorted(
        (
            (float(value), int(sector_row["target_sz2"]))
            for sector_row in sector_scan_rows
            for value in sector_row.get("eigenvalues", [])
        ),
        key=lambda item: item[0],
    )
    if scan_sz_sectors and len(global_sector_levels) > 1:
        spectrum["first_excited_energy"] = float(global_sector_levels[1][0])
        spectrum["first_excited_target_sz2"] = int(global_sector_levels[1][1])
        spectrum["spectral_gap"] = float(global_sector_levels[1][0] - global_sector_levels[0][0])
    elif len(eigenvalues) > 1:
        spectrum["first_excited_energy"] = float(eigenvalues[1])
        spectrum["spectral_gap"] = float(eigenvalues[1] - eigenvalues[0])
    else:
        spectrum["first_excited_energy"] = None
        spectrum["spectral_gap"] = None
    return spectrum, eigenvectors
