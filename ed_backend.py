#!/usr/bin/env python3
"""Exact-diagonalization backends for the Yao-Lee driver.

This module owns both full-Hilbert exact diagonalization and the strict
bitwise total-Sz=0 spin/orbital sparse ED path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from scipy.sparse.linalg import ArpackNoConvergence

from analysis import _end_stage, _make_progress_bar, _start_stage, resolve_low_energy_spectrum
from models import (
    GeometryData,
    ISING_AXIS,
    MODEL_FAMILY,
    ORBITAL_REP,
    SPIN_REP,
    ModelSpec,
    all_high_symmetry_structure_factors,
    build_model_spec,
    build_site_ops,
    honeycomb_plaquette_flux_operators,
    is_trivial_orbital,
    model_terms_for_bond,
    plaquette_flux_close_to_target,
    select_honeycomb_plaquette_flux_operator,
    two_site_operator_terms_for_bond,
)

# Full-Hilbert ED
# ----------------------------------------------------------------------

ED_EIGSH_DEFAULT_TOL = 1e-10
ED_EIGSH_DEFAULT_MAXITER = 20000
ED_EIGSH_DEGENERACY_PADDING = 4
ED_EIGSH_MIN_NCV = 20
ED_EIGSH_NCV_MULTIPLIER = 4
ED_EIGSH_RANDOM_SEED = 24681357

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

    bond_terms: List[Tuple[Any, List[Tuple[complex, str, str]]]] = []
    total_terms = 0
    field_terms = list(external_field_terms or [])
    for bond in geometry.bond_list:
        terms = list(
            two_site_operator_terms_for_bond(
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
        for coeff, op_i_name, op_j_name in terms:
            op_list = [ident] * n_sites
            op_list[i] = op_cache[op_i_name]
            op_list[j] = op_cache[op_j_name]
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
    solver: str = "auto",
    sparse_tol: float = 0.0,
    sparse_maxiter: int | None = None,
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
        solver=solver,
        sparse_tol=sparse_tol,
        sparse_maxiter=sparse_maxiter,
    )
    return float(spectrum["ground_state_energy"]), eigenvectors[:, 0]


def _normalize_ed_solver(solver: str | None) -> str:
    solver_name = "auto" if solver is None else str(solver).strip().lower()
    if solver_name not in ("auto", "dense", "sparse"):
        raise ValueError(
            f"Unknown ED solver '{solver}'. Expected one of: auto, dense, sparse."
        )
    return solver_name


def _padded_eigsh_count(
    requested_count: int,
    matrix_dimension: int,
    *,
    check_ground_state_degeneracy: bool,
) -> Tuple[int, int]:
    """Choose k for eigsh, optionally asking for a few extra states.

    Extra low-energy states make ground-manifold checks less fragile when the
    lowest level is nearly or exactly degenerate.
    """
    dim = int(matrix_dimension)
    if dim <= 1:
        return 1, 0
    requested = max(1, int(requested_count))
    padding = ED_EIGSH_DEGENERACY_PADDING if bool(check_ground_state_degeneracy) else 0
    solve_count = min(dim - 1, requested + padding)
    return solve_count, max(0, solve_count - requested)


def _eigsh_ncv(matrix_dimension: int, eigenstate_count: int) -> int | None:
    """Use a larger Krylov subspace than SciPy's default for clustered spectra."""
    dim = int(matrix_dimension)
    k = int(eigenstate_count)
    if dim <= k + 1:
        return None
    ncv = max(ED_EIGSH_MIN_NCV, ED_EIGSH_NCV_MULTIPLIER * k + 1, 2 * k + 1)
    ncv = min(dim, ncv)
    if ncv <= k:
        return None
    return int(ncv)


def _eigsh_start_vector(matrix_dimension: int) -> np.ndarray:
    """Deterministic nonzero start vector avoids unlucky starts in degenerate sectors."""
    rng = np.random.default_rng(ED_EIGSH_RANDOM_SEED + int(matrix_dimension))
    vector = rng.standard_normal(int(matrix_dimension))
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector
    return vector / norm


def _run_lowest_eigsh(
    hamiltonian: sparse.spmatrix,
    *,
    eigenstate_count: int,
    sparse_tol: float,
    sparse_maxiter: int | None,
    show_progress: bool,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run ARPACK with the sparse-matrix settings needed for ED workloads."""
    if show_progress:
        print(f"[{label}] preparing sparse matrix")
    hamiltonian_csr = hamiltonian.tocsr()
    hamiltonian_csr.sum_duplicates()
    hamiltonian_csr.eliminate_zeros()

    dim = int(hamiltonian_csr.shape[0])
    k = int(eigenstate_count)
    if not (1 <= k < dim):
        raise ValueError(f"eigsh requires 1 <= k < matrix dimension; got k={k}, dim={dim}.")

    effective_tol = float(sparse_tol) if float(sparse_tol) > 0.0 else ED_EIGSH_DEFAULT_TOL
    effective_maxiter = (
        int(sparse_maxiter)
        if sparse_maxiter is not None and int(sparse_maxiter) > 0
        else ED_EIGSH_DEFAULT_MAXITER
    )
    ncv = _eigsh_ncv(dim, k)
    eigsh_kwargs: Dict[str, Any] = {
        "k": k,
        "which": "SA",
        "tol": effective_tol,
        "maxiter": effective_maxiter,
        "v0": _eigsh_start_vector(dim),
    }
    if ncv is not None:
        eigsh_kwargs["ncv"] = ncv

    info: Dict[str, Any] = {
        "matrix_format": hamiltonian_csr.getformat(),
        "eigsh_k": k,
        "eigsh_which": "SA",
        "eigsh_ncv": ncv,
        "eigsh_tol_requested": float(sparse_tol),
        "eigsh_tol_effective": effective_tol,
        "eigsh_maxiter": effective_maxiter,
        "eigsh_start_vector": "deterministic_random_normalized",
        "csr_conversion": "mandatory_before_eigsh",
    }
    if show_progress:
        print(
            f"[{label}] eigsh started: dim={dim}, nnz={hamiltonian_csr.nnz}, k={k}"
        )
    try:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(hamiltonian_csr, **eigsh_kwargs)
        info["arpack_status"] = "converged"
        return eigenvalues, eigenvectors, info
    except ArpackNoConvergence as exc:
        partial_values = np.asarray(exc.eigenvalues)
        partial_vectors = exc.eigenvectors
        if partial_values.size > 0 and partial_vectors is not None and partial_vectors.shape[1] > 0:
            info["arpack_status"] = "partial_convergence"
            info["arpack_converged_eigenpairs"] = int(partial_values.size)
            info["arpack_message"] = str(exc)
            return partial_values, partial_vectors, info
        raise RuntimeError(
            f"{label} eigsh did not converge and returned no partial eigenpairs. "
            f"Try increasing sparse_maxiter above {effective_maxiter}, loosening sparse_tol, "
            "or lowering the requested eigenstate count."
        ) from exc


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
    solver: str = "auto",
    sparse_tol: float = 0.0,
    sparse_maxiter: int | None = None,
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
    solver_requested = _normalize_ed_solver(solver)
    solver_note = None
    sparse_solve_count, degeneracy_padding = _padded_eigsh_count(
        requested_count,
        hilbert_dim,
        check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
    )
    solve_count = (
        sparse_solve_count
        if solver_requested == "sparse"
        else min(requested_count + degeneracy_padding, hilbert_dim)
    )
    if solver_requested == "sparse" and requested_count >= hilbert_dim:
        if hilbert_dim <= 1:
            solver_note = (
                "Sparse eigsh requires k < Hilbert dimension; dense fallback used "
                "for this one-dimensional Hamiltonian."
            )
        else:
            solver_note = (
                "Sparse eigsh requires k < Hilbert dimension; capped returned "
                f"eigenstates from {requested_count} to {solve_count}."
            )
    if show_progress:
        print(
            "[ed] eigensolve started: "
            f"dim={hamiltonian.shape[0]}, nnz={hamiltonian.nnz}, "
            f"k={solve_count}, solver={solver_requested}"
        )
    use_dense = (
        solver_requested == "dense"
        or (solver_note is not None and hilbert_dim <= 1)
        or (solver_requested == "auto" and solve_count >= hilbert_dim - 1)
    )
    if use_dense:
        dense_hamiltonian = hamiltonian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
        eigenvalues = eigenvalues[:solve_count]
        eigenvectors = eigenvectors[:, :solve_count]
        solver_mode = "dense"
        eigsh_info: Dict[str, Any] = {}
    else:
        eigenvalues, eigenvectors, eigsh_info = _run_lowest_eigsh(
            hamiltonian,
            eigenstate_count=solve_count,
            sparse_tol=sparse_tol,
            sparse_maxiter=sparse_maxiter,
            show_progress=show_progress,
            label="ed",
        )
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
        "solver_requested": solver_requested,
        "hilbert_dim": hilbert_dim,
        "eigenstates_requested": requested_count,
        "eigenstates_returned": int(eigenvalues.size),
        "eigenstates_degeneracy_padding": int(degeneracy_padding),
        "energies": [float(value) for value in eigenvalues],
        "ground_state_energy": e0,
        **low_energy_resolution,
    }
    if solver_note is not None:
        spectrum["solver_note"] = solver_note
    if solver_mode == "sparse":
        spectrum["sparse_tol"] = float(eigsh_info.get("eigsh_tol_effective", sparse_tol))
        spectrum["sparse_tol_requested"] = float(sparse_tol)
        spectrum["sparse_maxiter"] = int(eigsh_info.get("eigsh_maxiter", 0))
        spectrum["eigsh"] = eigsh_info
    try:
        plaquette_flux = plaquette_flux_from_ed_state(
            geometry,
            np.asarray(eigenvectors[:, 0], dtype=np.complex128),
            model_spec,
            plaquette_center_idx=None,
        )
        spectrum["plaquette_flux"] = plaquette_flux
        spectrum["all_plaquette_fluxes"] = plaquette_flux.get("all_plaquette_fluxes", {})
        spectrum["plaquette_flux_map"] = plaquette_flux.get("plaquette_flux_map", {})
    except Exception as exc:
        spectrum["plaquette_flux"] = {"available": False, "warning": str(exc)}
        spectrum["all_plaquette_fluxes"] = {}
        spectrum["plaquette_flux_map"] = {}
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


def plaquette_flux_from_ed_state(
    geometry: GeometryData,
    state: np.ndarray,
    model_spec: ModelSpec,
    plaquette_center_idx: int | None = None,
) -> Dict[str, Any]:
    """Evaluate normalized honeycomb plaquette flux from a full ED state."""
    n_sites = int(geometry.number_of_sites)
    op_cache = build_global_operator_cache_for_model(model_spec)
    ident = op_cache["Id"]
    plaquettes = honeycomb_plaquette_flux_operators(geometry)
    if len(plaquettes) == 0:
        raise ValueError("No honeycomb length-six plaquette was found in this geometry.")
    selected = select_honeycomb_plaquette_flux_operator(geometry, plaquette_center_idx)
    selected_index = int(selected["plaquette_index"])
    flux_map: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}
    for plaquette in plaquettes:
        op_list = [ident] * n_sites
        for site, operator_name in zip(plaquette["sites"], plaquette["operator_names"]):
            op_list[int(site)] = 2.0 * op_cache[str(operator_name)]
        flux_op = kron_all(op_list)
        value = complex(np.vdot(state, flux_op.dot(state)))
        normalized_value = float(np.real(value))
        plaquette_index = int(plaquette["plaquette_index"])
        flux_map[plaquette_index] = normalized_value
        details[plaquette_index] = {
            "plaquette_index": plaquette_index,
            "sites": [int(site) for site in plaquette["sites"]],
            "axes": [str(axis) for axis in plaquette["axes"]],
            "operators": [str(op) for op in plaquette["operator_names"]],
            "raw_tau_product": float(np.real(value) / float(plaquette["normalization"])),
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


def compute_all_plaquette_fluxes(
    geometry: GeometryData,
    state: np.ndarray,
    model_spec: ModelSpec,
) -> Dict[int, float]:
    """Return normalized ``W_p`` on every valid elementary honeycomb plaquette."""
    flux_payload = plaquette_flux_from_ed_state(
        geometry,
        state,
        model_spec,
        plaquette_center_idx=None,
    )
    flux_map = flux_payload.get("all_plaquette_fluxes", flux_payload.get("plaquette_flux_map", {}))
    if not isinstance(flux_map, dict):
        return {}
    return {int(index): float(value) for index, value in flux_map.items()}


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
    op_pairs.extend(
        [
            (f"S{spin_axis}T{orbital_axis}", f"S{spin_axis}T{orbital_axis}")
            for orbital_axis in ("x", "y", "z")
            for spin_axis in ("x", "y", "z")
        ]
    )
    op_pairs = list(dict.fromkeys(op_pairs))
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
        "ST": np.zeros_like(correlations["Sx_Sx"]),
    }
    for gamma in ("x", "y", "z"):
        scalar["S"] = scalar["S"] + correlations[f"S{gamma}_S{gamma}"]
        scalar["T"] = scalar["T"] + correlations[f"T{gamma}_T{gamma}"]
    mixed_terms_found = False
    for orbital_axis in ("x", "y", "z"):
        for spin_axis in ("x", "y", "z"):
            key = f"S{spin_axis}T{orbital_axis}_S{spin_axis}T{orbital_axis}"
            if key in correlations:
                scalar["ST"] = scalar["ST"] + correlations[key]
                mixed_terms_found = True
    if not mixed_terms_found:
        for gamma in ("x", "y", "z"):
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
    return float(
        np.sum(
            [
                component["energy"]
                for component in bond_energy_components_from_correlations(
                    i,
                    j,
                    gamma,
                    correlations,
                    model_spec,
                    alpha,
                    beta,
                    coupling_j,
                    jx=jx,
                    jy=jy,
                    jz=jz,
                )
            ]
        )
    )


def _bond_energy_channel_from_operator(op_name: str) -> str:
    if op_name.startswith("ST"):
        return "ST"
    if op_name.startswith("S") and "T" in op_name:
        return "ST"
    if op_name.startswith("T"):
        return "T"
    if op_name.startswith("S"):
        return "S"
    return op_name


def bond_energy_components_from_correlations(
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
) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    if str(model_spec.model_family).strip().lower() == "yao_lee" and not is_trivial_orbital(model_spec):
        axis = str(gamma).strip().lower()
        spin_dot = sum(complex(correlations[f"S{spin_axis}_S{spin_axis}"][i, j]) for spin_axis in ("x", "y", "z"))
        orbital_gamma = complex(correlations[f"T{axis}_T{axis}"][i, j])
        mixed_gamma = sum(
            complex(correlations[f"S{spin_axis}T{axis}_S{spin_axis}T{axis}"][i, j])
            for spin_axis in ("x", "y", "z")
        )
        physical_components = [
            ("S", "Sdot", "dot", float(coupling_j) * (1.0 + float(beta)), spin_dot),
            ("T", f"T{axis}", axis, float(coupling_j) * (1.0 - float(beta)), orbital_gamma),
            ("ST", f"SdotT{axis}", axis, float(coupling_j) * float(alpha), mixed_gamma),
        ]
        for channel, operator, component_axis, coeff, correlation_value in physical_components:
            energy_value = coeff * correlation_value
            components.append(
                {
                    "channel": channel,
                    "operator": operator,
                    "axis": component_axis,
                    "coefficient": float(coeff),
                    "correlation": float(np.real(correlation_value)),
                    "energy": float(np.real(energy_value)),
                }
            )
        return components

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
        correlation_value = correlations[key][i, j]
        energy_value = coeff * correlation_value
        channel = _bond_energy_channel_from_operator(str(op_name))
        components.append(
            {
                "channel": channel,
                "operator": str(op_name),
                "axis": str(op_name).replace(channel, "", 1).lower(),
                "coefficient": float(np.real(coeff)),
                "correlation": float(np.real(correlation_value)),
                "energy": float(np.real(energy_value)),
            }
        )
    return components


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
        components = bond_energy_components_from_correlations(
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
        )
        channel_energies: Dict[str, float] = {}
        for component in components:
            channel = str(component["channel"])
            channel_energies[channel] = channel_energies.get(channel, 0.0) + float(component["energy"])
        rows.append(
            {
                "i": bond.i,
                "j": bond.j,
                "gamma": bond.gamma,
                "O_ij_gamma": float(np.sum([float(component["energy"]) for component in components])),
                "components": components,
                "channel_energies": channel_energies,
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
    thermal_eigsh_info: Dict[str, Any] = {}
    if hilbert_dim <= int(full_spectrum_max_dim):
        if show_progress:
            print(f"[thermal-ed] dense full diagonalization started: dim={hilbert_dim}")
        dense_hamiltonian = hamiltonian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
        spectrum_mode = "full"
        full_spectrum = True
    else:
        eigenstate_count = max(1, min(int(max_eigenstates), hilbert_dim - 1))
        if show_progress:
            print(
                "[thermal-ed] sparse low-energy eigensolve started: "
                f"dim={hilbert_dim}, nnz={hamiltonian.nnz}, k={eigenstate_count}"
            )
        eigenvalues, eigenvectors, thermal_eigsh_info = _run_lowest_eigsh(
            hamiltonian,
            eigenstate_count=eigenstate_count,
            sparse_tol=0.0,
            sparse_maxiter=None,
            show_progress=show_progress,
            label="thermal-ed",
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
            "eigsh": thermal_eigsh_info if not full_spectrum else None,
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


# Bitwise ED in the strict total-Sz=0 sector
# ----------------------------------------------------------------------

BITWISE_ED_FORMULA = (
    "H = J sum_<ij>_gamma [(1+beta) S_i.S_j + (1-beta) T_i^gamma T_j^gamma "
    "+ alpha (S_i.S_j)(T_i^gamma T_j^gamma)]"
)
BITWISE_ED_NOTE = (
    "Many-body states are represented only as (spin_state, orbital_state), "
    "where bit 0 means down and bit 1 means up. The ED basis fixes "
    "one requested total-Sz spin sector and leaves orbital_state unrestricted."
)


def _spin_half_nup_from_total_sz2(n_sites: int, target_sz2: int) -> int:
    numerator = int(n_sites) + int(target_sz2)
    if numerator % 2 != 0:
        return -1
    nup = numerator // 2
    if nup < 0 or nup > int(n_sites):
        return -1
    return int(nup)


def estimate_sz_conserved_dimension(n_sites: int, target_sz2: int = 0) -> int:
    """Return dim(C(N,Nup) x 2^N) for a fixed total-Sz spin/orbital sector."""
    n = int(n_sites)
    if n < 0:
        raise ValueError("n_sites must be non-negative.")
    target_up_spins = _spin_half_nup_from_total_sz2(n, int(target_sz2))
    if target_up_spins < 0:
        return 0
    return int(math.comb(n, target_up_spins) * (1 << n))


def _bit_is_up(state: int, site: int) -> bool:
    return bool((int(state) >> int(site)) & 1)


def _z_value_from_bit(state: int, site: int) -> float:
    return 0.5 if _bit_is_up(state, site) else -0.5


def build_sz_conserved_basis(
    N: int,
    target_sz2: int = 0,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Build a fixed total-Sz ED basis using two binary integers per state.

    Returns:
      basis_list:
        List of ``(spin_state, orbital_state)`` tuples.
      basis_map:
        Dictionary mapping each tuple to its matrix row/column index.

    The spin convention is bit 0 = down, bit 1 = up. The orbital convention is
    also bit 0 = down, bit 1 = up. No base-4 many-body integer is used here.
    """
    n_sites = int(N)
    if n_sites < 0:
        raise ValueError("N must be non-negative.")
    target_up_spins = _spin_half_nup_from_total_sz2(n_sites, int(target_sz2))
    if target_up_spins < 0:
        raise ValueError(f"Total 2*Sz={int(target_sz2)} is unreachable for {n_sites} spin-1/2 sites.")
    spin_limit = 1 << n_sites
    orbital_limit = 1 << n_sites
    spin_basis = [
        spin_state
        for spin_state in range(spin_limit)
        if int(spin_state).bit_count() == target_up_spins
    ]

    basis_list: List[Tuple[int, int]] = []
    basis_map: Dict[Tuple[int, int], int] = {}
    for spin_state in spin_basis:
        for orbital_state in range(orbital_limit):
            key = (int(spin_state), int(orbital_state))
            basis_map[key] = len(basis_list)
            basis_list.append(key)
    return basis_list, basis_map


def _spin_dot_actions(spin_state: int, i: int, j: int) -> List[Tuple[int, complex]]:
    """Actions of S_i.S_j that preserve total Sz for spin-1/2 bits."""
    mask_i = 1 << int(i)
    mask_j = 1 << int(j)
    up_i = bool(spin_state & mask_i)
    up_j = bool(spin_state & mask_j)
    actions: List[Tuple[int, complex]] = [
        (int(spin_state), complex((0.5 if up_i else -0.5) * (0.5 if up_j else -0.5)))
    ]

    # 1/2 * S_i^+ S_j^- and 1/2 * S_i^- S_j^+.
    # They are allowed only when one spin is down and the other is up.
    if up_i != up_j:
        actions.append((int(spin_state ^ (mask_i | mask_j)), 0.5 + 0.0j))
    return actions


def _orbital_pair_actions(orbital_state: int, i: int, j: int, gamma: str) -> List[Tuple[int, complex]]:
    """Actions of T_i^gamma T_j^gamma on orbital-1/2 bits.

    Orbital total Tz is intentionally not conserved, so x/y bonds flip orbital
    bits without any sector check.
    """
    axis = str(gamma).strip().lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Unsupported orbital bond axis '{gamma}'.")

    mask_i = 1 << int(i)
    mask_j = 1 << int(j)
    up_i = bool(orbital_state & mask_i)
    up_j = bool(orbital_state & mask_j)
    if axis == "z":
        return [
            (
                int(orbital_state),
                complex((0.5 if up_i else -0.5) * (0.5 if up_j else -0.5)),
            )
        ]
    if axis == "x":
        return [(int(orbital_state ^ (mask_i | mask_j)), 0.25 + 0.0j)]

    # Ty |down> = -i/2 |up>, Ty |up> = +i/2 |down>.
    coeff_i = 0.5j if up_i else -0.5j
    coeff_j = 0.5j if up_j else -0.5j
    return [(int(orbital_state ^ (mask_i | mask_j)), complex(coeff_i * coeff_j))]


def _single_spin_axis_actions(spin_state: int, site: int, axis: str) -> List[Tuple[int, complex]]:
    site_mask = 1 << int(site)
    up = bool(spin_state & site_mask)
    axis_text = str(axis).lower()
    if axis_text == "z":
        return [(int(spin_state), 0.5 + 0.0j if up else -0.5 + 0.0j)]
    if axis_text == "x":
        return [(int(spin_state ^ site_mask), 0.5 + 0.0j)]
    if axis_text == "y":
        return [(int(spin_state ^ site_mask), 0.5j if up else -0.5j)]
    raise ValueError(f"Unknown spin axis '{axis}'.")


def _single_orbital_axis_actions(orbital_state: int, site: int, axis: str) -> List[Tuple[int, complex]]:
    site_mask = 1 << int(site)
    up = bool(orbital_state & site_mask)
    axis_text = str(axis).lower()
    if axis_text == "z":
        return [(int(orbital_state), 0.5 + 0.0j if up else -0.5 + 0.0j)]
    if axis_text == "x":
        return [(int(orbital_state ^ site_mask), 0.5 + 0.0j)]
    if axis_text == "y":
        return [(int(orbital_state ^ site_mask), 0.5j if up else -0.5j)]
    raise ValueError(f"Unknown orbital axis '{axis}'.")


def _parse_one_site_bitwise_operator(op_name: str) -> Tuple[str | None, str | None]:
    """Return (spin_axis, orbital_axis) for a one-site operator label."""
    text = str(op_name).strip()
    if text == "Id":
        return None, None
    if text in ("Sx", "Sy", "Sz"):
        return text[1].lower(), None
    if text in ("Tx", "Ty", "Tz"):
        return None, text[1].lower()
    if text in ("STx", "STy", "STz"):
        axis = text[2].lower()
        return axis, axis
    if len(text) == 4 and text[0] == "S" and text[2] == "T":
        spin_axis = text[1].lower()
        orbital_axis = text[3].lower()
        if spin_axis in ("x", "y", "z") and orbital_axis in ("x", "y", "z"):
            return spin_axis, orbital_axis
    raise ValueError(f"Unsupported bitwise one-site operator '{op_name}'.")


def _apply_one_site_bitwise_operator(
    spin_state: int,
    orbital_state: int,
    site: int,
    op_name: str,
) -> List[Tuple[int, int, complex]]:
    spin_axis, orbital_axis = _parse_one_site_bitwise_operator(op_name)
    spin_actions = (
        _single_spin_axis_actions(spin_state, site, spin_axis)
        if spin_axis is not None
        else [(int(spin_state), 1.0 + 0.0j)]
    )
    out: List[Tuple[int, int, complex]] = []
    for next_spin, spin_coeff in spin_actions:
        orbital_actions = (
            _single_orbital_axis_actions(orbital_state, site, orbital_axis)
            if orbital_axis is not None
            else [(int(orbital_state), 1.0 + 0.0j)]
        )
        for next_orbital, orbital_coeff in orbital_actions:
            out.append((int(next_spin), int(next_orbital), complex(spin_coeff * orbital_coeff)))
    return out


def _two_site_bitwise_operator_actions(
    spin_state: int,
    orbital_state: int,
    i: int,
    op_i: str,
    j: int,
    op_j: str,
) -> List[Tuple[int, int, complex]]:
    first_actions = _apply_one_site_bitwise_operator(spin_state, orbital_state, i, op_i)
    out: List[Tuple[int, int, complex]] = []
    for spin_mid, orbital_mid, coeff_i in first_actions:
        for spin_out, orbital_out, coeff_j in _apply_one_site_bitwise_operator(
            spin_mid,
            orbital_mid,
            j,
            op_j,
        ):
            out.append((int(spin_out), int(orbital_out), complex(coeff_i * coeff_j)))
    return out


def build_sparse_hamiltonian_sz_conserved(
    N: int,
    geometry: GeometryData,
    alpha: float,
    beta: float,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    *,
    coupling_j: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
) -> sparse.csr_matrix:
    """Build the sparse Hamiltonian in the strict total-Sz=0 basis.

    The implemented spin/orbital form is recorded in ``BITWISE_ED_FORMULA``.
    It uses spin-isotropic ``S_i.S_j`` terms, so total spin Sz is conserved,
    and bond-dependent orbital ``T_i^gamma T_j^gamma`` terms, so orbital Tz is
    not conserved.

    Bitwise off-diagonal template for S_i^+ S_j^- T_i^z T_j^z:

        mask_i = 1 << i
        mask_j = 1 << j
        if (S & mask_i) == 0 and (S & mask_j) != 0:
            new_S = S ^ (mask_i | mask_j)
            new_O = O
            tau_z_i = 0.5 if (O & mask_i) else -0.5
            tau_z_j = 0.5 if (O & mask_j) else -0.5
            row = basis_map.get((new_S, new_O))
            if row is not None:
                H[row, col] += coeff * tau_z_i * tau_z_j

    No dense ``4**N`` Hamiltonian is allocated by this function.
    """
    n_sites = int(N)
    if n_sites != int(geometry.number_of_sites):
        raise ValueError(
            f"N={n_sites} does not match geometry.number_of_sites={int(geometry.number_of_sites)}."
        )
    if len(basis_list) != len(basis_map):
        raise ValueError("basis_list and basis_map sizes do not match.")
    target_up_counts = {int(spin_state).bit_count() for spin_state, _orbital_state in basis_list}
    if len(target_up_counts) != 1:
        raise ValueError("basis_list must contain exactly one fixed total-Sz spin sector.")
    target_up_spins = int(next(iter(target_up_counts))) if target_up_counts else 0

    for state in basis_list:
        if len(state) != 2:
            raise ValueError("Each basis state must be a (spin_state, orbital_state) tuple.")
        spin_state, orbital_state = int(state[0]), int(state[1])
        if spin_state.bit_count() != target_up_spins:
            raise ValueError(f"Invalid spin_state {spin_state}: expected {target_up_spins} up spins.")
        if spin_state < 0 or spin_state >= (1 << n_sites):
            raise ValueError(f"spin_state {spin_state} is outside the N-bit range.")
        if orbital_state < 0 or orbital_state >= (1 << n_sites):
            raise ValueError(f"orbital_state {orbital_state} is outside the N-bit range.")

    dim = int(len(basis_list))
    hamiltonian = sparse.lil_matrix((dim, dim), dtype=np.complex128)
    field_terms = list(external_field_terms or [])
    unsupported_field_terms = [
        op_name for _coefficient, op_name in field_terms if str(op_name) not in ("Sz",)
    ]
    if unsupported_field_terms:
        raise ValueError(
            "The Sz-conserved ED basis only accepts external-field Hamiltonian terms "
            f"that conserve total Sz. Unsupported one-site terms: {unsupported_field_terms}."
        )

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=dim,
        desc="Sz-ED H columns",
        unit="state",
        leave=False,
    )
    spin_coeff = float(coupling_j) * (1.0 + float(beta))
    orbital_coeff = float(coupling_j) * (1.0 - float(beta))
    mixed_coeff = float(coupling_j) * float(alpha)

    for col, (spin_state_raw, orbital_state_raw) in enumerate(basis_list):
        spin_state = int(spin_state_raw)
        orbital_state = int(orbital_state_raw)

        for bond in geometry.bond_list:
            i = int(bond.i)
            j = int(bond.j)
            gamma = str(bond.gamma).lower()

            if abs(spin_coeff) > 1e-14:
                for next_spin, spin_matrix_element in _spin_dot_actions(spin_state, i, j):
                    row = basis_map.get((int(next_spin), orbital_state))
                    if row is not None:
                        hamiltonian[row, col] += spin_coeff * spin_matrix_element

            if abs(orbital_coeff) > 1e-14:
                for next_orbital, orbital_matrix_element in _orbital_pair_actions(
                    orbital_state,
                    i,
                    j,
                    gamma,
                ):
                    row = basis_map.get((spin_state, int(next_orbital)))
                    if row is not None:
                        hamiltonian[row, col] += orbital_coeff * orbital_matrix_element

            if abs(mixed_coeff) > 1e-14:
                spin_actions = _spin_dot_actions(spin_state, i, j)
                orbital_actions = _orbital_pair_actions(orbital_state, i, j, gamma)
                for next_spin, spin_matrix_element in spin_actions:
                    for next_orbital, orbital_matrix_element in orbital_actions:
                        row = basis_map.get((int(next_spin), int(next_orbital)))
                        if row is not None:
                            hamiltonian[row, col] += (
                                mixed_coeff * spin_matrix_element * orbital_matrix_element
                            )

        for site in range(n_sites):
            spin_z = _z_value_from_bit(spin_state, site)
            for coefficient, _op_name in field_terms:
                hamiltonian[col, col] += float(coefficient) * spin_z

        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    return hamiltonian.tocsr()


def run_sz_conserved_exact_spectrum(
    geometry: GeometryData,
    alpha: float,
    beta: float,
    *,
    coupling_j: float = 1.0,
    eigenstate_count: int = 3,
    check_ground_state_degeneracy: bool = True,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
    ground_manifold_abs_tol: float = 1e-12,
    ground_manifold_rel_tol: float = 1e-12,
    sparse_tol: float = 0.0,
    sparse_maxiter: int | None = None,
    target_sz2: int = 0,
) -> Tuple[Dict[str, Any], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Diagonalize the bitwise fixed-total-Sz sparse Hamiltonian with ARPACK eigsh."""
    n_sites = int(geometry.number_of_sites)
    stage_start = _start_stage("Sz-conserved ED", show_progress)
    basis_list, basis_map = build_sz_conserved_basis(n_sites, target_sz2=target_sz2)
    hamiltonian = build_sparse_hamiltonian_sz_conserved(
        n_sites,
        geometry,
        alpha,
        beta,
        basis_list,
        basis_map,
        coupling_j=coupling_j,
        external_field_terms=external_field_terms,
        show_progress=show_progress,
    )
    dim = int(hamiltonian.shape[0])
    if dim <= 0:
        raise ValueError("Empty Sz-conserved ED basis.")
    requested_count = max(1, int(eigenstate_count))
    if dim == 1:
        eigenvalues = np.asarray([complex(hamiltonian[0, 0])], dtype=np.complex128)
        eigenvectors = np.ones((1, 1), dtype=np.complex128)
        solver_mode = "reduced_dense_dim1"
        solve_count = 1
        degeneracy_padding = 0
        eigsh_info: Dict[str, Any] = {}
    else:
        solve_count, degeneracy_padding = _padded_eigsh_count(
            requested_count,
            dim,
            check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
        )
        if show_progress:
            print(
                "[sz-ed] basis ready: "
                f"dim={dim}, nnz={hamiltonian.nnz}, k={solve_count}"
            )
        eigenvalues, eigenvectors, eigsh_info = _run_lowest_eigsh(
            hamiltonian,
            eigenstate_count=solve_count,
            sparse_tol=sparse_tol,
            sparse_maxiter=sparse_maxiter,
            show_progress=show_progress,
            label="sz-ed",
        )
        solver_mode = "sz_conserved_sparse"

    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.asarray(np.real(eigenvalues[order]), dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    _end_stage("Sz-conserved ED", stage_start, show_progress)

    low_energy_resolution = resolve_low_energy_spectrum(
        eigenvalues,
        check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
        hilbert_dim=dim,
        degeneracy_tolerance_abs=float(ground_manifold_abs_tol),
        degeneracy_tolerance_rel=float(ground_manifold_rel_tol),
    )
    spectrum: Dict[str, Any] = {
        "solver_mode": solver_mode,
        "solver_requested": "sz_conserved_sparse",
        "basis_type": "bitwise_spin_orbital_total_sz_block",
        "target_sz2": int(target_sz2),
        "formula": BITWISE_ED_FORMULA,
        "basis_note": BITWISE_ED_NOTE,
        "hilbert_dim": dim,
        "full_spin_orbital_hilbert_dim": int(4 ** n_sites),
        "number_of_sites": n_sites,
        "eigenstates_requested": requested_count,
        "eigenstates_returned": int(eigenvalues.size),
        "eigenstates_degeneracy_padding": int(degeneracy_padding),
        "energies": [float(value) for value in eigenvalues],
        "ground_state_energy": float(low_energy_resolution["ground_state_energy"]),
        "sparse_tol": float(eigsh_info.get("eigsh_tol_effective", sparse_tol)),
        "sparse_tol_requested": float(sparse_tol),
        "sparse_maxiter": eigsh_info.get("eigsh_maxiter"),
        "eigsh": eigsh_info if solver_mode == "sz_conserved_sparse" else None,
        **low_energy_resolution,
    }
    if requested_count >= dim and dim > 1:
        spectrum["solver_note"] = (
            "eigsh requires k < basis dimension; returned eigenstate count was capped "
            f"from {requested_count} to {solve_count}."
        )
    try:
        plaquette_flux = plaquette_flux_from_sz_conserved_ed_state(
            geometry,
            np.asarray(eigenvectors[:, 0], dtype=np.complex128),
            basis_list,
            basis_map,
            plaquette_center_idx=None,
        )
        spectrum["plaquette_flux"] = plaquette_flux
        spectrum["all_plaquette_fluxes"] = plaquette_flux.get("all_plaquette_fluxes", {})
        spectrum["plaquette_flux_map"] = plaquette_flux.get("plaquette_flux_map", {})
    except Exception as exc:
        spectrum["plaquette_flux"] = {"available": False, "warning": str(exc)}
        spectrum["all_plaquette_fluxes"] = {}
        spectrum["plaquette_flux_map"] = {}
    return spectrum, eigenvectors, basis_list, basis_map


def _expectation_two_site_bitwise(
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    i: int,
    op_i: str,
    j: int,
    op_j: str,
    *,
    amplitude_tol: float = 1e-15,
) -> complex:
    value = 0.0j
    for col, (spin_state_raw, orbital_state_raw) in enumerate(basis_list):
        ket_amp = complex(state[col])
        if abs(ket_amp) <= amplitude_tol:
            continue
        spin_state = int(spin_state_raw)
        orbital_state = int(orbital_state_raw)
        for next_spin, next_orbital, matrix_element in _two_site_bitwise_operator_actions(
            spin_state,
            orbital_state,
            i,
            op_i,
            j,
            op_j,
        ):
            row = basis_map.get((int(next_spin), int(next_orbital)))
            if row is None:
                continue
            value += complex(np.conj(state[row])) * matrix_element * ket_amp
    return value


def _expectation_multi_site_bitwise(
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    site_ops: List[Tuple[int, str]],
    *,
    amplitude_tol: float = 1e-15,
) -> complex:
    value = 0.0j
    for col, (spin_state_raw, orbital_state_raw) in enumerate(basis_list):
        ket_amp = complex(state[col])
        if abs(ket_amp) <= amplitude_tol:
            continue
        actions: List[Tuple[int, int, complex]] = [
            (int(spin_state_raw), int(orbital_state_raw), 1.0 + 0.0j)
        ]
        for site, op_name in site_ops:
            next_actions: List[Tuple[int, int, complex]] = []
            for spin_state, orbital_state, coeff in actions:
                for next_spin, next_orbital, matrix_element in _apply_one_site_bitwise_operator(
                    spin_state,
                    orbital_state,
                    int(site),
                    str(op_name),
                ):
                    next_actions.append(
                        (int(next_spin), int(next_orbital), complex(coeff * matrix_element))
                    )
            actions = next_actions
        for next_spin, next_orbital, matrix_element in actions:
            row = basis_map.get((int(next_spin), int(next_orbital)))
            if row is None:
                continue
            value += complex(np.conj(state[row])) * matrix_element * ket_amp
    return value


def collect_uniform_z_observables_from_sz_conserved_ed_state(
    geometry: GeometryData,
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
) -> Dict[str, float]:
    n_sites = int(geometry.number_of_sites)
    spin_z = 0.0
    orbital_z = 0.0
    probabilities = np.asarray(np.abs(state) ** 2, dtype=float)
    for probability, (spin_state_raw, orbital_state_raw) in zip(probabilities, basis_list):
        spin_state = int(spin_state_raw)
        orbital_state = int(orbital_state_raw)
        for site in range(n_sites):
            spin_z += float(probability) * _z_value_from_bit(spin_state, site)
            orbital_z += float(probability) * _z_value_from_bit(orbital_state, site)
    return {
        "spin_z_per_site": float(spin_z / float(max(1, n_sites))),
        "orbital_z_per_site": float(orbital_z / float(max(1, n_sites))),
    }


def plaquette_flux_from_sz_conserved_ed_state(
    geometry: GeometryData,
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    plaquette_center_idx: int | None = None,
) -> Dict[str, Any]:
    """Evaluate normalized honeycomb plaquette flux in the reduced Sz basis."""
    plaquettes = honeycomb_plaquette_flux_operators(geometry)
    if len(plaquettes) == 0:
        raise ValueError("No honeycomb length-six plaquette was found in this geometry.")
    selected = select_honeycomb_plaquette_flux_operator(geometry, plaquette_center_idx)
    selected_index = int(selected["plaquette_index"])
    flux_map: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}
    for plaquette in plaquettes:
        site_ops = [
            (int(site), str(operator_name))
            for site, operator_name in zip(plaquette["sites"], plaquette["operator_names"])
        ]
        raw_value = _expectation_multi_site_bitwise(state, basis_list, basis_map, site_ops)
        normalized_value = float(np.real(raw_value) * float(plaquette["normalization"]))
        plaquette_index = int(plaquette["plaquette_index"])
        flux_map[plaquette_index] = normalized_value
        details[plaquette_index] = {
            "plaquette_index": plaquette_index,
            "sites": [int(site) for site in plaquette["sites"]],
            "axes": [str(axis) for axis in plaquette["axes"]],
            "operators": [str(op) for op in plaquette["operator_names"]],
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


def compute_all_plaquette_fluxes_sz_conserved(
    geometry: GeometryData,
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
) -> Dict[int, float]:
    """Return normalized ``W_p`` on every valid plaquette in the reduced Sz basis."""
    flux_payload = plaquette_flux_from_sz_conserved_ed_state(
        geometry,
        state,
        basis_list,
        basis_map,
        plaquette_center_idx=None,
    )
    flux_map = flux_payload.get("all_plaquette_fluxes", flux_payload.get("plaquette_flux_map", {}))
    if not isinstance(flux_map, dict):
        return {}
    return {int(index): float(value) for index, value in flux_map.items()}


def collect_correlation_matrices_from_sz_conserved_ed(
    geometry: GeometryData,
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Collect two-site correlations directly in the reduced bitwise basis."""
    n_sites = int(geometry.number_of_sites)
    spin_pairs = [(f"S{axis}", f"S{axis}") for axis in ("x", "y", "z")]
    orbital_pairs = [(f"T{axis}", f"T{axis}") for axis in ("x", "y", "z")]
    same_axis_mixed_pairs = [(f"ST{axis}", f"ST{axis}") for axis in ("x", "y", "z")]
    cross_mixed_pairs = [
        (f"S{spin_axis}T{orbital_axis}", f"S{spin_axis}T{orbital_axis}")
        for orbital_axis in ("x", "y", "z")
        for spin_axis in ("x", "y", "z")
    ]
    op_pairs = spin_pairs + orbital_pairs + same_axis_mixed_pairs + cross_mixed_pairs
    seen: set[Tuple[str, str]] = set()
    unique_op_pairs: List[Tuple[str, str]] = []
    for pair in op_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique_op_pairs.append(pair)

    correlations = {
        f"{op1}_{op2}": np.zeros((n_sites, n_sites), dtype=np.complex128)
        for op1, op2 in unique_op_pairs
    }
    pair_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=(n_sites * (n_sites - 1)) * len(unique_op_pairs),
        desc="Sz-ED correlations",
        unit="pair",
        leave=False,
    )
    row_progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=n_sites,
        desc="Sz-ED corr rows",
        unit="row",
        leave=False,
    )
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            for op1, op2 in unique_op_pairs:
                correlations[f"{op1}_{op2}"][i, j] = _expectation_two_site_bitwise(
                    state,
                    basis_list,
                    basis_map,
                    i,
                    op1,
                    j,
                    op2,
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


def build_sz_conserved_scalar_correlations(correlations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build scalar S, T, and mixed ST correlations from reduced-basis data."""
    scalar = {
        "S": np.zeros_like(correlations["Sx_Sx"]),
        "T": np.zeros_like(correlations["Tx_Tx"]),
        "ST": np.zeros_like(correlations["SxTx_SxTx"]),
    }
    for axis in ("x", "y", "z"):
        scalar["S"] = scalar["S"] + correlations[f"S{axis}_S{axis}"]
        scalar["T"] = scalar["T"] + correlations[f"T{axis}_T{axis}"]
    for orbital_axis in ("x", "y", "z"):
        for spin_axis in ("x", "y", "z"):
            scalar["ST"] = scalar["ST"] + correlations[f"S{spin_axis}T{orbital_axis}_S{spin_axis}T{orbital_axis}"]
    return scalar


def all_bond_energies_sz_conserved(
    geometry: GeometryData,
    correlations: Dict[str, np.ndarray],
    alpha: float,
    beta: float,
    coupling_j: float,
    show_progress: bool = False,
    progress_desc: str = "Sz-ED bond energies",
) -> List[Dict[str, Any]]:
    """Bond-energy rows matching ``BITWISE_ED_FORMULA``."""
    rows: List[Dict[str, Any]] = []
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(geometry.bond_list),
        desc=progress_desc,
        unit="bond",
        leave=False,
    )
    spin_coeff = float(coupling_j) * (1.0 + float(beta))
    orbital_coeff = float(coupling_j) * (1.0 - float(beta))
    mixed_coeff = float(coupling_j) * float(alpha)
    for bond in geometry.bond_list:
        i = int(bond.i)
        j = int(bond.j)
        gamma = str(bond.gamma).lower()
        spin_dot = sum(complex(correlations[f"S{axis}_S{axis}"][i, j]) for axis in ("x", "y", "z"))
        orbital_gamma = complex(correlations[f"T{gamma}_T{gamma}"][i, j])
        mixed_gamma = sum(
            complex(correlations[f"S{spin_axis}T{gamma}_S{spin_axis}T{gamma}"][i, j])
            for spin_axis in ("x", "y", "z")
        )
        components = [
            {
                "channel": "S",
                "operator": "Sdot",
                "axis": "dot",
                "coefficient": float(spin_coeff),
                "correlation": float(np.real(spin_dot)),
                "energy": float(np.real(spin_coeff * spin_dot)),
            },
            {
                "channel": "T",
                "operator": f"T{gamma}",
                "axis": gamma,
                "coefficient": float(orbital_coeff),
                "correlation": float(np.real(orbital_gamma)),
                "energy": float(np.real(orbital_coeff * orbital_gamma)),
            },
            {
                "channel": "ST",
                "operator": f"SdotT{gamma}",
                "axis": gamma,
                "coefficient": float(mixed_coeff),
                "correlation": float(np.real(mixed_gamma)),
                "energy": float(np.real(mixed_coeff * mixed_gamma)),
            },
        ]
        channel_energies = {
            str(component["channel"]): float(component["energy"])
            for component in components
        }
        rows.append(
            {
                "i": i,
                "j": j,
                "gamma": gamma,
                "O_ij_gamma": float(sum(float(component["energy"]) for component in components)),
                "components": components,
                "channel_energies": channel_energies,
                "formula": BITWISE_ED_FORMULA,
            }
        )
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return rows


