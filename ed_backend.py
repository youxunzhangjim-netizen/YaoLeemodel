#!/usr/bin/env python3
"""Exact-diagonalization backends for the Yao-Lee driver.

This module owns both full-Hilbert exact diagonalization and the strict
bitwise total-Sz=0 spin/orbital sparse ED path.
"""

from __future__ import annotations

import functools
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from scipy.sparse.linalg import ArpackNoConvergence

from analysis import _end_stage, _make_progress_bar, _start_stage, profile_stage, resolve_low_energy_spectrum
from models import (
    GeometryData,
    ISING_AXIS,
    MODEL_FAMILY,
    ORBITAL_REP,
    SPIN_REP,
    ModelSpec,
    all_high_symmetry_structure_factors,
    analyze_hamiltonian_symmetries,
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

# Hard caps for the in-repo standard_projector ED path.  These protect long
# phase scans from accidentally materializing large projectors or C3 sectors as
# dense arrays.  Single selected-point runs can override them explicitly.
MAX_PROJECTOR_PARENT_DIM = 2_000_000
MAX_PROJECTOR_NNZ = 50_000_000
MAX_DENSE_PROJECTOR_ENTRIES = 8_000_000
MAX_DENSE_PROJECTOR_MB = 512.0
MAX_EXPLICIT_C3_PARENT_DIM = 2_500
MAX_EXPLICIT_C3_DIM = 2_500
MAX_PHASE_SCAN_C3_SECONDS_PER_POINT = 30.0


def _dense_memory_estimate_mb(entries: int, dtype: Any = np.complex128) -> float:
    return float(int(entries) * np.dtype(dtype).itemsize / (1024.0 ** 2))


def _dense_allocation_diagnostics(
    *,
    label: str,
    entries: int,
    dtype: Any = np.complex128,
    max_dense_entries: int = MAX_DENSE_PROJECTOR_ENTRIES,
    max_dense_mb: float = MAX_DENSE_PROJECTOR_MB,
) -> Dict[str, Any]:
    entry_count = int(entries)
    dtype_obj = np.dtype(dtype)
    estimate_mb = _dense_memory_estimate_mb(entry_count, dtype_obj)
    entries_ok = entry_count <= int(max_dense_entries)
    mb_ok = estimate_mb <= float(max_dense_mb)
    reason = None
    if not entries_ok:
        reason = (
            f"{label} would allocate {entry_count:,} dense entries, exceeding "
            f"MAX_DENSE_PROJECTOR_ENTRIES={int(max_dense_entries):,}."
        )
    elif not mb_ok:
        reason = (
            f"{label} would allocate {estimate_mb:.3f} MiB as {dtype_obj}, exceeding "
            f"MAX_DENSE_PROJECTOR_MB={float(max_dense_mb):.3f}."
        )
    return {
        "label": str(label),
        "entries": entry_count,
        "dtype": str(dtype_obj),
        "dtype_itemsize": int(dtype_obj.itemsize),
        "memory_estimate_MB": float(estimate_mb),
        "max_dense_entries": int(max_dense_entries),
        "max_dense_projector_MB": float(max_dense_mb),
        "allowed": bool(entries_ok and mb_ok),
        "reason": reason,
    }


def _raise_dense_memory_error(diagnostics: Dict[str, Any]) -> None:
    reason = diagnostics.get("reason") or "Dense projector allocation exceeds configured memory caps."
    raise MemoryError(f"[projector-ed] {reason}")


def _max_recorded_memory_estimate_mb(payload: Any) -> float | None:
    values: List[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("memory_estimate_MB") is not None:
                try:
                    values.append(float(item["memory_estimate_MB"]))
                except Exception:
                    pass
            for value in item.values():
                visit(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                visit(value)

    visit(payload)
    return float(max(values)) if values else None

def kron_all(op_list: List[sparse.spmatrix]) -> sparse.spmatrix:
    out = op_list[0]
    for op in op_list[1:]:
        out = sparse.kron(out, op, format="csr")
    return out


@functools.lru_cache(maxsize=32)
def _global_operator_cache_for_model_cached(model_spec: ModelSpec) -> Tuple[Tuple[str, sparse.spmatrix], ...]:
    ops = build_site_ops(model_spec)
    return tuple((name, sparse.csr_matrix(mat)) for name, mat in ops.items())


def build_global_operator_cache() -> Dict[str, sparse.spmatrix]:
    default_spec = build_model_spec(
        spin_rep=SPIN_REP,
        orbital_rep=ORBITAL_REP,
        model_family=MODEL_FAMILY,
        ising_axis=ISING_AXIS,
    )
    return build_global_operator_cache_for_model(default_spec)


def build_global_operator_cache_for_model(model_spec: ModelSpec) -> Dict[str, sparse.spmatrix]:
    return {name: mat.copy() for name, mat in _global_operator_cache_for_model_cached(model_spec)}


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
    hamiltonian_csr = hamiltonian if sparse.isspmatrix_csr(hamiltonian) else hamiltonian.tocsr()
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
    with profile_stage("ED Hamiltonian construction"):
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
    with profile_stage("diagonalization"):
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
        spin_gamma = complex(correlations[f"S{axis}_S{axis}"][i, j])
        orbital_dot = sum(complex(correlations[f"T{orbital_axis}_T{orbital_axis}"][i, j]) for orbital_axis in ("x", "y", "z"))
        mixed_dot_dot = sum(
            complex(correlations[f"S{spin_axis}T{orbital_axis}_S{spin_axis}T{orbital_axis}"][i, j])
            for spin_axis in ("x", "y", "z")
            for orbital_axis in ("x", "y", "z")
        )
        mixed_gamma_dot = sum(
            complex(correlations[f"S{axis}T{orbital_axis}_S{axis}T{orbital_axis}"][i, j])
            for orbital_axis in ("x", "y", "z")
        )
        physical_components = [
            ("ST", "SdotTdot", "dot", -float(coupling_j) * float(alpha), mixed_dot_dot),
            ("S", "Sdot", "dot", float(coupling_j) * float(alpha) * float(beta), spin_dot),
            ("ST", f"S{axis}Tdot", axis, 2.0 * float(coupling_j), mixed_gamma_dot),
            ("S", f"S{axis}", axis, -2.0 * float(coupling_j) * float(beta), spin_gamma),
            ("T", "Tdot", "dot", float(coupling_j) * float(beta), orbital_dot),
            ("constant", "Id", "identity", -float(coupling_j) * float(beta) * float(beta), 1.0 + 0.0j),
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
    "Disabled for yao_lee Eq. 7 because S_i^gamma S_j^gamma x/y terms do not conserve total Sz."
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


def estimate_spin_orbital_u1_dimension(
    n_sites: int,
    *,
    use_sz_block: bool = False,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
) -> int:
    """Return the spin-orbital bit-basis dimension after Sz/Tz U(1) filters."""
    n = int(n_sites)
    if n < 0:
        raise ValueError("n_sites must be non-negative.")
    spin_dim = 1 << n
    orbital_dim = 1 << n
    if bool(use_sz_block):
        nup_spin = _spin_half_nup_from_total_sz2(n, int(target_sz2))
        spin_dim = 0 if nup_spin < 0 else int(math.comb(n, nup_spin))
    if bool(use_tau_z_block):
        nup_orbital = _spin_half_nup_from_total_sz2(n, int(target_tz2))
        orbital_dim = 0 if nup_orbital < 0 else int(math.comb(n, nup_orbital))
    return int(spin_dim * orbital_dim)


def build_spin_orbital_u1_basis(
    N: int,
    *,
    use_sz_block: bool = False,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Build a bit basis with optional total-Sz and total-Tz spin-1/2 sectors."""
    basis_tuple, basis_map = _build_spin_orbital_u1_basis_cached(
        int(N),
        bool(use_sz_block),
        int(target_sz2),
        bool(use_tau_z_block),
        int(target_tz2),
    )
    return list(basis_tuple), dict(basis_map)


@functools.lru_cache(maxsize=64)
def _build_spin_orbital_u1_basis_cached(
    N: int,
    use_sz_block: bool,
    target_sz2: int,
    use_tau_z_block: bool,
    target_tz2: int,
) -> Tuple[Tuple[Tuple[int, int], ...], Dict[Tuple[int, int], int]]:
    """Cached immutable-ish basis payload for repeated scan points."""
    n_sites = int(N)
    if n_sites < 0:
        raise ValueError("N must be non-negative.")
    spin_limit = 1 << n_sites
    orbital_limit = 1 << n_sites
    if bool(use_sz_block):
        spin_nup = _spin_half_nup_from_total_sz2(n_sites, int(target_sz2))
        if spin_nup < 0:
            raise ValueError(f"Total 2*Sz={int(target_sz2)} is unreachable for {n_sites} spin-1/2 sites.")
        spin_basis = [state for state in range(spin_limit) if int(state).bit_count() == spin_nup]
    else:
        spin_basis = list(range(spin_limit))
    if bool(use_tau_z_block):
        orbital_nup = _spin_half_nup_from_total_sz2(n_sites, int(target_tz2))
        if orbital_nup < 0:
            raise ValueError(f"Total 2*Tz={int(target_tz2)} is unreachable for {n_sites} orbital-1/2 sites.")
        orbital_basis = [state for state in range(orbital_limit) if int(state).bit_count() == orbital_nup]
    else:
        orbital_basis = list(range(orbital_limit))

    basis_list: List[Tuple[int, int]] = []
    basis_map: Dict[Tuple[int, int], int] = {}
    for spin_state in spin_basis:
        for orbital_state in orbital_basis:
            key = (int(spin_state), int(orbital_state))
            basis_map[key] = len(basis_list)
            basis_list.append(key)
    return tuple(basis_list), basis_map


def _validate_ed_u1_block_request(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    *,
    jx: float,
    jy: float,
    jz: float,
    external_field_terms: List[Tuple[float, str]] | None,
    use_sz_block: bool,
    target_sz2: int,
    use_tau_z_block: bool,
    target_tz2: int,
) -> Dict[str, Any] | None:
    if bool(use_sz_block) and bool(use_tau_z_block):
        requested_mode = "u1"
    elif bool(use_sz_block):
        requested_mode = "u1_sz"
    elif bool(use_tau_z_block):
        requested_mode = "u1_tz"
    else:
        return None

    report = analyze_hamiltonian_symmetries(
        geometry=geometry,
        model_spec=model_spec,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
        external_field_terms=list(external_field_terms or []),
        requested_symmetry_mode=requested_mode,
        u1_target_total_sz2=int(target_sz2),
        u1_target_total_tz2=int(target_tz2),
    )
    mode_report = report.get(requested_mode, {})
    target_sector = mode_report.get("target_sector", {}) if isinstance(mode_report, dict) else {}
    conserved = bool(isinstance(mode_report, dict) and mode_report.get("conserved", False))
    reachable = bool(isinstance(target_sector, dict) and target_sector.get("reachable", False))
    if conserved and reachable:
        return report

    first_issue = None
    if isinstance(mode_report, dict):
        issues = mode_report.get("issues", [])
        if isinstance(issues, list) and issues:
            first_issue = issues[0]
    raise ValueError(
        f"Requested ED symmetry block {requested_mode} is not valid for "
        f"model_family={model_spec.model_family}: conserved={conserved}, "
        f"target_reachable={reachable}. First validator issue: {first_issue}"
    )


def _bit_is_up(state: int, site: int) -> bool:
    return bool((int(state) >> int(site)) & 1)


def _z_value_from_bit(state: int, site: int) -> float:
    return 0.5 if _bit_is_up(state, site) else -0.5


def _local_index_from_spin_orbital_bits(spin_up: bool, orbital_up: bool) -> int:
    # build_site_ops uses spin/orbital matrix order |up>, |down>; bits use 1=up, 0=down.
    spin_index = 0 if bool(spin_up) else 1
    orbital_index = 0 if bool(orbital_up) else 1
    return int(2 * spin_index + orbital_index)


def _spin_orbital_bits_from_local_index(local_index: int) -> Tuple[bool, bool]:
    spin_index = int(local_index) // 2
    orbital_index = int(local_index) % 2
    return bool(spin_index == 0), bool(orbital_index == 0)


def _set_bit_value(state: int, site: int, is_up: bool) -> int:
    mask = 1 << int(site)
    return int(state | mask) if bool(is_up) else int(state & ~mask)


def _apply_local_matrix_to_bit_state(
    spin_state: int,
    orbital_state: int,
    site: int,
    operator: sparse.spmatrix,
) -> List[Tuple[int, int, complex]]:
    matrix = operator.toarray() if sparse.issparse(operator) else np.asarray(operator)
    spin_up = _bit_is_up(spin_state, site)
    orbital_up = _bit_is_up(orbital_state, site)
    col = _local_index_from_spin_orbital_bits(spin_up, orbital_up)
    out: List[Tuple[int, int, complex]] = []
    for row in range(int(matrix.shape[0])):
        coeff = complex(matrix[row, col])
        if abs(coeff) <= 1.0e-14:
            continue
        next_spin_up, next_orbital_up = _spin_orbital_bits_from_local_index(row)
        next_spin = _set_bit_value(spin_state, site, next_spin_up)
        next_orbital = _set_bit_value(orbital_state, site, next_orbital_up)
        out.append((int(next_spin), int(next_orbital), coeff))
    return out


def _apply_two_local_matrices_to_bit_state(
    spin_state: int,
    orbital_state: int,
    i: int,
    op_i: sparse.spmatrix,
    j: int,
    op_j: sparse.spmatrix,
) -> List[Tuple[int, int, complex]]:
    first_actions = _apply_local_matrix_to_bit_state(spin_state, orbital_state, i, op_i)
    out: List[Tuple[int, int, complex]] = []
    for spin_mid, orbital_mid, coeff_i in first_actions:
        for spin_out, orbital_out, coeff_j in _apply_local_matrix_to_bit_state(
            spin_mid,
            orbital_mid,
            j,
            op_j,
        ):
            out.append((int(spin_out), int(orbital_out), complex(coeff_i * coeff_j)))
    return out


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


def build_sparse_hamiltonian_spin_orbital_u1(
    N: int,
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    *,
    coupling_j: float = 1.0,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
) -> sparse.csr_matrix:
    """Build a sparse ED Hamiltonian in the supplied spin/orbital U(1) bit basis."""
    n_sites = int(N)
    if n_sites != int(geometry.number_of_sites):
        raise ValueError(
            f"N={n_sites} does not match geometry.number_of_sites={int(geometry.number_of_sites)}."
        )
    if model_spec.spin_rep != "1/2" or model_spec.orbital_rep != "1/2":
        raise ValueError("spin_orbital_u1 sparse ED currently supports spin_rep=1/2 and orbital_rep=1/2.")
    op_cache = build_global_operator_cache_for_model(model_spec)
    dim = int(len(basis_list))
    hamiltonian = sparse.lil_matrix((dim, dim), dtype=np.complex128)
    field_terms = list(external_field_terms or [])
    total_columns = int(dim)
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=total_columns,
        desc="U1-ED H columns",
        unit="state",
        leave=False,
    )
    bond_terms: List[Tuple[int, int, List[Tuple[complex, str, str]]]] = []
    for bond in geometry.bond_list:
        bond_terms.append(
            (
                int(bond.i),
                int(bond.j),
                list(
                    two_site_operator_terms_for_bond(
                        str(bond.gamma).lower(),
                        model_spec,
                        alpha,
                        beta,
                        coupling_j,
                        jx=jx,
                        jy=jy,
                        jz=jz,
                    )
                ),
            )
        )
    for col, (spin_state_raw, orbital_state_raw) in enumerate(basis_list):
        spin_state = int(spin_state_raw)
        orbital_state = int(orbital_state_raw)
        for i, j, terms in bond_terms:
            for coeff, op_i_name, op_j_name in terms:
                op_i = op_cache[str(op_i_name)]
                op_j = op_cache[str(op_j_name)]
                for next_spin, next_orbital, matrix_element in _apply_two_local_matrices_to_bit_state(
                    spin_state,
                    orbital_state,
                    i,
                    op_i,
                    j,
                    op_j,
                ):
                    row = basis_map.get((int(next_spin), int(next_orbital)))
                    if row is not None:
                        hamiltonian[row, col] += complex(coeff) * matrix_element
        for site in range(n_sites):
            for coefficient, op_name in field_terms:
                for next_spin, next_orbital, matrix_element in _apply_local_matrix_to_bit_state(
                    spin_state,
                    orbital_state,
                    site,
                    op_cache[str(op_name)],
                ):
                    row = basis_map.get((int(next_spin), int(next_orbital)))
                    if row is not None:
                        hamiltonian[row, col] += float(coefficient) * matrix_element
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return hamiltonian.tocsr()


def run_spin_orbital_u1_exact_spectrum(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    *,
    coupling_j: float = 1.0,
    eigenstate_count: int = 3,
    check_ground_state_degeneracy: bool = True,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
    ground_manifold_abs_tol: float = 1e-12,
    ground_manifold_rel_tol: float = 1e-12,
    sparse_tol: float = 0.0,
    sparse_maxiter: int | None = None,
    use_sz_block: bool = False,
    target_sz2: int = 0,
    use_tau_z_block: bool = True,
    target_tz2: int = 0,
) -> Tuple[Dict[str, Any], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Diagonalize a sparse spin/orbital U(1)-restricted ED Hamiltonian."""
    n_sites = int(geometry.number_of_sites)
    symmetry_validation_report = _validate_ed_u1_block_request(
        geometry,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
        external_field_terms=external_field_terms,
        use_sz_block=bool(use_sz_block),
        target_sz2=int(target_sz2),
        use_tau_z_block=bool(use_tau_z_block),
        target_tz2=int(target_tz2),
    )
    stage_start = _start_stage("spin-orbital U1 ED", show_progress)
    with profile_stage("ED basis construction"):
        basis_list, basis_map = build_spin_orbital_u1_basis(
            n_sites,
            use_sz_block=bool(use_sz_block),
            target_sz2=int(target_sz2),
            use_tau_z_block=bool(use_tau_z_block),
            target_tz2=int(target_tz2),
        )
    with profile_stage("ED Hamiltonian construction"):
        hamiltonian = build_sparse_hamiltonian_spin_orbital_u1(
            n_sites,
            geometry,
            model_spec,
            alpha,
            beta,
            basis_list,
            basis_map,
            coupling_j=coupling_j,
            jx=jx,
            jy=jy,
            jz=jz,
            external_field_terms=external_field_terms,
            show_progress=show_progress,
        )
    dim = int(hamiltonian.shape[0])
    if dim <= 0:
        raise ValueError("Empty spin-orbital U1 ED basis.")
    requested_count = max(1, int(eigenstate_count))
    with profile_stage("diagonalization"):
        if dim == 1:
            eigenvalues = np.asarray([complex(hamiltonian[0, 0])], dtype=np.complex128)
            eigenvectors = np.ones((1, 1), dtype=np.complex128)
            solver_mode = "u1_reduced_dense_dim1"
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
                    "[u1-ed] basis ready: "
                    f"dim={dim}, nnz={hamiltonian.nnz}, k={solve_count}, "
                    f"Sz_block={bool(use_sz_block)}, Tz_block={bool(use_tau_z_block)}"
                )
            eigenvalues, eigenvectors, eigsh_info = _run_lowest_eigsh(
                hamiltonian,
                eigenstate_count=solve_count,
                sparse_tol=sparse_tol,
                sparse_maxiter=sparse_maxiter,
                show_progress=show_progress,
                label="u1-ed",
            )
            solver_mode = "spin_orbital_u1_sparse"
    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.asarray(np.real(eigenvalues[order]), dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    _end_stage("spin-orbital U1 ED", stage_start, show_progress)
    low_energy_resolution = resolve_low_energy_spectrum(
        eigenvalues,
        check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
        hilbert_dim=dim,
        degeneracy_tolerance_abs=float(ground_manifold_abs_tol),
        degeneracy_tolerance_rel=float(ground_manifold_rel_tol),
    )
    spectrum: Dict[str, Any] = {
        "solver_mode": solver_mode,
        "solver_requested": "spin_orbital_u1_sparse",
        "basis_type": "bitwise_spin_orbital_u1_block",
        "use_sz_block": bool(use_sz_block),
        "target_sz2": int(target_sz2),
        "use_tau_z_block": bool(use_tau_z_block),
        "target_tz2": int(target_tz2),
        "symmetry_validation": symmetry_validation_report,
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
        "eigsh": eigsh_info if solver_mode == "spin_orbital_u1_sparse" else None,
        **low_energy_resolution,
    }
    return spectrum, eigenvectors, basis_list, basis_map


def _spin_pi_z_eigenvalue_from_spin_bits(spin_state: int) -> int:
    """Diagonal spin-pi-z parity used by the projected ED path.

    The convention requested for the Yao-Lee spin basis is
    Pz = (-1) ** N_up_spin, with spin bits using 1=up.
    """
    return 1 if (int(spin_state).bit_count() % 2 == 0) else -1


def build_spin_pi_z_operator_in_spin_orbital_u1_basis(
    basis_list: List[Tuple[int, int]],
) -> sparse.csr_matrix:
    """Return Pz=(-1)^Nup_spin inside the supplied spin-orbital U1 basis."""
    operator = _build_spin_pi_z_operator_cached(
        tuple((int(spin_state), int(orbital_state)) for spin_state, orbital_state in basis_list)
    )
    return operator.copy()


@functools.lru_cache(maxsize=64)
def _build_spin_pi_z_operator_cached(
    basis_list: Tuple[Tuple[int, int], ...],
) -> sparse.csr_matrix:
    dim = int(len(basis_list))
    diagonal = np.asarray(
        [_spin_pi_z_eigenvalue_from_spin_bits(spin_state) for spin_state, _ in basis_list],
        dtype=np.complex128,
    )
    return sparse.diags(diagonal, offsets=0, shape=(dim, dim), format="csr")


def _geometry_translation_cache_key(
    geometry: GeometryData,
    direction: str,
) -> Tuple[Any, ...]:
    axis = str(direction).strip().lower()
    if axis not in ("x", "y"):
        raise ValueError(f"Unsupported translation direction '{direction}'.")
    n_sites = int(geometry.number_of_sites)
    cell_indices = tuple(
        (int(cell[0]), int(cell[1]))
        for cell in list(getattr(geometry, "cell_indices", []))
    )
    sublattice_indices = tuple(int(value) for value in list(getattr(geometry, "sublattice_indices", [])))
    if len(cell_indices) != n_sites or len(sublattice_indices) != n_sites:
        raise ValueError("Geometry does not expose complete cell/sublattice labels for fused-site translation.")
    return (
        axis,
        n_sites,
        int(getattr(geometry, "length_x", 0) or 0),
        int(getattr(geometry, "length_y", 0) or 0),
        bool(getattr(geometry, "circumference_x", False)),
        bool(getattr(geometry, "circumference_y", False)),
        cell_indices,
        sublattice_indices,
    )


def _site_translation_permutation(geometry: GeometryData, direction: str) -> Tuple[List[int], int]:
    """Map each physical site to its translated site for a periodic cell shift."""
    permutation, order = _site_translation_permutation_cached(_geometry_translation_cache_key(geometry, direction))
    return list(permutation), int(order)


@functools.lru_cache(maxsize=64)
def _site_translation_permutation_cached(cache_key: Tuple[Any, ...]) -> Tuple[Tuple[int, ...], int]:
    axis = str(cache_key[0])
    n_sites = int(cache_key[1])
    length_x = int(cache_key[2])
    length_y = int(cache_key[3])
    circumference_x = bool(cache_key[4])
    circumference_y = bool(cache_key[5])
    cell_indices = tuple(cache_key[6])
    sublattice_indices = tuple(cache_key[7])

    if length_x <= 0:
        length_x = 1 + max(int(cell[0]) for cell in cell_indices)
    if length_y <= 0:
        length_y = 1 + max(int(cell[1]) for cell in cell_indices)
    length_x = int(length_x)
    length_y = int(length_y)
    if length_x <= 0 or length_y <= 0:
        raise ValueError("Invalid geometry lengths for translation projector.")
    if axis == "x" and not circumference_x:
        raise ValueError("Translation-x projector requires periodic x boundary conditions.")
    if axis == "y" and not circumference_y:
        raise ValueError("Translation-y projector requires periodic y boundary conditions.")

    site_by_label: Dict[Tuple[int, int, int], int] = {}
    for site, (cell, sublattice) in enumerate(zip(cell_indices, sublattice_indices)):
        key = (int(cell[0]), int(cell[1]), int(sublattice))
        if key in site_by_label:
            raise ValueError(f"Duplicate fused site label {key}; cannot build translation projector.")
        site_by_label[key] = int(site)

    permutation = [0] * n_sites
    for site, (cell, sublattice) in enumerate(zip(cell_indices, sublattice_indices)):
        x_cell = int(cell[0])
        y_cell = int(cell[1])
        sub = int(sublattice)
        if axis == "x":
            target_key = ((x_cell + 1) % length_x, y_cell, sub)
            order = length_x
        else:
            target_key = (x_cell, (y_cell + 1) % length_y, sub)
            order = length_y
        if target_key not in site_by_label:
            raise ValueError(f"Translated fused site {target_key} is missing from the geometry.")
        permutation[int(site)] = int(site_by_label[target_key])
    return tuple(permutation), int(order)


def _permute_bits_by_site_map(state: int, site_permutation: List[int]) -> int:
    """Apply a physical-site permutation to a bitstring.

    ``site_permutation[old_site] = new_site``. The translated bit at
    ``new_site`` is copied from ``old_site``.
    """
    out = 0
    raw_state = int(state)
    for old_site, new_site in enumerate(site_permutation):
        if (raw_state >> int(old_site)) & 1:
            out |= 1 << int(new_site)
    return int(out)


def _translation_index_permutation_in_spin_orbital_u1_basis(
    geometry: GeometryData,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    direction: str,
) -> Tuple[np.ndarray, int]:
    site_permutation, order = _site_translation_permutation(geometry, direction)
    index_permutation = np.empty(int(len(basis_list)), dtype=np.int64)
    for col, (spin_state, orbital_state) in enumerate(basis_list):
        translated_key = (
            _permute_bits_by_site_map(int(spin_state), site_permutation),
            _permute_bits_by_site_map(int(orbital_state), site_permutation),
        )
        row = basis_map.get(translated_key)
        if row is None:
            raise ValueError(
                "Fused translation left the selected Tz basis. "
                "This indicates an inconsistent orbital Tz sector or geometry map."
            )
        index_permutation[int(col)] = int(row)
    return index_permutation, int(order)


def build_fused_translation_operator_in_spin_orbital_u1_basis(
    geometry: GeometryData,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    direction: str,
) -> Tuple[sparse.csr_matrix, int, np.ndarray]:
    """Return a combined spin+orbital translation operator in the Tz basis."""
    if any(basis_map.get((int(spin_state), int(orbital_state))) != idx for idx, (spin_state, orbital_state) in enumerate(basis_list)):
        index_permutation, order = _translation_index_permutation_in_spin_orbital_u1_basis(
            geometry,
            basis_list,
            basis_map,
            direction,
        )
        dim = int(len(basis_list))
        columns = np.arange(dim, dtype=np.int64)
        data = np.ones(dim, dtype=np.complex128)
        operator = sparse.csr_matrix((data, (index_permutation, columns)), shape=(dim, dim))
        return operator, int(order), index_permutation
    operator, order, index_permutation = _build_fused_translation_operator_cached(
        _geometry_translation_cache_key(geometry, direction),
        tuple((int(spin_state), int(orbital_state)) for spin_state, orbital_state in basis_list),
    )
    return operator.copy(), int(order), index_permutation.copy()


@functools.lru_cache(maxsize=64)
def _build_fused_translation_operator_cached(
    geometry_key: Tuple[Any, ...],
    basis_list: Tuple[Tuple[int, int], ...],
) -> Tuple[sparse.csr_matrix, int, np.ndarray]:
    site_permutation, order = _site_translation_permutation_cached(geometry_key)
    basis_map = {key: index for index, key in enumerate(basis_list)}
    dim = int(len(basis_list))
    index_permutation = np.empty(dim, dtype=np.int64)
    for col, (spin_state, orbital_state) in enumerate(basis_list):
        translated_key = (
            _permute_bits_by_site_map(int(spin_state), site_permutation),
            _permute_bits_by_site_map(int(orbital_state), site_permutation),
        )
        row = basis_map.get(translated_key)
        if row is None:
            raise ValueError(
                "Fused translation left the selected Tz basis. "
                "This indicates an inconsistent orbital Tz sector or geometry map."
            )
        index_permutation[int(col)] = int(row)
    columns = np.arange(dim, dtype=np.int64)
    data = np.ones(dim, dtype=np.complex128)
    operator = sparse.csr_matrix((data, (index_permutation, columns)), shape=(dim, dim))
    return operator, int(order), index_permutation


def _sparse_relative_commutator_norm(left: sparse.spmatrix, right: sparse.spmatrix) -> float:
    commutator = (left @ right) - (right @ left)
    numerator = float(sparse_linalg.norm(commutator))
    denominator = max(1.0, float(sparse_linalg.norm(left)) * float(sparse_linalg.norm(right)))
    return float(numerator / denominator)


def _projected_column_space_from_orbits(
    basis_list: List[Tuple[int, int]],
    *,
    use_spin_pi_z: bool,
    z2_target_parity: int,
    translation_powers: List[Tuple[int, int, np.ndarray, complex]],
    projector_tol: float,
) -> Tuple[sparse.csc_matrix, Dict[str, Any]]:
    """Build candidate projector columns from parity-filtered translation orbits."""
    translation_key = _translation_powers_cache_key(translation_powers)
    q_candidates, metadata = _projected_column_space_from_orbits_cached(
        tuple((int(spin_state), int(orbital_state)) for spin_state, orbital_state in basis_list),
        bool(use_spin_pi_z),
        int(z2_target_parity),
        translation_key,
        float(projector_tol),
    )
    return q_candidates.copy(), dict(metadata)


def _translation_powers_cache_key(
    translation_powers: List[Tuple[int, int, np.ndarray, complex]],
) -> Tuple[Tuple[int, int, Tuple[int, ...], float, float], ...]:
    return tuple(
        (
            int(nx),
            int(ny),
            tuple(int(value) for value in np.asarray(image, dtype=np.int64).tolist()),
            float(np.real(phase)),
            float(np.imag(phase)),
        )
        for nx, ny, image, phase in translation_powers
    )


@functools.lru_cache(maxsize=64)
def _projected_column_space_from_orbits_cached(
    basis_list: Tuple[Tuple[int, int], ...],
    use_spin_pi_z: bool,
    z2_target_parity: int,
    translation_key: Tuple[Tuple[int, int, Tuple[int, ...], float, float], ...],
    projector_tol: float,
) -> Tuple[sparse.csc_matrix, Dict[str, Any]]:
    dim = int(len(basis_list))
    translation_powers = [
        (int(nx), int(ny), np.asarray(image, dtype=np.int64), complex(real_phase, imag_phase))
        for nx, ny, image, real_phase, imag_phase in translation_key
    ]
    target_pz = 1 if int(z2_target_parity) % 2 == 0 else -1
    parity_values = np.asarray(
        [_spin_pi_z_eigenvalue_from_spin_bits(spin_state) for spin_state, _ in basis_list],
        dtype=np.int8,
    )
    visited = np.zeros(dim, dtype=bool)
    rows: List[int] = []
    cols: List[int] = []
    data: List[complex] = []
    kept_columns = 0
    skipped_by_parity = 0
    skipped_null_orbits = 0
    orbit_count = 0

    for representative in range(dim):
        if bool(visited[representative]):
            continue
        orbit_indices = {int(image[representative]) for _, _, image, _ in translation_powers}
        for index in orbit_indices:
            visited[int(index)] = True
        orbit_count += 1
        if bool(use_spin_pi_z) and int(parity_values[representative]) != int(target_pz):
            skipped_by_parity += 1
            continue

        coefficients: Dict[int, complex] = {}
        for _, _, image, phase in translation_powers:
            row = int(image[representative])
            coefficients[row] = coefficients.get(row, 0.0 + 0.0j) + complex(phase)
        norm = math.sqrt(sum(abs(value) ** 2 for value in coefficients.values()))
        if norm <= float(projector_tol):
            skipped_null_orbits += 1
            continue
        for row, value in coefficients.items():
            amplitude = complex(value) / norm
            if abs(amplitude) > float(projector_tol):
                rows.append(int(row))
                cols.append(int(kept_columns))
                data.append(amplitude)
        kept_columns += 1

    q_candidates = sparse.csc_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(dim, kept_columns),
        dtype=np.complex128,
    )
    metadata = {
        "candidate_columns": int(kept_columns),
        "translation_orbits": int(orbit_count),
        "skipped_orbits_by_spin_pi_z": int(skipped_by_parity),
        "skipped_null_momentum_orbits": int(skipped_null_orbits),
        "z2_target_eigenvalue": int(target_pz) if bool(use_spin_pi_z) else None,
    }
    return q_candidates, metadata


def _copy_projector_matrix(matrix: sparse.spmatrix | np.ndarray) -> sparse.spmatrix | np.ndarray:
    return matrix.copy() if sparse.issparse(matrix) else np.array(matrix, copy=True)


def _projector_basis_from_orbits(
    basis_list: List[Tuple[int, int]],
    *,
    use_spin_pi_z: bool,
    z2_target_parity: int,
    translation_powers: List[Tuple[int, int, np.ndarray, complex]],
    projector_tol: float,
    dense_svd_entry_cap: int,
    dense_svd_mb_cap: float,
) -> Tuple[sparse.csc_matrix, sparse.spmatrix | np.ndarray, Dict[str, Any], Dict[str, Any]]:
    basis_key = tuple((int(spin_state), int(orbital_state)) for spin_state, orbital_state in basis_list)
    (
        q_candidates,
        q_matrix,
        projector_metadata,
        orthonormalization_metadata,
    ) = _projector_basis_from_orbits_cached(
        basis_key,
        bool(use_spin_pi_z),
        int(z2_target_parity),
        _translation_powers_cache_key(translation_powers),
        float(projector_tol),
        int(dense_svd_entry_cap),
        float(dense_svd_mb_cap),
    )
    return (
        q_candidates.copy(),
        _copy_projector_matrix(q_matrix),
        dict(projector_metadata),
        dict(orthonormalization_metadata),
    )


@functools.lru_cache(maxsize=64)
def _projector_basis_from_orbits_cached(
    basis_list: Tuple[Tuple[int, int], ...],
    use_spin_pi_z: bool,
    z2_target_parity: int,
    translation_key: Tuple[Tuple[int, int, Tuple[int, ...], float, float], ...],
    projector_tol: float,
    dense_svd_entry_cap: int,
    dense_svd_mb_cap: float,
) -> Tuple[sparse.csc_matrix, sparse.spmatrix | np.ndarray, Dict[str, Any], Dict[str, Any]]:
    q_candidates, projector_metadata = _projected_column_space_from_orbits_cached(
        basis_list,
        bool(use_spin_pi_z),
        int(z2_target_parity),
        translation_key,
        float(projector_tol),
    )
    q_matrix, orthonormalization_metadata = _orthonormalize_projector_columns(
        q_candidates,
        tol=float(projector_tol),
        dense_svd_entry_cap=int(dense_svd_entry_cap),
        dense_svd_mb_cap=float(dense_svd_mb_cap),
    )
    return q_candidates, q_matrix, projector_metadata, orthonormalization_metadata


def _matrix_storage_diagnostics(matrix: sparse.spmatrix | np.ndarray | None) -> Dict[str, Any]:
    if matrix is None:
        return {"available": False}
    if sparse.issparse(matrix):
        mat = matrix
        return {
            "available": True,
            "format": str(mat.getformat()),
            "shape": [int(mat.shape[0]), int(mat.shape[1])],
            "nnz": int(mat.nnz),
            "estimated_bytes": int(mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes),
            "dense_entries": int(mat.shape[0]) * int(mat.shape[1]),
        }
    arr = np.asarray(matrix)
    return {
        "available": True,
        "format": "dense",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "nnz": None,
        "estimated_bytes": int(arr.nbytes),
        "dense_entries": int(arr.size),
    }


def _check_projector_matrix_caps(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    label: str,
    max_nnz: int,
    max_dense_entries: int,
    max_dense_mb: float = MAX_DENSE_PROJECTOR_MB,
) -> None:
    diagnostics = _matrix_storage_diagnostics(matrix)
    dense_entries = int(diagnostics.get("dense_entries", 0))
    if sparse.issparse(matrix):
        nnz = int(diagnostics.get("nnz", 0))
        if nnz > int(max_nnz):
            raise MemoryError(
                f"[projector-ed] {label} has nnz={nnz:,}, exceeding MAX_PROJECTOR_NNZ={int(max_nnz):,}."
            )
        return
    dense_guard = _dense_allocation_diagnostics(
        label=label,
        entries=dense_entries,
        dtype=getattr(matrix, "dtype", np.complex128),
        max_dense_entries=int(max_dense_entries),
        max_dense_mb=float(max_dense_mb),
    )
    if not bool(dense_guard["allowed"]):
        _raise_dense_memory_error(dense_guard)


def _orthonormalize_projector_columns(
    q_candidates: sparse.spmatrix | np.ndarray,
    *,
    tol: float,
    dense_svd_entry_cap: int = MAX_DENSE_PROJECTOR_ENTRIES,
    dense_svd_mb_cap: float = MAX_DENSE_PROJECTOR_MB,
    force_svd: bool = False,
) -> Tuple[sparse.spmatrix | np.ndarray, Dict[str, Any]]:
    """Orthonormalize projector columns when small; otherwise keep exact orbit columns.

    Translation orbit columns built by ``_projected_column_space_from_orbits`` are
    already orthonormal after per-orbit normalization.  For small projected
    spaces we still run SVD as a direct numerical check and cleanup.
    """
    rows, columns = q_candidates.shape
    if int(columns) <= 0:
        raise ValueError("Requested projectors produced an empty ED subspace.")
    entry_count = int(rows) * int(columns)
    dense_guard = _dense_allocation_diagnostics(
        label="projector candidate SVD",
        entries=entry_count,
        dtype=getattr(q_candidates, "dtype", np.complex128),
        max_dense_entries=int(dense_svd_entry_cap),
        max_dense_mb=float(dense_svd_mb_cap),
    )
    if force_svd or bool(dense_guard["allowed"]):
        if force_svd and not bool(dense_guard["allowed"]):
            _raise_dense_memory_error(dense_guard)
        dense = q_candidates.toarray() if sparse.issparse(q_candidates) else np.asarray(q_candidates)
        u_matrix, singular_values, _ = np.linalg.svd(dense, full_matrices=False)
        max_singular = float(np.max(singular_values)) if singular_values.size else 0.0
        keep = singular_values > max(float(tol), float(tol) * max_singular)
        if not np.any(keep):
            raise ValueError("Projector SVD removed every candidate column.")
        q_matrix = np.asarray(u_matrix[:, keep], dtype=np.complex128)
        metadata = {
            "orthonormalization": "svd",
            "singular_values_kept": int(np.count_nonzero(keep)),
            "singular_values_total": int(singular_values.size),
            "smallest_kept_singular_value": float(np.min(singular_values[keep])),
            "largest_singular_value": max_singular,
            "dense_svd_entry_cap": int(dense_svd_entry_cap),
            "dense_svd_mb_cap": float(dense_svd_mb_cap),
            "candidate_dense_entries": int(entry_count),
            "memory_estimate_MB": float(dense_guard["memory_estimate_MB"]),
            "dense_memory_guard": dense_guard,
            "projector_strategy": "dense_small_safe",
            "candidate_storage": _matrix_storage_diagnostics(q_candidates),
        }
        return q_matrix, metadata

    metadata = {
        "orthonormalization": "orbit_normalization",
        "singular_values_kept": int(columns),
        "singular_values_total": None,
        "dense_svd_entry_cap": int(dense_svd_entry_cap),
        "dense_svd_mb_cap": float(dense_svd_mb_cap),
        "candidate_dense_entries": int(entry_count),
        "memory_estimate_MB": float(dense_guard["memory_estimate_MB"]),
        "dense_memory_guard": dense_guard,
        "projector_strategy": "sparse",
        "candidate_storage": _matrix_storage_diagnostics(q_candidates),
        "note": (
            "Projector columns are disjoint normalized translation-orbit columns; "
            "dense SVD was skipped to avoid materializing a very large matrix."
        ),
    }
    return q_candidates, metadata


def _honeycomb_bond_axis_lookup(geometry: GeometryData) -> Dict[Tuple[int, int], str]:
    lookup: Dict[Tuple[int, int], str] = {}
    for bond in geometry.bond_list:
        i = int(bond.i)
        j = int(bond.j)
        lookup[(min(i, j), max(i, j))] = str(bond.gamma).strip().lower()
    return lookup


def _permutation_has_order(site_permutation: List[int], order: int) -> bool:
    n_sites = int(len(site_permutation))
    image = list(range(n_sites))
    for _ in range(int(order)):
        image = [int(site_permutation[int(site)]) for site in image]
    return all(int(image[site]) == int(site) for site in range(n_sites))


def _matrix_mod_order_three(matrix: Tuple[Tuple[int, int], Tuple[int, int]], modulus: int) -> bool:
    m = np.asarray(matrix, dtype=int)
    identity = np.eye(2, dtype=int)
    m_mod = np.mod(m, int(modulus))
    if np.array_equal(m_mod, identity % int(modulus)):
        return False
    cube = np.mod(m_mod @ m_mod @ m_mod, int(modulus))
    return bool(np.array_equal(cube, identity % int(modulus)))


def _candidate_order_three_cell_matrices(length: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    values = (-1, 0, 1)
    candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for a in values:
        for b in values:
            for c in values:
                for d in values:
                    matrix = ((int(a), int(b)), (int(c), int(d)))
                    if _matrix_mod_order_three(matrix, int(length)):
                        if matrix not in candidates:
                            candidates.append(matrix)
    return candidates


def _find_honeycomb_combined_c3_site_permutation(
    geometry: GeometryData,
) -> Tuple[List[int], Dict[str, Any]]:
    """Find a 120-degree honeycomb site rotation compatible with the bond labels."""
    n_sites = int(geometry.number_of_sites)
    length_x = int(getattr(geometry, "length_x", 0) or 0)
    length_y = int(getattr(geometry, "length_y", 0) or 0)
    if length_x <= 0 or length_y <= 0 or length_x != length_y:
        raise ValueError("Combined C3 requires a honeycomb torus with length_x=length_y.")
    if not (bool(getattr(geometry, "circumference_x", False)) and bool(getattr(geometry, "circumference_y", False))):
        raise ValueError("Combined C3 requires periodic x and y boundaries.")
    cell_indices = list(getattr(geometry, "cell_indices", []))
    sublattice_indices = list(getattr(geometry, "sublattice_indices", []))
    if len(cell_indices) != n_sites or len(sublattice_indices) != n_sites:
        raise ValueError("Geometry does not expose complete cell/sublattice labels for combined C3.")

    length = int(length_x)
    site_by_label: Dict[Tuple[int, int, int], int] = {}
    for site, (cell, sublattice) in enumerate(zip(cell_indices, sublattice_indices)):
        key = (int(cell[0]) % length, int(cell[1]) % length, int(sublattice))
        if key in site_by_label:
            raise ValueError(f"Duplicate honeycomb site label {key}; cannot build C3.")
        site_by_label[key] = int(site)

    bond_lookup = _honeycomb_bond_axis_lookup(geometry)
    allowed_axis_cycles = (
        {"x": "y", "y": "z", "z": "x"},
        {"x": "z", "z": "y", "y": "x"},
    )
    matrices = _candidate_order_three_cell_matrices(length)
    offsets = [(x, y) for x in range(length) for y in range(length)]
    for matrix in matrices:
        m00, m01 = matrix[0]
        m10, m11 = matrix[1]
        for flip_sublattice in (False, True):
            for offset_a in offsets:
                for offset_b in offsets:
                    offsets_by_sub = {0: offset_a, 1: offset_b}
                    permutation = [0] * n_sites
                    ok = True
                    for site, (cell, sublattice) in enumerate(zip(cell_indices, sublattice_indices)):
                        x_cell = int(cell[0]) % length
                        y_cell = int(cell[1]) % length
                        sub = int(sublattice)
                        target_sub = 1 - sub if flip_sublattice else sub
                        offset = offsets_by_sub[sub]
                        target_x = (m00 * x_cell + m01 * y_cell + int(offset[0])) % length
                        target_y = (m10 * x_cell + m11 * y_cell + int(offset[1])) % length
                        target_key = (int(target_x), int(target_y), int(target_sub))
                        target_site = site_by_label.get(target_key)
                        if target_site is None:
                            ok = False
                            break
                        permutation[int(site)] = int(target_site)
                    if not ok or len(set(permutation)) != n_sites:
                        continue
                    if not _permutation_has_order(permutation, 3):
                        continue

                    axis_map: Dict[str, str] = {}
                    for bond in geometry.bond_list:
                        mapped_pair = (int(permutation[int(bond.i)]), int(permutation[int(bond.j)]))
                        mapped_axis = bond_lookup.get((min(mapped_pair), max(mapped_pair)))
                        if mapped_axis is None:
                            ok = False
                            break
                        source_axis = str(bond.gamma).strip().lower()
                        previous = axis_map.get(source_axis)
                        if previous is not None and previous != mapped_axis:
                            ok = False
                            break
                        axis_map[source_axis] = mapped_axis
                    if not ok:
                        continue
                    if axis_map not in allowed_axis_cycles:
                        continue
                    return permutation, {
                        "cell_matrix": [[int(m00), int(m01)], [int(m10), int(m11)]],
                        "flip_sublattice": bool(flip_sublattice),
                        "offset_by_sublattice": {
                            "0": [int(offset_a[0]), int(offset_a[1])],
                            "1": [int(offset_b[0]), int(offset_b[1])],
                        },
                        "bond_axis_cycle": dict(axis_map),
                        "order": 3,
                    }

    raise ValueError("No honeycomb order-three site rotation compatible with the x/y/z bond labels was found.")


def _spin_half_c3_rotation_matrix() -> np.ndarray:
    """Local U_C3=exp[-i(2*pi/3)(Sx+Sy+Sz)/sqrt(3)] for S=sigma/2."""
    theta = 2.0 * math.pi / 3.0
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sigma_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    n_dot_sigma = (sigma_x + sigma_y + sigma_z) / math.sqrt(3.0)
    return (
        math.cos(theta / 2.0) * np.eye(2, dtype=np.complex128)
        - 1.0j * math.sin(theta / 2.0) * n_dot_sigma
    )


def _apply_local_spin_rotation_and_site_permutation(
    spin_state: int,
    site_permutation: List[int],
    local_spin_rotation: np.ndarray,
    *,
    amplitude_tol: float = 1e-14,
) -> List[Tuple[int, complex]]:
    states: List[Tuple[int, complex]] = [(0, 1.0 + 0.0j)]
    raw_state = int(spin_state)
    for old_site, new_site in enumerate(site_permutation):
        input_is_up = bool((raw_state >> int(old_site)) & 1)
        input_index = 0 if input_is_up else 1
        next_states: List[Tuple[int, complex]] = []
        for partial_state, partial_coeff in states:
            for output_index, output_is_up in ((0, True), (1, False)):
                coeff = complex(local_spin_rotation[int(output_index), int(input_index)])
                if abs(coeff) <= float(amplitude_tol):
                    continue
                next_state = int(partial_state)
                if bool(output_is_up):
                    next_state |= 1 << int(new_site)
                next_states.append((next_state, complex(partial_coeff) * coeff))
        states = next_states
    return states


def build_combined_c3_operator_in_spin_orbital_u1_basis(
    geometry: GeometryData,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    *,
    amplitude_tol: float = 1e-14,
    show_progress: bool = False,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Build combined honeycomb C3 in the Tz basis.

    The orbital state is only site-rotated.  No local orbital pseudospin
    rotation is applied, so total Tz remains a good first reduction.
    """
    site_permutation, metadata = _find_honeycomb_combined_c3_site_permutation(geometry)
    local_spin_rotation = _spin_half_c3_rotation_matrix()
    dim = int(len(basis_list))
    rows: List[int] = []
    cols: List[int] = []
    data: List[complex] = []
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=dim,
        desc="C3 operator",
        unit="state",
        leave=False,
    )
    for col, (spin_state, orbital_state) in enumerate(basis_list):
        rotated_orbital = _permute_bits_by_site_map(int(orbital_state), site_permutation)
        spin_outputs = _apply_local_spin_rotation_and_site_permutation(
            int(spin_state),
            site_permutation,
            local_spin_rotation,
            amplitude_tol=float(amplitude_tol),
        )
        for rotated_spin, coeff in spin_outputs:
            row = basis_map.get((int(rotated_spin), int(rotated_orbital)))
            if row is None:
                raise ValueError("Combined C3 left the selected Tz basis; orbital site rotation should preserve Tz.")
            if abs(coeff) > float(amplitude_tol):
                rows.append(int(row))
                cols.append(int(col))
                data.append(complex(coeff))
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    operator = sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(dim, dim),
    )
    metadata["spin_rotation"] = "exp[-i(2*pi/3)*(Sx+Sy+Sz)/sqrt(3)], S=sigma/2"
    metadata["orbital_rotation"] = "site permutation only; no local orbital pseudospin rotation"
    metadata["nnz"] = int(operator.nnz)
    return operator, metadata


def _solve_projected_hamiltonian(
    hamiltonian: sparse.spmatrix,
    q_matrix: sparse.spmatrix | np.ndarray,
    *,
    requested_count: int,
    check_ground_state_degeneracy: bool,
    sparse_tol: float,
    sparse_maxiter: int | None,
    show_progress: bool,
    label: str,
    max_dense_entries: int = MAX_DENSE_PROJECTOR_ENTRIES,
    max_dense_mb: float = MAX_DENSE_PROJECTOR_MB,
    strict_projector_memory: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str, int, Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    timing: Dict[str, float] = {}
    memory: Dict[str, Any] = {
        "q_matrix": _matrix_storage_diagnostics(q_matrix),
    }
    reduced_dim = int(q_matrix.shape[1])
    if reduced_dim <= 0:
        raise ValueError("Cannot diagonalize an empty projected ED sector.")
    project_start = time.perf_counter()
    with profile_stage("standard projector construction"):
        if sparse.issparse(q_matrix):
            hamiltonian_csr = hamiltonian if sparse.isspmatrix_csr(hamiltonian) else hamiltonian.tocsr()
            q_sparse = q_matrix if sparse.isspmatrix_csc(q_matrix) else q_matrix.tocsc()
            h_red = (q_sparse.getH() @ hamiltonian_csr @ q_sparse).tocsr()
            h_red = ((h_red + h_red.getH()) * 0.5).tocsr()
        else:
            hamiltonian_csr = hamiltonian if sparse.isspmatrix_csr(hamiltonian) else hamiltonian.tocsr()
            h_red = np.asarray(q_matrix.conj().T @ (hamiltonian_csr @ q_matrix), dtype=np.complex128)
            h_red = 0.5 * (h_red + h_red.conj().T)
    timing["project_or_build_Hred"] = float(time.perf_counter() - project_start)
    memory["h_red"] = _matrix_storage_diagnostics(h_red)
    if reduced_dim == 1:
        eigenvalues = np.asarray([complex(h_red[0, 0]) if sparse.issparse(h_red) else complex(h_red[0, 0])])
        eigenvectors = np.ones((1, 1), dtype=np.complex128)
        timing["diagonalize"] = 0.0
        return eigenvalues, eigenvectors, "spin_orbital_tz_projector_dense_dim1", 0, {}, timing, memory

    solve_count, degeneracy_padding = _padded_eigsh_count(
        int(requested_count),
        reduced_dim,
        check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
    )
    if solve_count >= reduced_dim - 1:
        dense_entries = int(reduced_dim) * int(reduced_dim)
        dense_guard = _dense_allocation_diagnostics(
            label=f"{label} dense projected Hamiltonian diagonalization",
            entries=dense_entries,
            dtype=np.complex128,
            max_dense_entries=int(max_dense_entries),
            max_dense_mb=float(max_dense_mb),
        )
        memory["dense_diagonalization"] = dense_guard
        if not bool(dense_guard["allowed"]):
            memory["dense_diagonalization_skipped_reason"] = dense_guard.get("reason")
            if bool(strict_projector_memory):
                _raise_dense_memory_error(dense_guard)
            if reduced_dim <= 2:
                _raise_dense_memory_error(dense_guard)
            solve_count = max(1, min(int(requested_count), reduced_dim - 2))
        else:
            diag_start = time.perf_counter()
            with profile_stage("diagonalization"):
                dense_h = h_red.toarray() if sparse.issparse(h_red) else np.asarray(h_red)
                eigenvalues, eigenvectors = np.linalg.eigh(dense_h)
            timing["diagonalize"] = float(time.perf_counter() - diag_start)
            return (
                eigenvalues,
                eigenvectors,
                "spin_orbital_tz_projector_dense",
                int(degeneracy_padding),
                {},
                timing,
                memory,
            )

    sparse_h = h_red if sparse.issparse(h_red) else sparse.csr_matrix(h_red)
    diag_start = time.perf_counter()
    with profile_stage("diagonalization"):
        eigenvalues, eigenvectors, eigsh_info = _run_lowest_eigsh(
            sparse_h,
            eigenstate_count=solve_count,
            sparse_tol=sparse_tol,
            sparse_maxiter=sparse_maxiter,
            show_progress=show_progress,
            label=label,
        )
    timing["diagonalize"] = float(time.perf_counter() - diag_start)
    return (
        eigenvalues,
        eigenvectors,
        "spin_orbital_tz_projector_sparse",
        int(degeneracy_padding),
        eigsh_info,
        timing,
        memory,
    )


def run_spin_orbital_projected_exact_spectrum(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    *,
    coupling_j: float = 1.0,
    eigenstate_count: int = 3,
    check_ground_state_degeneracy: bool = True,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    show_progress: bool = True,
    ground_manifold_abs_tol: float = 1e-12,
    ground_manifold_rel_tol: float = 1e-12,
    sparse_tol: float = 0.0,
    sparse_maxiter: int | None = None,
    target_tz2: int = 0,
    use_spin_pi_z: bool = False,
    z2_target_parity: int = 0,
    use_translation_x: bool = False,
    use_translation_y: bool = False,
    momentum_x: int = 0,
    momentum_y: int = 0,
    use_combined_c3: bool = False,
    c3_q_blocks: str | int = "0",
    projector_tol: float = 1e-10,
    commutator_tol: float = 1e-8,
    strict_projector_memory: bool = True,
    allow_drop_c3_on_memory: bool = False,
    max_projector_parent_dim: int = MAX_PROJECTOR_PARENT_DIM,
    max_projector_nnz: int = MAX_PROJECTOR_NNZ,
    max_dense_projector_entries: int = MAX_DENSE_PROJECTOR_ENTRIES,
    max_dense_projector_mb: float = MAX_DENSE_PROJECTOR_MB,
    max_explicit_c3_parent_dim: int = MAX_EXPLICIT_C3_PARENT_DIM,
    max_explicit_c3_dim: int = MAX_EXPLICIT_C3_DIM,
    phase_scan_c3_seconds_per_point: float | None = MAX_PHASE_SCAN_C3_SECONDS_PER_POINT,
) -> Tuple[Dict[str, Any], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Diagonalize Yao-Lee ED after Tz, spin-pi-z, and fused-translation projectors.

    Total Tz is always applied first via ``build_spin_orbital_u1_basis``.  The
    returned eigenvectors are expanded back into that Tz basis, so existing ED
    observable routines can be reused without reinterpretation.
    """
    if str(model_spec.model_family) != "yao_lee":
        raise ValueError("Projector ED path is currently implemented for model_family='yao_lee'.")
    if model_spec.spin_rep != "1/2" or model_spec.orbital_rep != "1/2":
        raise ValueError("Projector ED path requires spin_rep=1/2 and orbital_rep=1/2.")

    n_sites = int(geometry.number_of_sites)
    timing_seconds: Dict[str, float] = {}
    memory_diagnostics: Dict[str, Any] = {
        "caps": {
            "MAX_PROJECTOR_PARENT_DIM": int(max_projector_parent_dim),
            "MAX_PROJECTOR_NNZ": int(max_projector_nnz),
            "MAX_DENSE_PROJECTOR_ENTRIES": int(max_dense_projector_entries),
            "MAX_DENSE_PROJECTOR_MB": float(max_dense_projector_mb),
            "MAX_EXPLICIT_C3_PARENT_DIM": int(max_explicit_c3_parent_dim),
            "MAX_EXPLICIT_C3_DIM": int(max_explicit_c3_dim),
            "MAX_PHASE_SCAN_C3_SECONDS_PER_POINT": (
                float(phase_scan_c3_seconds_per_point)
                if phase_scan_c3_seconds_per_point is not None
                else None
            ),
        }
    }
    runtime_dropped_symmetries: List[Dict[str, Any]] = []
    runtime_drop_reasons: Dict[str, str] = {}
    projector_strategy = "sparse"
    projector_total_start = time.perf_counter()
    symmetry_validation_report = _validate_ed_u1_block_request(
        geometry,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
        external_field_terms=external_field_terms,
        use_sz_block=False,
        target_sz2=0,
        use_tau_z_block=True,
        target_tz2=int(target_tz2),
    )
    stage_start = _start_stage("spin-orbital projector ED", show_progress)
    t0 = time.perf_counter()
    with profile_stage("ED basis construction"):
        basis_list, basis_map = build_spin_orbital_u1_basis(
            n_sites,
            use_sz_block=False,
            target_sz2=0,
            use_tau_z_block=True,
            target_tz2=int(target_tz2),
        )
    timing_seconds["build_Tz_basis"] = float(time.perf_counter() - t0)
    if len(basis_list) > int(max_projector_parent_dim):
        raise MemoryError(
            f"[projector-ed] Tz parent basis dimension {len(basis_list):,} exceeds "
            f"MAX_PROJECTOR_PARENT_DIM={int(max_projector_parent_dim):,}."
        )
    t0 = time.perf_counter()
    with profile_stage("ED Hamiltonian construction"):
        hamiltonian = build_sparse_hamiltonian_spin_orbital_u1(
            n_sites,
            geometry,
            model_spec,
            alpha,
            beta,
            basis_list,
            basis_map,
            coupling_j=coupling_j,
            jx=jx,
            jy=jy,
            jz=jz,
            external_field_terms=external_field_terms,
            show_progress=show_progress,
        )
    timing_seconds["build_parent_H"] = float(time.perf_counter() - t0)
    u1_dim = int(hamiltonian.shape[0])
    if u1_dim <= 0:
        raise ValueError("Empty Tz basis in projector ED path.")
    if u1_dim > int(max_projector_parent_dim):
        raise MemoryError(
            f"[projector-ed] Tz parent Hamiltonian dimension {u1_dim:,} exceeds "
            f"MAX_PROJECTOR_PARENT_DIM={int(max_projector_parent_dim):,}."
        )
    if int(hamiltonian.nnz) > int(max_projector_nnz):
        raise MemoryError(
            f"[projector-ed] Tz parent Hamiltonian nnz={int(hamiltonian.nnz):,} exceeds "
            f"MAX_PROJECTOR_NNZ={int(max_projector_nnz):,}."
        )
    memory_diagnostics["parent_hamiltonian"] = _matrix_storage_diagnostics(hamiltonian)

    identity_image = np.arange(u1_dim, dtype=np.int64)
    translation_powers: List[Tuple[int, int, np.ndarray, complex]] = [(0, 0, identity_image, 1.0 + 0.0j)]
    translation_directions: List[str] = []
    momentum_blocks: Dict[str, int] = {}
    symmetry_operators: Dict[str, sparse.csr_matrix] = {}
    commutator_norms: Dict[str, float] = {}
    projector_warnings: List[str] = []
    c3_dropped_by_memory = False

    def _c3_memory_policy(message: str) -> bool:
        nonlocal c3_dropped_by_memory, projector_strategy
        if bool(strict_projector_memory) or not bool(allow_drop_c3_on_memory):
            raise MemoryError(message)
        c3_dropped_by_memory = True
        projector_strategy = "dropped_by_memory_cap"
        drop_reason = f"{message} Dropping combined C3 and continuing with Tz/translation/Z2."
        runtime_drop_reasons["combined_c3"] = drop_reason
        if not any(item.get("name") == "combined_c3" for item in runtime_dropped_symmetries):
            runtime_dropped_symmetries.append(
                {
                    "name": "combined_c3",
                    "reason": drop_reason,
                    "category": "memory_cap",
                }
            )
        projector_warnings.append(drop_reason)
        return False

    tx_powers: List[np.ndarray] = [identity_image]
    ty_powers: List[np.ndarray] = [identity_image]
    tx_order = 1
    ty_order = 1

    orbit_start = time.perf_counter()
    with profile_stage("standard projector construction"):
        if bool(use_spin_pi_z):
            pz_operator = build_spin_pi_z_operator_in_spin_orbital_u1_basis(basis_list)
            symmetry_operators["spin_pi_z"] = pz_operator
            commutator_norms["H_spin_pi_z"] = _sparse_relative_commutator_norm(hamiltonian, pz_operator)
            if commutator_norms["H_spin_pi_z"] > float(commutator_tol):
                raise ValueError(
                    "Requested spin_pi_z projector does not commute with the Hamiltonian: "
                    f"relative_commutator_norm={commutator_norms['H_spin_pi_z']:.3e}."
                )

    with profile_stage("translation projector/orbit construction"):
        if bool(use_translation_x):
            tx_operator, tx_order, tx_perm = build_fused_translation_operator_in_spin_orbital_u1_basis(
                geometry,
                basis_list,
                basis_map,
                "x",
            )
            symmetry_operators["translation_x"] = tx_operator
            translation_directions.append("x")
            momentum_blocks["x"] = int(momentum_x) % int(tx_order)
            commutator_norms["H_Tx"] = _sparse_relative_commutator_norm(hamiltonian, tx_operator)
            if commutator_norms["H_Tx"] > float(commutator_tol):
                raise ValueError(
                    "Requested fused Tx projector does not commute with the Hamiltonian: "
                    f"relative_commutator_norm={commutator_norms['H_Tx']:.3e}."
                )
            tx_powers = [identity_image]
            for _ in range(1, int(tx_order)):
                tx_powers.append(tx_perm[tx_powers[-1]])

        if bool(use_translation_y):
            ty_operator, ty_order, ty_perm = build_fused_translation_operator_in_spin_orbital_u1_basis(
                geometry,
                basis_list,
                basis_map,
                "y",
            )
            symmetry_operators["translation_y"] = ty_operator
            translation_directions.append("y")
            momentum_blocks["y"] = int(momentum_y) % int(ty_order)
            commutator_norms["H_Ty"] = _sparse_relative_commutator_norm(hamiltonian, ty_operator)
            if commutator_norms["H_Ty"] > float(commutator_tol):
                raise ValueError(
                    "Requested fused Ty projector does not commute with the Hamiltonian: "
                    f"relative_commutator_norm={commutator_norms['H_Ty']:.3e}."
                )
            ty_powers = [identity_image]
            for _ in range(1, int(ty_order)):
                ty_powers.append(ty_perm[ty_powers[-1]])

    if "translation_x" in symmetry_operators and "translation_y" in symmetry_operators:
        commutator_norms["Tx_Ty"] = _sparse_relative_commutator_norm(
            symmetry_operators["translation_x"],
            symmetry_operators["translation_y"],
        )

    if bool(use_translation_x) or bool(use_translation_y):
        translation_powers = []
        for nx in range(int(tx_order)):
            phase_x = np.exp(-2.0j * np.pi * (int(momentum_blocks.get("x", 0)) * nx) / float(tx_order))
            for ny in range(int(ty_order)):
                phase_y = np.exp(-2.0j * np.pi * (int(momentum_blocks.get("y", 0)) * ny) / float(ty_order))
                image = ty_powers[ny][tx_powers[nx]]
                translation_powers.append((int(nx), int(ny), image, complex(phase_x * phase_y)))

    c3_operator: sparse.csr_matrix | None = None
    c3_metadata: Dict[str, Any] | None = None
    c3_q_text = str(c3_q_blocks).strip().lower()
    c3_q_values: List[int] = []
    if bool(use_combined_c3):
        c3_start = time.perf_counter()
        if (bool(use_translation_x) and int(momentum_blocks.get("x", 0)) != 0) or (
            bool(use_translation_y) and int(momentum_blocks.get("y", 0)) != 0
        ):
            raise ValueError(
                "Combined C3 projector is currently enabled only in C3-invariant momentum sectors; "
                "use kx=0 and ky=0 with translation projectors."
            )
        if c3_q_text == "all":
            c3_q_values = [0, 1, 2]
        else:
            try:
                c3_q_values = [int(c3_q_text) % 3]
            except ValueError as exc:
                raise ValueError("c3_q_blocks must be one of: all, 0, 1, 2.") from exc
        if int(u1_dim) > int(max_explicit_c3_parent_dim):
            use_combined_c3 = _c3_memory_policy(
                f"[projector-ed] Combined C3 parent dimension {int(u1_dim):,} exceeds "
                f"MAX_EXPLICIT_C3_PARENT_DIM={int(max_explicit_c3_parent_dim):,}."
            )
        estimated_c3_operator_nnz_upper = int(u1_dim) * (1 << int(n_sites))
        memory_diagnostics["c3_operator_estimate"] = {
            "upper_bound_nnz": int(estimated_c3_operator_nnz_upper),
            "parent_dim": int(u1_dim),
            "spin_rotation_outputs_per_basis_state_upper_bound": int(1 << int(n_sites)),
        }
        if bool(use_combined_c3) and estimated_c3_operator_nnz_upper > int(max_projector_nnz):
            use_combined_c3 = _c3_memory_policy(
                f"[projector-ed] Combined C3 operator upper-bound nnz={estimated_c3_operator_nnz_upper:,} exceeds "
                f"MAX_PROJECTOR_NNZ={int(max_projector_nnz):,} before construction."
            )
        if not bool(use_combined_c3):
            c3_q_values = []
            c3_metadata = {
                "available": False,
                "dropped_by_memory_guard": True,
                "reason": runtime_drop_reasons.get("combined_c3", "estimated C3 operator exceeds memory caps"),
            }
            timing_seconds["build_C3"] = float(time.perf_counter() - c3_start)
        if bool(use_combined_c3):
            with profile_stage("C3 operator/projector construction"):
                c3_operator, c3_metadata = build_combined_c3_operator_in_spin_orbital_u1_basis(
                    geometry,
                    basis_list,
                    basis_map,
                    amplitude_tol=float(projector_tol) * 1.0e-4,
                    show_progress=show_progress,
                )
            memory_diagnostics["c3_operator"] = _matrix_storage_diagnostics(c3_operator)
            if int(c3_operator.nnz) > int(max_projector_nnz):
                use_combined_c3 = _c3_memory_policy(
                    f"[projector-ed] Combined C3 operator nnz={int(c3_operator.nnz):,} exceeds "
                    f"MAX_PROJECTOR_NNZ={int(max_projector_nnz):,}."
                )
            if bool(use_combined_c3):
                timing_seconds["build_C3"] = float(time.perf_counter() - c3_start)
            else:
                timing_seconds["build_C3"] = float(time.perf_counter() - c3_start)
                c3_operator = None
                c3_metadata = {
                    "available": False,
                    "dropped_by_memory_guard": True,
                    "reason": runtime_drop_reasons.get("combined_c3", "C3 dropped by memory guard"),
                }
                c3_q_values = []

    if bool(use_combined_c3):
        assert c3_operator is not None
        symmetry_operators["combined_c3"] = c3_operator
        commutator_norms["H_C3"] = _sparse_relative_commutator_norm(hamiltonian, c3_operator)
        c3_cubed = (c3_operator @ c3_operator @ c3_operator).tocsr()
        commutator_norms["C3_cubed_minus_identity"] = float(
            sparse_linalg.norm(c3_cubed - sparse.identity(u1_dim, dtype=np.complex128, format="csr"))
            / max(1.0, math.sqrt(float(u1_dim)))
        )
        if commutator_norms["H_C3"] > float(commutator_tol):
            raise ValueError(
                "Requested combined C3 projector does not commute with the Hamiltonian: "
                f"relative_commutator_norm={commutator_norms['H_C3']:.3e}."
            )
        if commutator_norms["C3_cubed_minus_identity"] > max(float(commutator_tol), 1e-7):
            raise ValueError(
                "Constructed combined C3 operator is not order three in the selected Tz basis: "
                f"norm={commutator_norms['C3_cubed_minus_identity']:.3e}."
            )

    with profile_stage("translation projector/orbit construction"):
        (
            q_candidates,
            q_matrix,
            projector_metadata,
            orthonormalization_metadata,
        ) = _projector_basis_from_orbits(
            basis_list,
            use_spin_pi_z=bool(use_spin_pi_z),
            z2_target_parity=int(z2_target_parity),
            translation_powers=translation_powers,
            projector_tol=float(projector_tol),
            dense_svd_entry_cap=int(max_dense_projector_entries),
            dense_svd_mb_cap=float(max_dense_projector_mb),
        )
        _check_projector_matrix_caps(
            q_candidates,
            label="translation/Z2 projector candidate matrix",
            max_nnz=int(max_projector_nnz),
            max_dense_entries=int(max_dense_projector_entries),
            max_dense_mb=float(max_dense_projector_mb),
        )
        memory_diagnostics["q_candidates"] = _matrix_storage_diagnostics(q_candidates)
        _check_projector_matrix_caps(
            q_matrix,
            label="translation/Z2 projector basis",
            max_nnz=int(max_projector_nnz),
            max_dense_entries=int(max_dense_projector_entries),
            max_dense_mb=float(max_dense_projector_mb),
        )
        memory_diagnostics["q_matrix"] = _matrix_storage_diagnostics(q_matrix)
    if str(orthonormalization_metadata.get("projector_strategy", "")) == "dense_small_safe":
        projector_strategy = "dense_small_safe"
    timing_seconds["build_orbits"] = float(time.perf_counter() - orbit_start)
    base_reduced_dim = int(q_matrix.shape[1])
    if show_progress:
        print(
            "[projector-ed] basis ready: "
            f"Tz_dim={u1_dim}, reduced_dim={base_reduced_dim}, nnz={hamiltonian.nnz}, "
            f"spin_pi_z={bool(use_spin_pi_z)}, translations={translation_directions}, "
            f"momenta={momentum_blocks}, combined_c3={bool(use_combined_c3)}"
        )

    requested_count = max(1, int(eigenstate_count))
    c3_sector_results: Dict[str, Dict[str, Any]] = {}
    selected_c3_q: int | None = None
    selected_q_matrix: sparse.spmatrix | np.ndarray = q_matrix
    selected_orthonormalization = orthonormalization_metadata
    selected_projector_metadata = projector_metadata

    if bool(use_combined_c3):
        assert c3_operator is not None
        q_for_c3 = q_matrix if sparse.issparse(q_matrix) else sparse.csc_matrix(q_matrix)
        try:
            _check_projector_matrix_caps(
                q_for_c3,
                label="C3 base projector input",
                max_nnz=int(max_projector_nnz),
                max_dense_entries=int(max_dense_projector_entries),
                max_dense_mb=float(max_dense_projector_mb),
            )
        except MemoryError as exc:
            use_combined_c3 = _c3_memory_policy(str(exc))
        if not bool(use_combined_c3):
            q_for_c3 = sparse.csc_matrix((u1_dim, 0), dtype=np.complex128)
        c3_base_build_start = time.perf_counter()
        c3_base = (q_for_c3.getH() @ c3_operator @ q_for_c3).tocsc()
        h_base_for_embedded = (q_for_c3.getH() @ hamiltonian @ q_for_c3).tocsr()
        timing_seconds["build_C3_base"] = float(time.perf_counter() - c3_base_build_start)
        memory_diagnostics["c3_base_operator"] = _matrix_storage_diagnostics(c3_base)
        memory_diagnostics["c3_base_hamiltonian"] = _matrix_storage_diagnostics(h_base_for_embedded)
        if int(c3_base.nnz) > int(max_projector_nnz):
            use_combined_c3 = _c3_memory_policy(
                f"[projector-ed] C3 operator in the projected base has nnz={int(c3_base.nnz):,}, "
                f"exceeding MAX_PROJECTOR_NNZ={int(max_projector_nnz):,}."
            )
        if int(h_base_for_embedded.nnz) > int(max_projector_nnz):
            use_combined_c3 = _c3_memory_policy(
                f"[projector-ed] Embedded projected Hamiltonian for C3 has nnz={int(h_base_for_embedded.nnz):,}, "
                f"exceeding MAX_PROJECTOR_NNZ={int(max_projector_nnz):,}."
            )
        base_dim = int(q_matrix.shape[1])
        if bool(use_combined_c3):
            c3_base_squared = (c3_base @ c3_base).tocsc()
            if int(c3_base_squared.nnz) > int(max_projector_nnz):
                use_combined_c3 = _c3_memory_policy(
                    f"[projector-ed] C3 squared in the projected base has nnz={int(c3_base_squared.nnz):,}, "
                    f"exceeding MAX_PROJECTOR_NNZ={int(max_projector_nnz):,}."
                )
        if not bool(use_combined_c3):
            c3_q_values = []
        identity_base = sparse.identity(base_dim, dtype=np.complex128, format="csc")
        explicit_c3_dense_guard = _dense_allocation_diagnostics(
            label="explicit C3 q-sector basis",
            entries=int(base_dim) * int(base_dim),
            dtype=np.complex128,
            max_dense_entries=int(max_dense_projector_entries),
            max_dense_mb=float(max_dense_projector_mb),
        )
        memory_diagnostics["explicit_c3_basis_estimate"] = {
            **explicit_c3_dense_guard,
            "base_dimension": int(base_dim),
            "parent_dimension": int(u1_dim),
            "max_explicit_c3_parent_dimension": int(max_explicit_c3_parent_dim),
            "max_explicit_c3_dimension": int(max_explicit_c3_dim),
        }
        explicit_c3_basis = bool(
            use_combined_c3
            and u1_dim <= int(max_explicit_c3_parent_dim)
            and base_dim <= int(max_explicit_c3_dim)
            and bool(explicit_c3_dense_guard["allowed"])
        )
        if bool(use_combined_c3) and not explicit_c3_basis:
            explicit_skip_reason = explicit_c3_dense_guard.get("reason")
            if int(u1_dim) > int(max_explicit_c3_parent_dim):
                explicit_skip_reason = (
                    f"parent_dimension={int(u1_dim):,} exceeds "
                    f"MAX_EXPLICIT_C3_PARENT_DIM={int(max_explicit_c3_parent_dim):,}"
                )
            elif int(base_dim) > int(max_explicit_c3_dim):
                explicit_skip_reason = (
                    f"base_dimension={int(base_dim):,} exceeds "
                    f"MAX_EXPLICIT_C3_DIM={int(max_explicit_c3_dim):,}"
                )
            projector_warnings.append(
                "Combined C3 sector was solved with an embedded projector in the existing "
                f"{base_dim}-dimensional translation/Tz basis; explicit q-sector SVD basis was skipped"
                + (f" because {explicit_skip_reason}." if explicit_skip_reason else ".")
            )
        omega = np.exp(2.0j * np.pi / 3.0)
        best_energy: float | None = None
        best_payload: Tuple[np.ndarray, np.ndarray, str, int, Dict[str, Any], sparse.spmatrix | np.ndarray, Dict[str, Any], Dict[str, Any], int] | None = None
        for q_value in c3_q_values:
            c3_sector_start = time.perf_counter()
            c3_projector_base = (
                identity_base
                + (omega ** (-int(q_value))) * c3_base
                + (omega ** (-2 * int(q_value))) * c3_base_squared
            ) * (1.0 / 3.0)
            c3_projector_base = c3_projector_base.tocsc()
            if int(c3_projector_base.nnz) > int(max_projector_nnz):
                use_combined_c3 = _c3_memory_policy(
                    f"[projector-ed] C3 q={int(q_value)} projector has nnz={int(c3_projector_base.nnz):,}, "
                    f"exceeding MAX_PROJECTOR_NNZ={int(max_projector_nnz):,}."
                )
                if not bool(use_combined_c3):
                    break
            sector_dimension_estimate = int(round(float(np.real(c3_projector_base.diagonal().sum()))))
            if explicit_c3_basis:
                sector_basis_in_base, sector_orthonormalization = _orthonormalize_projector_columns(
                    c3_projector_base,
                    tol=float(projector_tol),
                    dense_svd_entry_cap=int(max_dense_projector_entries),
                    dense_svd_mb_cap=float(max_dense_projector_mb),
                    force_svd=False,
                )
                _check_projector_matrix_caps(
                    sector_basis_in_base,
                    label=f"C3 q={int(q_value)} explicit q-sector basis",
                    max_nnz=int(max_projector_nnz),
                    max_dense_entries=int(max_dense_projector_entries),
                    max_dense_mb=float(max_dense_projector_mb),
                )
                full_projector_dense_guard = _dense_allocation_diagnostics(
                    label=f"C3 q={int(q_value)} full projector basis",
                    entries=int(q_matrix.shape[0]) * int(sector_basis_in_base.shape[1]),
                    dtype=np.complex128,
                    max_dense_entries=int(max_dense_projector_entries),
                    max_dense_mb=float(max_dense_projector_mb),
                )
                memory_diagnostics.setdefault("c3_full_projector_estimates", {})[str(int(q_value))] = full_projector_dense_guard
                if not bool(full_projector_dense_guard["allowed"]):
                    use_combined_c3 = _c3_memory_policy(str(full_projector_dense_guard.get("reason")))
                    if not bool(use_combined_c3):
                        break
                sector_q_matrix = q_matrix @ sector_basis_in_base
                try:
                    _check_projector_matrix_caps(
                        sector_q_matrix,
                        label=f"C3 q={int(q_value)} full projector basis",
                        max_nnz=int(max_projector_nnz),
                        max_dense_entries=int(max_dense_projector_entries),
                        max_dense_mb=float(max_dense_projector_mb),
                    )
                except MemoryError as exc:
                    use_combined_c3 = _c3_memory_policy(str(exc))
                    if not bool(use_combined_c3):
                        break
                (
                    sector_values,
                    sector_vectors,
                    sector_solver_mode,
                    sector_padding,
                    sector_eigsh_info,
                    sector_timing,
                    sector_memory,
                ) = _solve_projected_hamiltonian(
                    hamiltonian,
                    sector_q_matrix,
                    requested_count=requested_count,
                    check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
                    sparse_tol=sparse_tol,
                    sparse_maxiter=sparse_maxiter,
                    show_progress=show_progress,
                    label=f"projector-ed-c3-q{int(q_value)}",
                    max_dense_entries=int(max_dense_projector_entries),
                    max_dense_mb=float(max_dense_projector_mb),
                    strict_projector_memory=bool(strict_projector_memory),
                )
                sector_solver_basis = "explicit_c3_q_basis"
            else:
                c3_projector_csr = c3_projector_base.tocsr()
                sector_project_start = time.perf_counter()
                with profile_stage("C3 operator/projector construction"):
                    h_sector = (c3_projector_csr.getH() @ h_base_for_embedded @ c3_projector_csr).tocsr()
                    h_sector = ((h_sector + h_sector.getH()) * 0.5).tocsr()
                sector_timing = {"project_or_build_Hred": float(time.perf_counter() - sector_project_start)}
                sector_memory = {
                    "c3_projector": _matrix_storage_diagnostics(c3_projector_csr),
                    "h_sector": _matrix_storage_diagnostics(h_sector),
                }
                if int(h_sector.nnz) > int(max_projector_nnz):
                    use_combined_c3 = _c3_memory_policy(
                        f"[projector-ed] C3 q={int(q_value)} embedded Hamiltonian has nnz={int(h_sector.nnz):,}, "
                        f"exceeding MAX_PROJECTOR_NNZ={int(max_projector_nnz):,}."
                    )
                    if not bool(use_combined_c3):
                        break
                if base_dim == 1:
                    sector_values = np.asarray([complex(h_sector[0, 0]) if sparse.issparse(h_sector) else complex(h_sector[0, 0])])
                    sector_vectors = np.ones((1, 1), dtype=np.complex128)
                    sector_solver_mode = "spin_orbital_tz_c3_embedded_dense_dim1"
                    sector_padding = 0
                    sector_eigsh_info = {}
                    sector_timing["diagonalize"] = 0.0
                else:
                    sector_solve_count, sector_padding = _padded_eigsh_count(
                        requested_count,
                        base_dim,
                        check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
                    )
                    if sector_solve_count >= base_dim - 1:
                        dense_entries = int(base_dim) * int(base_dim)
                        dense_guard = _dense_allocation_diagnostics(
                            label=f"projector-ed-c3-q{int(q_value)} embedded dense Hamiltonian diagonalization",
                            entries=dense_entries,
                            dtype=np.complex128,
                            max_dense_entries=int(max_dense_projector_entries),
                            max_dense_mb=float(max_dense_projector_mb),
                        )
                        sector_memory["dense_diagonalization"] = dense_guard
                        if bool(dense_guard["allowed"]):
                            diag_start = time.perf_counter()
                            with profile_stage("diagonalization"):
                                dense_h_sector = h_sector.toarray()
                                sector_values, sector_vectors = np.linalg.eigh(dense_h_sector)
                            sector_timing["diagonalize"] = float(time.perf_counter() - diag_start)
                            sector_solver_mode = "spin_orbital_tz_c3_embedded_dense"
                            sector_eigsh_info = {}
                        else:
                            sector_memory["dense_diagonalization_skipped_reason"] = dense_guard.get("reason")
                            if bool(strict_projector_memory):
                                _raise_dense_memory_error(dense_guard)
                            sector_solve_count = max(1, min(requested_count, base_dim - 2))
                            diag_start = time.perf_counter()
                            with profile_stage("diagonalization"):
                                sector_values, sector_vectors, sector_eigsh_info = _run_lowest_eigsh(
                                    h_sector,
                                    eigenstate_count=sector_solve_count,
                                    sparse_tol=sparse_tol,
                                    sparse_maxiter=sparse_maxiter,
                                    show_progress=show_progress,
                                    label=f"projector-ed-c3-q{int(q_value)}",
                                )
                            sector_timing["diagonalize"] = float(time.perf_counter() - diag_start)
                            sector_solver_mode = "spin_orbital_tz_c3_embedded_sparse"
                    else:
                        sparse_h_sector = h_sector
                        diag_start = time.perf_counter()
                        with profile_stage("diagonalization"):
                            sector_values, sector_vectors, sector_eigsh_info = _run_lowest_eigsh(
                                sparse_h_sector,
                                eigenstate_count=sector_solve_count,
                                sparse_tol=sparse_tol,
                                sparse_maxiter=sparse_maxiter,
                                show_progress=show_progress,
                                label=f"projector-ed-c3-q{int(q_value)}",
                            )
                        sector_timing["diagonalize"] = float(time.perf_counter() - diag_start)
                        sector_solver_mode = "spin_orbital_tz_c3_embedded_sparse"
                sector_vectors = np.asarray(c3_projector_csr @ sector_vectors, dtype=np.complex128)
                for column in range(int(sector_vectors.shape[1])):
                    norm = float(np.linalg.norm(sector_vectors[:, column]))
                    if norm > 0.0:
                        sector_vectors[:, column] /= norm
                sector_q_matrix = q_matrix
                sector_orthonormalization = {
                    "orthonormalization": "embedded_c3_projector",
                    "explicit_c3_basis": False,
                    "projector_rank_estimate": int(sector_dimension_estimate),
                }
                sector_solver_basis = "embedded_c3_projector_in_base"
            sector_order = np.argsort(np.real(sector_values))
            sector_values = np.asarray(np.real(sector_values[sector_order]), dtype=float)
            sector_vectors = np.asarray(sector_vectors[:, sector_order], dtype=np.complex128)
            sector_energy = float(sector_values[0])
            sector_metadata = {
                "q": int(q_value),
                "energy": sector_energy,
                "energies": [float(value) for value in sector_values],
                "reduced_dimension": int(sector_dimension_estimate),
                "solver_dimension": int(sector_q_matrix.shape[1]),
                "solver_mode": sector_solver_mode,
                "solver_basis": sector_solver_basis,
                "orthonormalization": sector_orthonormalization,
                "timing_seconds": {
                    **sector_timing,
                    "total_c3_sector": float(time.perf_counter() - c3_sector_start),
                },
                "memory_diagnostics": sector_memory,
                "eigsh": sector_eigsh_info if "sparse" in sector_solver_mode else None,
            }
            c3_sector_results[str(int(q_value))] = sector_metadata
            if best_energy is None or sector_energy < best_energy:
                best_energy = sector_energy
                best_payload = (
                    sector_values,
                    sector_vectors,
                    sector_solver_mode,
                    sector_padding,
                    sector_eigsh_info,
                    sector_q_matrix,
                    sector_orthonormalization,
                    {
                        **projector_metadata,
                        "combined_c3_q": int(q_value),
                        "combined_c3_sector_dimension": int(sector_dimension_estimate),
                        "combined_c3_solver_basis": sector_solver_basis,
                    },
                    int(q_value),
                )
        if best_payload is None:
            if bool(c3_dropped_by_memory):
                use_combined_c3 = False
            else:
                raise ValueError("Combined C3 projectors produced no non-empty q sectors.")
        if best_payload is not None:
            (
                eigenvalues,
                reduced_eigenvectors,
                solver_mode,
                degeneracy_padding,
                eigsh_info,
                selected_q_matrix,
                selected_orthonormalization,
                selected_projector_metadata,
                selected_c3_q,
            ) = best_payload
            selected_sector_timing = c3_sector_results.get(str(int(selected_c3_q)), {}).get("timing_seconds", {})
            if isinstance(selected_sector_timing, dict):
                if "project_or_build_Hred" in selected_sector_timing:
                    timing_seconds["project_or_build_Hred"] = float(selected_sector_timing["project_or_build_Hred"])
                if "diagonalize" in selected_sector_timing:
                    timing_seconds["diagonalize"] = float(selected_sector_timing["diagonalize"])
                if "total_c3_sector" in selected_sector_timing:
                    timing_seconds["selected_C3_sector_total"] = float(selected_sector_timing["total_c3_sector"])

    if not bool(use_combined_c3):
        (
            eigenvalues,
            reduced_eigenvectors,
            solver_mode,
            degeneracy_padding,
            eigsh_info,
            projected_timing,
            projected_memory,
        ) = _solve_projected_hamiltonian(
            hamiltonian,
            q_matrix,
            requested_count=requested_count,
            check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
            sparse_tol=sparse_tol,
            sparse_maxiter=sparse_maxiter,
            show_progress=show_progress,
            label="projector-ed",
            max_dense_entries=int(max_dense_projector_entries),
            max_dense_mb=float(max_dense_projector_mb),
            strict_projector_memory=bool(strict_projector_memory),
        )
        timing_seconds.update(projected_timing)
        memory_diagnostics["projected_solve"] = projected_memory
        order = np.argsort(np.real(eigenvalues))
        eigenvalues = np.asarray(np.real(eigenvalues[order]), dtype=float)
        reduced_eigenvectors = np.asarray(reduced_eigenvectors[:, order], dtype=np.complex128)

    solver_dim = int(selected_q_matrix.shape[1])
    reduced_dim = int(selected_projector_metadata.get("combined_c3_sector_dimension", solver_dim))
    if sparse.issparse(selected_q_matrix):
        expanded_eigenvectors = np.asarray(selected_q_matrix @ reduced_eigenvectors, dtype=np.complex128)
    else:
        expanded_eigenvectors = np.asarray(selected_q_matrix @ reduced_eigenvectors, dtype=np.complex128)
    _end_stage("spin-orbital projector ED", stage_start, show_progress)
    timing_seconds["total_standard_projector_ed"] = float(time.perf_counter() - projector_total_start)
    if (
        bool(use_combined_c3)
        and not bool(strict_projector_memory)
        and phase_scan_c3_seconds_per_point is not None
        and float(phase_scan_c3_seconds_per_point) > 0.0
        and float(timing_seconds.get("selected_C3_sector_total", timing_seconds.get("build_C3", 0.0)))
        > float(phase_scan_c3_seconds_per_point)
    ):
        projector_warnings.append(
            "[projector-ed] Combined C3 work took "
            f"{float(timing_seconds.get('selected_C3_sector_total', timing_seconds.get('build_C3', 0.0))):.3f}s, "
            f"above MAX_PHASE_SCAN_C3_SECONDS_PER_POINT={float(phase_scan_c3_seconds_per_point):.3f}s."
        )
    if bool(c3_dropped_by_memory):
        projector_strategy = "dropped_by_memory_cap"
    elif "dense" in str(solver_mode) or str(selected_orthonormalization.get("projector_strategy", "")) == "dense_small_safe":
        projector_strategy = "dense_small_safe"
    else:
        projector_strategy = "sparse"
    memory_estimate_mb = _max_recorded_memory_estimate_mb(memory_diagnostics)

    low_energy_resolution = resolve_low_energy_spectrum(
        eigenvalues,
        check_ground_state_degeneracy=bool(check_ground_state_degeneracy),
        hilbert_dim=reduced_dim,
        degeneracy_tolerance_abs=float(ground_manifold_abs_tol),
        degeneracy_tolerance_rel=float(ground_manifold_rel_tol),
    )
    spectrum: Dict[str, Any] = {
        "solver_mode": solver_mode,
        "solver_requested": "spin_orbital_tz_projector",
        "basis_type": "bitwise_spin_orbital_tz_projector_block",
        "use_sz_block": False,
        "target_sz2": None,
        "use_tau_z_block": True,
        "target_tz2": int(target_tz2),
        "use_z2_block": bool(use_spin_pi_z),
        "z2_kind": "spin_pi_z" if bool(use_spin_pi_z) else None,
        "z2_parity": int(z2_target_parity) if bool(use_spin_pi_z) else None,
        "translation_directions": list(translation_directions),
        "use_translation_x_block": bool(use_translation_x),
        "use_translation_y_block": bool(use_translation_y),
        "momentum_blocks": dict(momentum_blocks),
        "momentum_x_block": int(momentum_blocks.get("x", 0)) if bool(use_translation_x) else None,
        "momentum_y_block": int(momentum_blocks.get("y", 0)) if bool(use_translation_y) else None,
        "use_c3_block": bool(use_combined_c3),
        "c3_q_blocks_requested": str(c3_q_blocks),
        "selected_c3_q": selected_c3_q,
        "c3_sector_energies": c3_sector_results,
        "combined_c3": c3_metadata,
        "u1_basis_dimension": int(u1_dim),
        "projector_reduced_dimension": int(reduced_dim),
        "reduced_dimension": int(reduced_dim),
        "projector_solver_dimension": int(solver_dim),
        "hilbert_dim": int(reduced_dim),
        "full_spin_orbital_hilbert_dim": int(4 ** n_sites),
        "number_of_sites": n_sites,
        "vectors_are_expanded_to_u1_basis": True,
        "symmetry_validation": symmetry_validation_report,
        "projector_metadata": selected_projector_metadata,
        "projector_strategy": str(projector_strategy),
        "memory_estimate_MB": memory_estimate_mb,
        "dropped_symmetries": list(runtime_dropped_symmetries),
        "drop_reasons": dict(runtime_drop_reasons),
        "orthonormalization": selected_orthonormalization,
        "commutator_norms": commutator_norms,
        "timing_seconds": timing_seconds,
        "memory_diagnostics": memory_diagnostics,
        "strict_projector_memory": bool(strict_projector_memory),
        "allow_drop_c3_on_memory": bool(allow_drop_c3_on_memory),
        "c3_dropped_by_memory_guard": bool(c3_dropped_by_memory),
        "warnings": projector_warnings,
        "eigenstates_requested": requested_count,
        "eigenstates_returned": int(eigenvalues.size),
        "eigenstates_degeneracy_padding": int(degeneracy_padding),
        "energies": [float(value) for value in eigenvalues],
        "ground_state_energy": float(low_energy_resolution["ground_state_energy"]),
        "sparse_tol": float(eigsh_info.get("eigsh_tol_effective", sparse_tol)),
        "sparse_tol_requested": float(sparse_tol),
        "sparse_maxiter": eigsh_info.get("eigsh_maxiter"),
        "eigsh": eigsh_info if "sparse" in solver_mode else None,
        **low_energy_resolution,
    }
    return spectrum, expanded_eigenvectors, basis_list, basis_map


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

    The current yao_lee Hamiltonian is the Eq. 7 form and is not compatible
    with a strict total-Sz block, so the driver disables this path.
    """
    raise ValueError(BITWISE_ED_FORMULA)
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
    with profile_stage("ED basis construction"):
        basis_list, basis_map = build_sz_conserved_basis(n_sites, target_sz2=target_sz2)
    with profile_stage("ED Hamiltonian construction"):
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
    with profile_stage("diagonalization"):
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


def plaquette_flux_from_spin_orbital_u1_ed_state(
    geometry: GeometryData,
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    plaquette_center_idx: int | None = None,
) -> Dict[str, Any]:
    """Plaquette flux in the generic spin/orbital U(1) bit basis."""
    return plaquette_flux_from_sz_conserved_ed_state(
        geometry,
        state,
        basis_list,
        basis_map,
        plaquette_center_idx=plaquette_center_idx,
    )


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


def collect_correlation_matrices_from_spin_orbital_u1_ed(
    geometry: GeometryData,
    state: np.ndarray,
    basis_list: List[Tuple[int, int]],
    basis_map: Dict[Tuple[int, int], int],
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Collect correlations in the generic spin/orbital U(1) bit basis."""
    return collect_correlation_matrices_from_sz_conserved_ed(
        geometry,
        state,
        basis_list,
        basis_map,
        show_progress=show_progress,
    )


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


def build_spin_orbital_u1_scalar_correlations(correlations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build scalar correlations from generic spin/orbital U(1) ED data."""
    return build_sz_conserved_scalar_correlations(correlations)


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
    for bond in geometry.bond_list:
        i = int(bond.i)
        j = int(bond.j)
        gamma = str(bond.gamma).lower()
        spin_dot = sum(complex(correlations[f"S{axis}_S{axis}"][i, j]) for axis in ("x", "y", "z"))
        spin_gamma = complex(correlations[f"S{gamma}_S{gamma}"][i, j])
        orbital_dot = sum(complex(correlations[f"T{axis}_T{axis}"][i, j]) for axis in ("x", "y", "z"))
        mixed_dot_dot = sum(
            complex(correlations[f"S{spin_axis}T{orbital_axis}_S{spin_axis}T{orbital_axis}"][i, j])
            for spin_axis in ("x", "y", "z")
            for orbital_axis in ("x", "y", "z")
        )
        mixed_gamma_dot = sum(
            complex(correlations[f"S{gamma}T{orbital_axis}_S{gamma}T{orbital_axis}"][i, j])
            for orbital_axis in ("x", "y", "z")
        )
        components = [
            {
                "channel": "ST",
                "operator": "SdotTdot",
                "axis": "dot",
                "coefficient": -float(coupling_j) * float(alpha),
                "correlation": float(np.real(mixed_dot_dot)),
                "energy": float(np.real(-float(coupling_j) * float(alpha) * mixed_dot_dot)),
            },
            {
                "channel": "S",
                "operator": "Sdot",
                "axis": "dot",
                "coefficient": float(coupling_j) * float(alpha) * float(beta),
                "correlation": float(np.real(spin_dot)),
                "energy": float(np.real(float(coupling_j) * float(alpha) * float(beta) * spin_dot)),
            },
            {
                "channel": "ST",
                "operator": f"S{gamma}Tdot",
                "axis": gamma,
                "coefficient": 2.0 * float(coupling_j),
                "correlation": float(np.real(mixed_gamma_dot)),
                "energy": float(np.real(2.0 * float(coupling_j) * mixed_gamma_dot)),
            },
            {
                "channel": "S",
                "operator": f"S{gamma}",
                "axis": gamma,
                "coefficient": -2.0 * float(coupling_j) * float(beta),
                "correlation": float(np.real(spin_gamma)),
                "energy": float(np.real(-2.0 * float(coupling_j) * float(beta) * spin_gamma)),
            },
            {
                "channel": "T",
                "operator": "Tdot",
                "axis": "dot",
                "coefficient": float(coupling_j) * float(beta),
                "correlation": float(np.real(orbital_dot)),
                "energy": float(np.real(float(coupling_j) * float(beta) * orbital_dot)),
            },
            {
                "channel": "constant",
                "operator": "Id",
                "axis": "identity",
                "coefficient": -float(coupling_j) * float(beta) * float(beta),
                "correlation": 1.0,
                "energy": -float(coupling_j) * float(beta) * float(beta),
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


