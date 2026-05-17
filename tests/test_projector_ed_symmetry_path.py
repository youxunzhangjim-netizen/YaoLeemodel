"""Validation tests for the Yao-Lee projector ED symmetry path.

The fast tests use the N=4 honeycomb cylinder.  Combined C3 needs an Lx=Ly
honeycomb torus, so those checks are kept behind YL_RUN_SLOW_PROJECTOR_ED=1.
"""

from __future__ import annotations

import math
import os
import re
import sys
import unittest
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ed_backend import (  # noqa: E402
    _sparse_relative_commutator_norm,
    build_combined_c3_operator_in_spin_orbital_u1_basis,
    build_exact_hamiltonian,
    build_fused_translation_operator_in_spin_orbital_u1_basis,
    build_global_operator_cache_for_model,
    build_sparse_hamiltonian_spin_orbital_u1,
    build_spin_orbital_u1_basis,
    build_spin_pi_z_operator_in_spin_orbital_u1_basis,
    kron_all,
    run_small_cluster_exact_spectrum,
    run_spin_orbital_projected_exact_spectrum,
    run_spin_orbital_u1_exact_spectrum,
)
from models import build_lattice_geometry, build_model_spec  # noqa: E402
import models  # noqa: E402
import quspin_backend  # noqa: E402
import ylmodel_main  # noqa: E402


RTOL = 5.0e-8
ATOL = 5.0e-8


def _yao_lee_spec():
    return build_model_spec(
        spin_rep="1/2",
        orbital_rep="1/2",
        model_family="yao_lee",
        ising_axis="z",
    )


def _fast_honeycomb_cylinder():
    return build_lattice_geometry(
        "honeycomb",
        1,
        length_y=2,
        circumference_x=False,
        circumference_y=True,
    )


def _c3_honeycomb_torus():
    return build_lattice_geometry(
        "honeycomb",
        2,
        length_y=2,
        circumference_x=True,
        circumference_y=True,
    )


def _ed_planner_args(**overrides):
    defaults = {
        "external_field_treatment": "off",
        "ed_symmetry_engine": "auto",
        "ed_backend": "quspin",
        "ed_quspin_experimental_fused_translation": False,
        "run_phase_scan": False,
        "strict_symmetry_selection_rules": True,
        "u1_target_tz2": 0,
        "u1_target_sz2": 0,
        "z2_target_parity": 0,
        "lattice": "honeycomb",
        "length_x": 2,
        "length_y": 2,
        "ed_z2_mode": "off",
        "ed_z2_kind": "auto",
        "use_translation_x_block": False,
        "use_translation_y_block": False,
        "momentum_x_block": 0,
        "momentum_y_block": 0,
        "ed_c3_mode": "off",
        "ed_c3_q_blocks": "all",
        "symmetry_reductions": ("tz",),
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _full_ed_ground_energy(geometry, spec, *, coupling_j=1.0, field_terms=None, solver="dense"):
    spectrum, _vectors = run_small_cluster_exact_spectrum(
        geometry,
        spec,
        1.0,
        0.5,
        coupling_j=coupling_j,
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        external_field_terms=list(field_terms or []),
        show_progress=False,
        solver=solver,
        sparse_tol=1.0e-9,
    )
    return float(spectrum["ground_state_energy"])


def _reachable_tz2_values(n_sites: int):
    return list(range(-int(n_sites), int(n_sites) + 1, 2))


def _scan_tz_sector_energies(geometry, spec, *, coupling_j=1.0, field_terms=None):
    energies = {}
    for target_tz2 in _reachable_tz2_values(int(geometry.number_of_sites)):
        spectrum, _vectors, _basis, _basis_map = run_spin_orbital_u1_exact_spectrum(
            geometry,
            spec,
            1.0,
            0.5,
            coupling_j=coupling_j,
            eigenstate_count=1,
            check_ground_state_degeneracy=False,
            external_field_terms=list(field_terms or []),
            show_progress=False,
            sparse_tol=1.0e-10,
            target_tz2=target_tz2,
        )
        energies[int(target_tz2)] = float(spectrum["ground_state_energy"])
    return energies


def _tz_only_energy(geometry, spec, *, target_tz2=0):
    spectrum, _vectors, _basis, _basis_map = run_spin_orbital_u1_exact_spectrum(
        geometry,
        spec,
        1.0,
        0.5,
        coupling_j=1.0,
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        external_field_terms=[],
        show_progress=False,
        sparse_tol=1.0e-10,
        target_tz2=int(target_tz2),
    )
    return float(spectrum["ground_state_energy"])


def _projected_energy(
    geometry,
    spec,
    *,
    target_tz2=0,
    use_spin_pi_z=False,
    z2_target_parity=0,
    use_translation_x=False,
    use_translation_y=False,
    momentum_x=0,
    momentum_y=0,
    use_combined_c3=False,
    c3_q_blocks="all",
):
    spectrum, _vectors, _basis, _basis_map = run_spin_orbital_projected_exact_spectrum(
        geometry,
        spec,
        1.0,
        0.5,
        coupling_j=1.0,
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        external_field_terms=[],
        show_progress=False,
        sparse_tol=1.0e-10,
        target_tz2=int(target_tz2),
        use_spin_pi_z=bool(use_spin_pi_z),
        z2_target_parity=int(z2_target_parity),
        use_translation_x=bool(use_translation_x),
        use_translation_y=bool(use_translation_y),
        momentum_x=int(momentum_x),
        momentum_y=int(momentum_y),
        use_combined_c3=bool(use_combined_c3),
        c3_q_blocks=c3_q_blocks,
    )
    return float(spectrum["ground_state_energy"]), spectrum


def _projector_scan_energy(
    geometry,
    spec,
    *,
    target_tz2: int,
    field_terms=None,
    coupling_j=1.0,
    use_spin_pi_z=True,
    scan_parities=(0, 1),
    scan_ky=(0, 1),
):
    energies = {}
    for parity in scan_parities:
        for ky in scan_ky:
            spectrum, _vectors, _basis, _basis_map = run_spin_orbital_projected_exact_spectrum(
                geometry,
                spec,
                1.0,
                0.5,
                coupling_j=coupling_j,
                eigenstate_count=1,
                check_ground_state_degeneracy=False,
                external_field_terms=list(field_terms or []),
                show_progress=False,
                sparse_tol=1.0e-10,
                target_tz2=int(target_tz2),
                use_spin_pi_z=bool(use_spin_pi_z),
                z2_target_parity=int(parity),
                use_translation_y=True,
                momentum_y=int(ky),
            )
            energies[(int(parity), int(ky))] = float(spectrum["ground_state_energy"])
    return energies


def _global_sum_operator(geometry, spec, op_name: str) -> sparse.csr_matrix:
    op_cache = build_global_operator_cache_for_model(spec)
    ident = op_cache["Id"]
    n_sites = int(geometry.number_of_sites)
    out = sparse.csr_matrix((ident.shape[0] ** n_sites, ident.shape[0] ** n_sites), dtype=np.complex128)
    for site in range(n_sites):
        ops = [ident] * n_sites
        ops[site] = op_cache[op_name]
        out = out + kron_all(ops)
    return out.tocsr()


def _full_pz_operator(geometry, spec) -> sparse.csr_matrix:
    op_cache = build_global_operator_cache_for_model(spec)
    ident = op_cache["Id"]
    pz_local = sparse.diags([-1.0, -1.0, 1.0, 1.0], offsets=0, shape=ident.shape, format="csr")
    return kron_all([pz_local] * int(geometry.number_of_sites)).tocsr()


def _commutator_report_fast(geometry, spec, *, target_tz2=0, field_terms=None, coupling_j=1.0):
    full_h = build_exact_hamiltonian(
        geometry,
        spec,
        1.0,
        0.5,
        coupling_j,
        external_field_terms=list(field_terms or []),
        show_progress=False,
    ).tocsr()
    full_report = {
        "H_Tz": _sparse_relative_commutator_norm(full_h, _global_sum_operator(geometry, spec, "Tz")),
        "H_Pz": _sparse_relative_commutator_norm(full_h, _full_pz_operator(geometry, spec)),
    }

    basis, basis_map = build_spin_orbital_u1_basis(
        int(geometry.number_of_sites),
        use_tau_z_block=True,
        target_tz2=int(target_tz2),
    )
    h_u1 = build_sparse_hamiltonian_spin_orbital_u1(
        int(geometry.number_of_sites),
        geometry,
        spec,
        1.0,
        0.5,
        basis,
        basis_map,
        coupling_j=coupling_j,
        external_field_terms=list(field_terms or []),
        show_progress=False,
    )
    pz = build_spin_pi_z_operator_in_spin_orbital_u1_basis(basis)
    ty, _order_y, _perm_y = build_fused_translation_operator_in_spin_orbital_u1_basis(
        geometry,
        basis,
        basis_map,
        "y",
    )
    full_report.update(
        {
            "H_Tz_in_fixed_Tz_basis": 0.0,
            "H_spin_pi_z_in_Tz_basis": _sparse_relative_commutator_norm(h_u1, pz),
            "H_Ty_in_Tz_basis": _sparse_relative_commutator_norm(h_u1, ty),
        }
    )
    return full_report


def _commutator_report_c3(geometry, spec, *, field_terms=None, coupling_j=1.0):
    basis, basis_map = build_spin_orbital_u1_basis(
        int(geometry.number_of_sites),
        use_tau_z_block=True,
        target_tz2=0,
    )
    h_u1 = build_sparse_hamiltonian_spin_orbital_u1(
        int(geometry.number_of_sites),
        geometry,
        spec,
        1.0,
        0.5,
        basis,
        basis_map,
        coupling_j=coupling_j,
        external_field_terms=list(field_terms or []),
        show_progress=False,
    )
    pz = build_spin_pi_z_operator_in_spin_orbital_u1_basis(basis)
    tx, _order_x, _perm_x = build_fused_translation_operator_in_spin_orbital_u1_basis(
        geometry,
        basis,
        basis_map,
        "x",
    )
    ty, _order_y, _perm_y = build_fused_translation_operator_in_spin_orbital_u1_basis(
        geometry,
        basis,
        basis_map,
        "y",
    )
    c3, _metadata = build_combined_c3_operator_in_spin_orbital_u1_basis(
        geometry,
        basis,
        basis_map,
        show_progress=False,
    )
    return {
        "H_Tz_in_fixed_Tz_basis": 0.0,
        "H_Pz": _sparse_relative_commutator_norm(h_u1, pz),
        "H_Tx": _sparse_relative_commutator_norm(h_u1, tx),
        "H_Ty": _sparse_relative_commutator_norm(h_u1, ty),
        "H_C3": _sparse_relative_commutator_norm(h_u1, c3),
        "C3_cubed_minus_identity": float(
            sparse_linalg.norm(c3 @ c3 @ c3 - sparse.identity(h_u1.shape[0], dtype=np.complex128, format="csr"))
            / max(1.0, math.sqrt(float(h_u1.shape[0])))
        ),
    }


def _assert_close(left, right, *, label: str, atol=ATOL, rtol=RTOL):
    assert np.isclose(float(left), float(right), atol=atol, rtol=rtol), f"{label}: {left} != {right}"


def _skip_slow_projector_ed_unless_requested():
    if os.environ.get("YL_RUN_SLOW_PROJECTOR_ED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise unittest.SkipTest("set YL_RUN_SLOW_PROJECTOR_ED=1 to run N=8 C3 projector ED validations")


@contextmanager
def _raises_match(exception_type, match: str):
    try:
        yield
    except exception_type as exc:
        if re.search(match, str(exc)):
            return
        raise AssertionError(f"exception {exc!r} did not match {match!r}") from exc
    raise AssertionError(f"expected {exception_type.__name__} matching {match!r}")


def test_no_field_full_tz_and_projector_ed_are_consistent():
    geometry = _fast_honeycomb_cylinder()
    spec = _yao_lee_spec()

    full_energy = _full_ed_ground_energy(geometry, spec, solver="dense")
    tz_energies = _scan_tz_sector_energies(geometry, spec)
    _assert_close(min(tz_energies.values()), full_energy, label="full ED vs min over Tz sectors")

    target_tz2 = 0
    projector_energies = _projector_scan_energy(
        geometry,
        spec,
        target_tz2=target_tz2,
        scan_parities=(0, 1),
        scan_ky=(0, 1),
    )
    _assert_close(
        min(projector_energies.values()),
        tz_energies[target_tz2],
        label="Tz-only ED vs projector ED scanned over spin_pi_z and ky",
    )

    report = _commutator_report_fast(geometry, spec, target_tz2=target_tz2)
    print("no-field commutators:", report)
    assert report["H_Tz"] < 1.0e-12, report
    assert report["H_Pz"] < 1.0e-12, report
    assert report["H_Ty_in_Tz_basis"] < 1.0e-12, report


def test_hz_tz_spin_pi_z_translation_scan_matches_full_ed():
    geometry = _fast_honeycomb_cylinder()
    spec = _yao_lee_spec()
    hz_terms = [(-0.7, "Sz")]

    full_energy = _full_ed_ground_energy(geometry, spec, field_terms=hz_terms, solver="dense")
    projected_min_by_tz = []
    for target_tz2 in _reachable_tz2_values(int(geometry.number_of_sites)):
        projector_energies = _projector_scan_energy(
            geometry,
            spec,
            target_tz2=target_tz2,
            field_terms=hz_terms,
            scan_parities=(0, 1),
            scan_ky=(0, 1),
        )
        projected_min_by_tz.append(min(projector_energies.values()))
    _assert_close(
        min(projected_min_by_tz),
        full_energy,
        label="Hz full ED vs projector scan over Tz, spin_pi_z parity, and ky",
    )

    report = _commutator_report_fast(geometry, spec, target_tz2=0, field_terms=hz_terms)
    print("Hz commutators:", report)
    assert report["H_Tz"] < 1.0e-12, report
    assert report["H_Pz"] < 1.0e-12, report
    assert report["H_Ty_in_Tz_basis"] < 1.0e-12, report


def test_translation_projectors_require_periodic_boundaries():
    spec = _yao_lee_spec()
    open_geometry = build_lattice_geometry(
        "honeycomb",
        1,
        length_y=2,
        circumference_x=False,
        circumference_y=False,
    )
    basis, basis_map = build_spin_orbital_u1_basis(
        int(open_geometry.number_of_sites),
        use_tau_z_block=True,
        target_tz2=0,
    )
    with _raises_match(ValueError, "periodic x"):
        build_fused_translation_operator_in_spin_orbital_u1_basis(open_geometry, basis, basis_map, "x")
    with _raises_match(ValueError, "periodic y"):
        build_fused_translation_operator_in_spin_orbital_u1_basis(open_geometry, basis, basis_map, "y")

    y_periodic = _fast_honeycomb_cylinder()
    basis_y, basis_map_y = build_spin_orbital_u1_basis(
        int(y_periodic.number_of_sites),
        use_tau_z_block=True,
        target_tz2=0,
    )
    with _raises_match(ValueError, "periodic x"):
        build_fused_translation_operator_in_spin_orbital_u1_basis(y_periodic, basis_y, basis_map_y, "x")
    ty, order_y, _perm_y = build_fused_translation_operator_in_spin_orbital_u1_basis(
        y_periodic,
        basis_y,
        basis_map_y,
        "y",
    )
    assert order_y == 2
    assert ty.shape == (len(basis_y), len(basis_y))
    assert spec.model_family == "yao_lee"


def test_hz_rejects_combined_c3_on_torus_and_commutator_is_nonzero():
    _skip_slow_projector_ed_unless_requested()
    geometry = _c3_honeycomb_torus()
    spec = _yao_lee_spec()
    hz_terms = [(-1.0, "Sz")]

    report = _commutator_report_c3(geometry, spec, field_terms=hz_terms, coupling_j=0.0)
    print("Hz C3 commutators:", report)
    assert report["H_Tz_in_fixed_Tz_basis"] == 0.0, report
    assert report["H_Pz"] < 1.0e-12, report
    assert report["H_Tx"] < 1.0e-12, report
    assert report["H_Ty"] < 1.0e-12, report
    assert report["H_C3"] > 1.0e-4, report

    with _raises_match(ValueError, "combined C3 projector does not commute"):
        run_spin_orbital_projected_exact_spectrum(
            geometry,
            spec,
            1.0,
            0.5,
            coupling_j=0.0,
            eigenstate_count=1,
            check_ground_state_degeneracy=False,
            external_field_terms=hz_terms,
            show_progress=False,
            target_tz2=0,
            use_translation_x=True,
            use_translation_y=True,
            momentum_x=0,
            momentum_y=0,
            use_combined_c3=True,
            c3_q_blocks="0",
        )


def test_h111_translation_c3_q_scan_and_pure_field_energy():
    _skip_slow_projector_ed_unless_requested()
    geometry = _c3_honeycomb_torus()
    spec = _yao_lee_spec()
    h_strength = 1.0
    coefficient = -float(h_strength) / math.sqrt(3.0)
    h111_terms = [(coefficient, "Sx"), (coefficient, "Sy"), (coefficient, "Sz")]

    full_energy = _full_ed_ground_energy(
        geometry,
        spec,
        coupling_j=0.0,
        field_terms=h111_terms,
        solver="sparse",
    )
    expected_energy = -0.5 * h_strength * int(geometry.number_of_sites)
    _assert_close(full_energy, expected_energy, label="pure H[111] full ED energy", atol=1.0e-7, rtol=1.0e-7)
    _assert_close(
        full_energy / float(geometry.number_of_sites),
        -0.5 * h_strength,
        label="pure H[111] E/N",
        atol=1.0e-7,
        rtol=1.0e-7,
    )

    spectrum, _vectors, _basis, _basis_map = run_spin_orbital_projected_exact_spectrum(
        geometry,
        spec,
        1.0,
        0.5,
        coupling_j=0.0,
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        external_field_terms=h111_terms,
        show_progress=False,
        sparse_tol=1.0e-8,
        target_tz2=0,
        use_translation_x=True,
        use_translation_y=True,
        momentum_x=0,
        momentum_y=0,
        use_combined_c3=True,
        c3_q_blocks="all",
    )
    q_energies = {
        int(q): float(payload["energy"])
        for q, payload in spectrum["c3_sector_energies"].items()
    }
    _assert_close(min(q_energies.values()), full_energy, label="H[111] full ED vs min C3 q-sector")
    assert set(q_energies) == {0, 1, 2}

    report = _commutator_report_c3(geometry, spec, field_terms=h111_terms, coupling_j=0.0)
    print("H[111] C3 commutators:", report)
    assert report["H_Tz_in_fixed_Tz_basis"] == 0.0, report
    assert report["H_Tx"] < 1.0e-12, report
    assert report["H_Ty"] < 1.0e-12, report
    assert report["H_C3"] < 1.0e-12, report
    assert report["C3_cubed_minus_identity"] < 1.0e-10, report


def test_quspin_tensor_basis_translation_with_tz_is_rejected():
    try:
        import quspin_backend
        available, reason = quspin_backend.quspin_package_available()
    except Exception as exc:
        raise unittest.SkipTest(f"QuSpin is not importable in this environment: {exc}")
    if not available:
        raise unittest.SkipTest(f"QuSpin is not importable in this environment: {reason}")

    geometry = _c3_honeycomb_torus()
    with unittest.TestCase().assertRaisesRegex(ValueError, "fused physical-site translation"):
        quspin_backend.build_quspin_yao_lee_basis(
            int(geometry.number_of_sites),
            geometry=geometry,
            use_sz_block=False,
            target_sz2=0,
            use_tau_z_block=True,
            target_tz2=0,
            use_z2_block=False,
            use_translation_block=True,
            use_translation_x_block=False,
            use_translation_y_block=True,
            momentum_block_1=0,
            momentum_block_2=0,
            momentum_x_block=0,
            momentum_y_block=0,
            use_reflection_block=False,
            reflection_block=0,
        )


def test_ed_auto_routes_quspin_tz_translation_to_standard_projector():
    ylmodel_main.classify_external_field = models.classify_external_field
    geometry = _c3_honeycomb_torus()
    spec = _yao_lee_spec()
    args = _ed_planner_args(
        ed_backend="quspin",
        ed_symmetry_engine="quspin_native",
        use_translation_x_block=True,
        use_translation_y_block=True,
    )
    plan = ylmodel_main._resolve_ed_symmetry_plan(
        args=args,
        model_spec_obj=spec,
        geometry_obj=geometry,
        resolved_field_vector=(0.0, 0.0, 0.0),
        hamiltonian_field_terms=[],
        shared_symmetry_settings={
            "use_sz_block": False,
            "use_tau_z_block": True,
            "use_z2_block": False,
        },
    )
    expected_reason = (
        "QuSpin tensor_basis translation is not used with Tz because Yao-Lee translation must act "
        "on fused spin-orbital physical sites."
    )
    assert plan["requested_backend"] == "quspin", plan
    assert plan["actual_backend"] == "standard", plan
    assert plan["effective_engine"] == "standard_projector", plan
    assert "translation_x" in plan["accepted_symmetries"], plan
    assert "translation_y" in plan["accepted_symmetries"], plan
    assert plan["backend_override_reason"] == expected_reason, plan
    assert plan["engine_selection_reason"] == expected_reason, plan
    fused_report = plan.get("quspin_experimental_fused_translation")
    assert isinstance(fused_report, dict), plan
    assert fused_report.get("available") is False, fused_report


def test_combined_c3_requires_gamma_momentum_in_ed_planner():
    ylmodel_main.classify_external_field = models.classify_external_field
    geometry = _c3_honeycomb_torus()
    spec = _yao_lee_spec()
    expected_reason = (
        "combined C3 is implemented only in the Gamma momentum sector because translations "
        "and C3 do not commute at generic momentum."
    )

    nonstrict_args = _ed_planner_args(
        ed_backend="standard",
        ed_symmetry_engine="standard_projector",
        ed_c3_mode="on",
        momentum_x_block=1,
        momentum_y_block=0,
        strict_symmetry_selection_rules=False,
    )
    plan = ylmodel_main._resolve_ed_symmetry_plan(
        args=nonstrict_args,
        model_spec_obj=spec,
        geometry_obj=geometry,
        resolved_field_vector=(0.0, 0.0, 0.0),
        hamiltonian_field_terms=[],
        shared_symmetry_settings={
            "use_sz_block": False,
            "use_tau_z_block": True,
            "use_z2_block": False,
        },
    )
    c3_drops = [item for item in plan["dropped_symmetries"] if item.get("name") == "combined_c3"]
    assert c3_drops, plan
    assert c3_drops[-1]["reason"] == expected_reason, plan
    assert plan["use_c3_block"] is False, plan

    strict_args = _ed_planner_args(
        ed_backend="standard",
        ed_symmetry_engine="standard_projector",
        ed_c3_mode="auto",
        momentum_x_block=0,
        momentum_y_block=1,
        strict_symmetry_selection_rules=True,
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, re.escape(expected_reason)):
        ylmodel_main._resolve_ed_symmetry_plan(
            args=strict_args,
            model_spec_obj=spec,
            geometry_obj=geometry,
            resolved_field_vector=(0.0, 0.0, 0.0),
            hamiltonian_field_terms=[],
            shared_symmetry_settings={
                "use_sz_block": False,
                "use_tau_z_block": True,
                "use_z2_block": False,
            },
        )


def test_quspin_z2_routing_distinguishes_spin_flip_and_spin_pi_z():
    ylmodel_main.classify_external_field = models.classify_external_field
    geometry = _c3_honeycomb_torus()
    spec = _yao_lee_spec()
    shared = {
        "use_sz_block": False,
        "use_tau_z_block": True,
        "use_z2_block": True,
    }

    spin_pi_z_args = _ed_planner_args(
        ed_backend="quspin",
        ed_symmetry_engine="quspin_native",
        ed_z2_mode="on",
        ed_z2_kind="spin_pi_z",
    )
    spin_pi_z_plan = ylmodel_main._resolve_ed_symmetry_plan(
        args=spin_pi_z_args,
        model_spec_obj=spec,
        geometry_obj=geometry,
        resolved_field_vector=(0.0, 0.0, 0.0),
        hamiltonian_field_terms=[],
        shared_symmetry_settings=shared,
    )
    assert spin_pi_z_plan["actual_backend"] == "standard", spin_pi_z_plan
    assert spin_pi_z_plan["effective_engine"] == "standard_projector", spin_pi_z_plan
    assert spin_pi_z_plan["z2_generator_used"] == "spin_pi_z", spin_pi_z_plan
    assert "z2:spin_pi_z" in spin_pi_z_plan["accepted_symmetries"], spin_pi_z_plan
    assert "spin_pi_z parity requires standard_projector" in spin_pi_z_plan["backend_override_reason"]
    assert "does not implement spin_pi_z" in spin_pi_z_plan["quspin_z2_selection_reason"]

    spin_flip_args = _ed_planner_args(
        ed_backend="quspin",
        ed_symmetry_engine="quspin_native",
        ed_z2_mode="on",
        ed_z2_kind="spin_flip",
    )
    spin_flip_plan = ylmodel_main._resolve_ed_symmetry_plan(
        args=spin_flip_args,
        model_spec_obj=spec,
        geometry_obj=geometry,
        resolved_field_vector=(0.0, 0.0, 0.0),
        hamiltonian_field_terms=[],
        shared_symmetry_settings=shared,
    )
    assert spin_flip_plan["actual_backend"] == "quspin", spin_flip_plan
    assert spin_flip_plan["effective_engine"] == "quspin_native", spin_flip_plan
    assert spin_flip_plan["z2_generator_used"] == "spin_flip", spin_flip_plan
    assert "z2:spin_flip" in spin_flip_plan["accepted_symmetries"], spin_flip_plan
    assert "QuSpin-native selected the spin_flip zblock" in spin_flip_plan["z2_selection_reason"]

    spin_flip_translation_args = _ed_planner_args(
        ed_backend="quspin",
        ed_symmetry_engine="quspin_native",
        ed_z2_mode="on",
        ed_z2_kind="spin_flip",
        use_translation_y_block=True,
    )
    spin_flip_translation_plan = ylmodel_main._resolve_ed_symmetry_plan(
        args=spin_flip_translation_args,
        model_spec_obj=spec,
        geometry_obj=geometry,
        resolved_field_vector=(0.0, 0.0, 0.0),
        hamiltonian_field_terms=[],
        shared_symmetry_settings=shared,
    )
    assert spin_flip_translation_plan["actual_backend"] == "standard", spin_flip_translation_plan
    assert spin_flip_translation_plan["z2_generator_used"] == "spin_flip", spin_flip_translation_plan
    assert "Translation is requested, so the route is standard_projector" in spin_flip_translation_plan["z2_selection_reason"]
    assert "fused spin-orbital physical sites" in spin_flip_translation_plan["backend_override_reason"]


def test_n8_quspin_regression_routing_matrix_for_old_translation_failure():
    ylmodel_main.classify_external_field = models.classify_external_field
    geometry = _c3_honeycomb_torus()
    spec = _yao_lee_spec()
    shared = {
        "use_sz_block": False,
        "use_tau_z_block": True,
        "use_z2_block": False,
    }
    exact_fused_translation_reason = (
        "QuSpin tensor_basis translation is not used with Tz because Yao-Lee translation must act "
        "on fused spin-orbital physical sites."
    )

    def resolve(**kwargs):
        args = _ed_planner_args(**kwargs)
        return ylmodel_main._resolve_ed_symmetry_plan(
            args=args,
            model_spec_obj=spec,
            geometry_obj=geometry,
            resolved_field_vector=(0.0, 0.0, 0.0),
            hamiltonian_field_terms=[],
            shared_symmetry_settings={
                **shared,
                "use_z2_block": bool(kwargs.get("ed_z2_mode") == "on"),
            },
        )

    plan_tz = resolve(ed_backend="quspin", ed_symmetry_engine="auto")
    assert plan_tz["actual_backend"] == "quspin", plan_tz
    assert plan_tz["effective_engine"] == "quspin_native", plan_tz
    assert plan_tz["accepted_symmetries"] == ["tz"], plan_tz

    plan_tz_z2 = resolve(
        ed_backend="quspin",
        ed_symmetry_engine="auto",
        ed_z2_mode="on",
        ed_z2_kind="spin_flip",
        symmetry_reductions=("tz", "z2"),
    )
    assert plan_tz_z2["actual_backend"] == "quspin", plan_tz_z2
    assert plan_tz_z2["z2_generator_used"] == "spin_flip", plan_tz_z2
    assert "z2:spin_flip" in plan_tz_z2["accepted_symmetries"], plan_tz_z2

    plan_tz_translation = resolve(
        ed_backend="quspin",
        ed_symmetry_engine="auto",
        use_translation_x_block=True,
        use_translation_y_block=True,
        momentum_x_block=0,
        momentum_y_block=0,
    )
    assert plan_tz_translation["actual_backend"] == "standard", plan_tz_translation
    assert plan_tz_translation["effective_engine"] == "standard_projector", plan_tz_translation
    assert plan_tz_translation["backend_override_reason"] == exact_fused_translation_reason, plan_tz_translation
    assert "translation_x" in plan_tz_translation["accepted_symmetries"], plan_tz_translation
    assert "translation_y" in plan_tz_translation["accepted_symmetries"], plan_tz_translation

    plan_tz_z2_translation = resolve(
        ed_backend="quspin",
        ed_symmetry_engine="auto",
        ed_z2_mode="on",
        ed_z2_kind="spin_flip",
        symmetry_reductions=("tz", "z2"),
        use_translation_x_block=True,
        use_translation_y_block=True,
        momentum_x_block=0,
        momentum_y_block=0,
    )
    assert plan_tz_z2_translation["actual_backend"] == "standard", plan_tz_z2_translation
    assert plan_tz_z2_translation["z2_generator_used"] == "spin_flip", plan_tz_z2_translation
    assert plan_tz_z2_translation["backend_override_reason"] == exact_fused_translation_reason, plan_tz_z2_translation

    plan_tz_translation_c3 = resolve(
        ed_backend="quspin",
        ed_symmetry_engine="auto",
        ed_c3_mode="on",
        use_translation_x_block=True,
        use_translation_y_block=True,
        momentum_x_block=0,
        momentum_y_block=0,
    )
    assert plan_tz_translation_c3["actual_backend"] == "standard", plan_tz_translation_c3
    assert "combined_c3" in plan_tz_translation_c3["accepted_symmetries"], plan_tz_translation_c3
    assert plan_tz_translation_c3["use_c3_block"] is True, plan_tz_translation_c3

    gamma_reason = (
        "combined C3 is implemented only in the Gamma momentum sector because translations "
        "and C3 do not commute at generic momentum."
    )
    plan_nonzero_c3 = resolve(
        ed_backend="quspin",
        ed_symmetry_engine="auto",
        ed_c3_mode="on",
        use_translation_x_block=True,
        use_translation_y_block=True,
        momentum_x_block=1,
        momentum_y_block=0,
        strict_symmetry_selection_rules=False,
    )
    c3_drops = [item for item in plan_nonzero_c3["dropped_symmetries"] if item.get("name") == "combined_c3"]
    assert c3_drops and c3_drops[-1]["reason"] == gamma_reason, plan_nonzero_c3
    assert plan_nonzero_c3["use_c3_block"] is False, plan_nonzero_c3
    assert plan_nonzero_c3["actual_backend"] == "standard", plan_nonzero_c3

    with unittest.TestCase().assertRaisesRegex(ValueError, re.escape(gamma_reason)):
        resolve(
            ed_backend="quspin",
            ed_symmetry_engine="auto",
            ed_c3_mode="on",
            use_translation_x_block=True,
            use_translation_y_block=True,
            momentum_x_block=0,
            momentum_y_block=1,
            strict_symmetry_selection_rules=True,
        )


def test_n8_quspin_regression_energy_consistency_against_tz_parent():
    _skip_slow_projector_ed_unless_requested()
    geometry = _c3_honeycomb_torus()
    spec = _yao_lee_spec()
    target_tz2 = 0
    tz_energy = _tz_only_energy(geometry, spec, target_tz2=target_tz2)

    try:
        import quspin_backend
        quspin_available, quspin_reason = quspin_backend.quspin_package_available()
    except Exception as exc:
        quspin_available, quspin_reason = False, str(exc)
    if quspin_available:
        quspin_spectrum, _vectors = quspin_backend.run_small_cluster_exact_spectrum(
            geometry=geometry,
            model_spec=spec,
            alpha=1.0,
            beta=0.5,
            coupling_j=1.0,
            eigenstate_count=1,
            check_ground_state_degeneracy=False,
            external_field_terms=[],
            show_progress=False,
            solver="sparse",
            sparse_tol=1.0e-10,
            use_sz_block=False,
            use_tau_z_block=True,
            target_tz2=target_tz2,
            use_z2_block=False,
            use_translation_block=False,
        )
        _assert_close(
            float(quspin_spectrum["ground_state_energy"]),
            tz_energy,
            label="QuSpin Tz-only vs standard Tz-only",
            atol=1.0e-7,
            rtol=1.0e-7,
        )

        spin_flip_parity_energies = []
        for parity in (0, 1):
            z2_spectrum, _vectors = quspin_backend.run_small_cluster_exact_spectrum(
                geometry=geometry,
                model_spec=spec,
                alpha=1.0,
                beta=0.5,
                coupling_j=1.0,
                eigenstate_count=1,
                check_ground_state_degeneracy=False,
                external_field_terms=[],
                show_progress=False,
                solver="sparse",
                sparse_tol=1.0e-10,
                use_sz_block=False,
                use_tau_z_block=True,
                target_tz2=target_tz2,
                use_z2_block=True,
                z2_generator="spin_flip",
                z2_target_parity=parity,
                use_translation_block=False,
            )
            spin_flip_parity_energies.append(float(z2_spectrum["ground_state_energy"]))
        _assert_close(
            min(spin_flip_parity_energies),
            tz_energy,
            label="QuSpin spin_flip parity scan vs standard Tz-only",
            atol=1.0e-7,
            rtol=1.0e-7,
        )
    else:
        print(f"QuSpin unavailable; skipped simple QuSpin energy comparison: {quspin_reason}")

    translation_energies = {}
    for kx in (0, 1):
        for ky in (0, 1):
            energy, _spectrum = _projected_energy(
                geometry,
                spec,
                target_tz2=target_tz2,
                use_translation_x=True,
                use_translation_y=True,
                momentum_x=kx,
                momentum_y=ky,
            )
            translation_energies[(kx, ky)] = energy
    _assert_close(
        min(translation_energies.values()),
        tz_energy,
        label="all fused translation sectors vs Tz-only",
        atol=1.0e-7,
        rtol=1.0e-7,
    )

    gamma_translation_energy = translation_energies[(0, 0)]
    c3_energy, c3_spectrum = _projected_energy(
        geometry,
        spec,
        target_tz2=target_tz2,
        use_translation_x=True,
        use_translation_y=True,
        momentum_x=0,
        momentum_y=0,
        use_combined_c3=True,
        c3_q_blocks="all",
    )
    q_energies = {
        int(q): float(payload["energy"])
        for q, payload in c3_spectrum["c3_sector_energies"].items()
    }
    _assert_close(
        min(q_energies.values()),
        gamma_translation_energy,
        label="C3 q-sector scan vs Gamma fused translation sector",
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert c3_energy >= tz_energy - 1.0e-7


def test_quspin_alias_and_combined_c3_report_are_explicit():
    assert ylmodel_main._normalize_ed_symmetry_engine("quspin") == "quspin_native"
    report = quspin_backend.quspin_combined_c3_api_support_report(model_family="yao_lee")
    assert report["combined_c3_implemented"] is False
    assert report["pure_site_permutation_c3_rejected_for_yao_lee"] is True
    assert "non-diagonal local spin rotation" in report["why_not_native_quspin"]
    assert "[111] basis" in report["what_would_be_needed_for_quspin"]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite


if __name__ == "__main__":
    unittest.main()
