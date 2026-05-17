#!/usr/bin/env python3
"""quimb.tensor PEPS/iPEPS backend for two-dimensional Yao-Lee runs.

The public functions here deliberately mirror the other backend modules:

``build_2d_local_hamiltonian``
    Convert the shared ``models.py`` Hamiltonian terms into a
    ``quimb.tensor.LocalHam2D`` object.

``optimize_ipeps_simple_update``
    Build a random PEPS/iPEPS unit cell and run quimb's Simple Update imaginary
    time evolution.

``evaluate_ipeps_observables``
    Contract the optimized PEPS with quimb's boundary/plaquette-environment
    machinery and return the common JSON fields used by the driver.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba-cache"))

import quimb.tensor as qtn

from analysis import (
    DEFAULT_PHASE_CLASSIFIER_THRESHOLDS,
    _classify_phase_from_diagnostics,
    _end_stage,
    _make_progress_bar,
    _phase_observable_diagnostics,
    _start_stage,
)
from ed_backend import all_bond_energies, build_spin_orbital_scalar_correlations
from models import (
    QUIMB_IPEPS_LATTICE_OPTIONS,
    QUIMB_PEPS_LATTICE_OPTIONS,
    all_high_symmetry_structure_factors,
    build_lattice_geometry,
    build_site_ops,
    build_model_spec,
    honeycomb_plaquette_flux_operators,
    is_trivial_orbital,
    plaquette_flux_close_to_target,
    quimb_ipeps_supports_lattice,
    quimb_peps_supports_lattice,
    two_site_operator_terms_for_bond,
)


SUPPORTED_IPEPS_LATTICES = QUIMB_IPEPS_LATTICE_OPTIONS
SUPPORTED_PEPS_LATTICES = QUIMB_PEPS_LATTICE_OPTIONS
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.0
DEFAULT_COUPLING_J = 1.0
DEFAULT_JX = 1.0
DEFAULT_JY = 1.0
DEFAULT_JZ = 1.0
DEFAULT_TAU = 0.1
DEFAULT_CTM_CHI = 64
ZERO_TOL = 1.0e-14
DENSE_PEPS_SYMMETRY_WARNING = (
    "Current PEPS/iPEPS backend uses dense tensors; U(1)_Tz charge maps and "
    "Tz-neutral gates are enabled, but block-sparse PEPS/iPEPS cost reduction "
    "is not exposed by this quimb SimpleUpdate API."
)
PEPS_SYMMETRY_MODE_OPTIONS = ("auto", "none", "u1_tz", "u1_tz_z2")
IPEPS_CONTRACTION_METHOD_OPTIONS = ("auto", "ctmrg", "crtg", "boundary")
IPEPS_UNIT_CELL_CANDIDATES = (
    "minimal",
    "two_sublattice",
    "stripy",
    "zigzag",
    "plaquette",
)
# Internal order follows ``build_site_ops`` in models.py, where the composite
# one-site operator matrices are Kronecker products in spin-major order.
LOCAL_SPIN_ORBITAL_BASIS = [
    "|S up, T up>",
    "|S up, T down>",
    "|S down, T up>",
    "|S down, T down>",
]
LOCAL_SPIN_ORBITAL_NAMED_BASIS = [
    "S_up_O_up",
    "S_up_O_down",
    "S_down_O_up",
    "S_down_O_down",
]
LOCAL_TZ_CHARGES = [1, -1, 1, -1]
LOCAL_TZ_CHARGE_MAP = {
    "S_down_O_down": -1,
    "S_down_O_up": 1,
    "S_up_O_down": -1,
    "S_up_O_up": 1,
}
LOCAL_TZ_REQUESTED_BASIS_ORDER = [
    "S_down_O_down",
    "S_down_O_up",
    "S_up_O_down",
    "S_up_O_up",
]


def _as_float_list(values: Iterable[float] | None, fallback: float) -> List[float]:
    if values is None:
        return [float(fallback)]
    parsed = [float(value) for value in values]
    return parsed if parsed else [float(fallback)]


def _as_step_list(steps: int | Iterable[int], tau_count: int) -> List[int]:
    if isinstance(steps, (list, tuple)):
        parsed = [max(0, int(value)) for value in steps]
        return parsed if parsed else [0]
    return [max(0, int(steps)) for _ in range(max(1, int(tau_count)))]


def _parse_use_sz_conserved_flag(
    *,
    args: Any = None,
    use_sz_conserved: bool | None = None,
    symmetry_reductions: Any = None,
) -> bool:
    """Read the legacy --use-sz-conserved request from args or normalized reductions."""
    if args is not None and getattr(args, "use_sz_conserved", None) is not None:
        return bool(getattr(args, "use_sz_conserved"))
    if use_sz_conserved is not None:
        return bool(use_sz_conserved)
    if isinstance(symmetry_reductions, dict):
        return bool(symmetry_reductions.get("use_sz_block", False))
    if isinstance(symmetry_reductions, (list, tuple, set)):
        return "sz" in {str(item).strip().lower() for item in symmetry_reductions}
    return False


def _dedupe_messages(messages: Iterable[Any]) -> List[str]:
    unique: List[str] = []
    for message in messages:
        text = str(message)
        if text and text not in unique:
            unique.append(text)
    return unique


def _is_yao_lee_spin_orbital_half(model_spec: Any) -> bool:
    return (
        str(getattr(model_spec, "model_family", "")).strip().lower() == "yao_lee"
        and str(getattr(model_spec, "orbital_rep", "")).strip() == "1/2"
    )


def _local_tz_charge_metadata(
    model_spec: Any | None = None,
    site_ops: Dict[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Return the physical-index U(1)_Tz charge map for PEPS tensors.

    Charges use the integer convention q = 2*Tz.  The named charge map is
    basis-order independent and matches the requested physics convention:
    orbital down has q=-1, orbital up has q=+1.
    """
    charges = list(LOCAL_TZ_CHARGES)
    if site_ops is not None and "Tz" in site_ops:
        tz = np.asarray(site_ops["Tz"], dtype=np.complex128)
        if tz.ndim == 2 and tz.shape[0] == tz.shape[1]:
            diagonal = np.diag(tz)
            if np.max(np.abs(tz - np.diag(diagonal))) <= 1.0e-12:
                parsed = [int(round(float(np.real(value)) * 2.0)) for value in diagonal]
                if len(parsed) == len(LOCAL_SPIN_ORBITAL_BASIS):
                    charges = parsed
    return {
        "charge_convention": "q = 2*Tz",
        "internal_basis_order": list(LOCAL_SPIN_ORBITAL_BASIS),
        "internal_named_basis_order": list(LOCAL_SPIN_ORBITAL_NAMED_BASIS),
        "physical_index_charges": [int(value) for value in charges],
        "physical_index_charge_map": {
            int(index): int(charge) for index, charge in enumerate(charges)
        },
        "basis_label_charge_map": {
            str(label): int(charge)
            for label, charge in zip(LOCAL_SPIN_ORBITAL_BASIS, charges)
        },
        "named_charge_map": dict(LOCAL_TZ_CHARGE_MAP),
        "requested_named_basis_order": list(LOCAL_TZ_REQUESTED_BASIS_ORDER),
        "requested_order_charges": [
            int(LOCAL_TZ_CHARGE_MAP[label]) for label in LOCAL_TZ_REQUESTED_BASIS_ORDER
        ],
        "model_family": None if model_spec is None else str(getattr(model_spec, "model_family", "")),
        "orbital_rep": None if model_spec is None else str(getattr(model_spec, "orbital_rep", "")),
    }


def _quimb_symmetric_tensor_capabilities() -> Dict[str, Any]:
    """Inspect whether this quimb install exposes PEPS SimpleUpdate U(1) hooks."""
    names = set(dir(qtn))
    symmetric_names = sorted(
        name
        for name in names
        if any(token in name.lower() for token in ("symm", "block", "abelian", "charge", "u1"))
    )
    try:
        peps_rand_signature = inspect.signature(qtn.PEPS.rand)
        peps_rand_parameters = list(peps_rand_signature.parameters)
    except Exception as exc:
        peps_rand_signature = None
        peps_rand_parameters = []
        peps_rand_error = f"{exc.__class__.__name__}: {exc}"
    else:
        peps_rand_error = None
    try:
        simple_update_signature = inspect.signature(qtn.SimpleUpdate)
        simple_update_parameters = list(simple_update_signature.parameters)
    except Exception as exc:
        simple_update_signature = None
        simple_update_parameters = []
        simple_update_error = f"{exc.__class__.__name__}: {exc}"
    else:
        simple_update_error = None

    charge_parameter_names = {
        "symmetry",
        "symmetry_mode",
        "charges",
        "charge_map",
        "phys_charges",
        "physical_charges",
        "bond_charges",
        "sector",
        "backend",
    }
    peps_rand_accepts_charges = any(name in charge_parameter_names for name in peps_rand_parameters)
    simple_update_accepts_charges = any(name in charge_parameter_names for name in simple_update_parameters)
    has_symmetric_tensor_names = bool(symmetric_names)
    explicit_simple_update_u1_hooks = bool(
        has_symmetric_tensor_names and peps_rand_accepts_charges and simple_update_accepts_charges
    )
    # Keep this false until this backend has a tested symmetric PEPS constructor
    # and SimpleUpdate path.  The charge map below prepares that integration
    # without pretending the dense quimb run is block sparse.
    simple_update_supports_u1_tz = False
    if explicit_simple_update_u1_hooks:
        reason = (
            "quimb exposes charge-related PEPS/SimpleUpdate parameters, but this backend has not yet "
            "wired and validated the symmetric PEPS constructor."
        )
    elif not has_symmetric_tensor_names:
        reason = "this quimb install does not expose obvious block-sparse or Abelian tensor classes through quimb.tensor."
    elif not peps_rand_accepts_charges:
        reason = "qtn.PEPS.rand does not expose charge or symmetry parameters for physical indices."
    else:
        reason = "qtn.SimpleUpdate does not expose charge or symmetry parameters."
    return {
        "backend_supports_symmetric_tensors": bool(has_symmetric_tensor_names),
        "backend_supports_u1_tz": bool(simple_update_supports_u1_tz),
        "backend_supports_s_z2": False,
        "explicit_simple_update_u1_hooks_detected": bool(explicit_simple_update_u1_hooks),
        "symmetric_tensor_names_sample": symmetric_names[:20],
        "peps_rand_parameters": peps_rand_parameters,
        "simple_update_parameters": simple_update_parameters,
        "peps_rand_signature": None if peps_rand_signature is None else str(peps_rand_signature),
        "simple_update_signature": None if simple_update_signature is None else str(simple_update_signature),
        "peps_rand_error": peps_rand_error,
        "simple_update_error": simple_update_error,
        "u1_tz_support_reason": reason,
    }


def _normalise_peps_symmetry_mode(mode: str | None) -> str:
    text = str(mode if mode is not None else "auto").strip().lower().replace("-", "_")
    aliases = {
        "dense": "none",
        "off": "none",
        "false": "none",
        "u1": "u1_tz",
        "tz": "u1_tz",
        "u1tz": "u1_tz",
        "u1_tz": "u1_tz",
        "u1+z2": "u1_tz_z2",
        "u1_tz+z2": "u1_tz_z2",
        "u1_tz_z2": "u1_tz_z2",
        "auto": "auto",
        "none": "none",
    }
    normalized = aliases.get(text, text)
    if normalized not in PEPS_SYMMETRY_MODE_OPTIONS:
        raise ValueError(
            f"Unsupported PEPS/iPEPS symmetry mode '{mode}'. "
            f"Choose from: {', '.join(PEPS_SYMMETRY_MODE_OPTIONS)}."
        )
    return normalized


def _normalise_ipeps_contraction_method(method: str | None) -> str:
    text = str(method if method is not None else "ctmrg").strip().lower().replace("-", "_")
    aliases = {
        "auto": "ctmrg",
        "ctm": "ctmrg",
        "ctmrg": "ctmrg",
        "crtg": "ctmrg",
        "boundary": "boundary",
        "boundary_mps": "boundary",
        "ctmrg_boundary": "ctmrg",
    }
    normalized = aliases.get(text, text)
    if normalized not in ("ctmrg", "boundary"):
        raise ValueError(
            f"Unsupported iPEPS contraction method '{method}'. "
            "Choose from: auto, ctmrg, crtg, boundary."
        )
    return normalized


def _field_class_from_inserted_spin_terms(external_field_terms: Any) -> str:
    components = {"x": 0.0, "y": 0.0, "z": 0.0}
    for item in list(external_field_terms or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        coefficient, operator_name = item[0], str(item[1])
        if operator_name == "Sx":
            components["x"] += float(np.real_if_close(coefficient))
        elif operator_name == "Sy":
            components["y"] += float(np.real_if_close(coefficient))
        elif operator_name == "Sz":
            components["z"] += float(np.real_if_close(coefficient))
    active_axes = [axis for axis, value in components.items() if abs(float(value)) > ZERO_TOL]
    if not active_axes:
        return "none"
    if len(active_axes) == 1:
        return f"h{active_axes[0]}"
    values = [components[axis] for axis in ("x", "y", "z")]
    if (
        len(active_axes) == 3
        and abs(values[0] - values[1]) <= ZERO_TOL
        and abs(values[1] - values[2]) <= ZERO_TOL
    ):
        return "h111"
    return "generic"


def _field_class_from_shared_reductions(symmetry_reductions: Any) -> str | None:
    if not isinstance(symmetry_reductions, dict):
        return None
    candidates: List[Any] = [symmetry_reductions]
    model_selection = symmetry_reductions.get("model_aware_selection")
    if isinstance(model_selection, dict):
        candidates.insert(0, model_selection)
    for candidate in candidates:
        field_class = candidate.get("field_class")
        if field_class:
            return str(field_class)
        field_info = candidate.get("field")
        if isinstance(field_info, dict) and field_info.get("field_class"):
            return str(field_info["field_class"])
    return None


def _physical_s_z2_generator_for_field(field_class: str) -> str | None:
    if field_class in ("none", "perturbation_only", "hz"):
        return "Rz_pi"
    if field_class == "hx":
        return "Rx_pi"
    if field_class == "hy":
        return "Ry_pi"
    return None


def _operator_tz_transfer(operator_name: str) -> int | None:
    """Return the local Tz charge transfer for a one-site operator.

    The PEPS tensor symmetry is only U(1)_Tz. Spin operators are neutral. In
    the spin-orbital operator names produced by ``models.py``, orbital ladder
    suffixes carry charge transfer +/-2 in the local ``2*Tz`` convention.
    """
    op = str(operator_name)
    if op in ("Tx", "Ty") or op.endswith("Tx") or op.endswith("Ty"):
        return None
    if op == "Tp" or op.endswith("Tp"):
        return 2
    if op == "Tm" or op.endswith("Tm"):
        return -2
    return 0


def _tz_neutrality_report_for_terms(
    *,
    bond_terms_by_bond: Iterable[Tuple[int, int, str, Iterable[Tuple[complex, str, str]]]],
    external_field_terms: Any = None,
) -> Dict[str, Any]:
    checked_two_site_terms = 0
    checked_one_site_terms = 0
    violations: List[Dict[str, Any]] = []
    for site_i, site_j, gamma, bond_terms in bond_terms_by_bond:
        for coefficient, op_i, op_j in bond_terms:
            if abs(complex(coefficient)) <= ZERO_TOL:
                continue
            checked_two_site_terms += 1
            transfer_i = _operator_tz_transfer(str(op_i))
            transfer_j = _operator_tz_transfer(str(op_j))
            if transfer_i is None or transfer_j is None or (transfer_i + transfer_j) != 0:
                violations.append(
                    {
                        "kind": "two_site",
                        "bond": [int(site_i), int(site_j)],
                        "gamma": str(gamma),
                        "coefficient": float(np.real_if_close(coefficient)),
                        "op_i": str(op_i),
                        "op_j": str(op_j),
                        "tz_transfer_i": transfer_i,
                        "tz_transfer_j": transfer_j,
                    }
                )
    for item in list(external_field_terms or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        coefficient, operator_name = item[0], str(item[1])
        if abs(float(np.real_if_close(coefficient))) <= ZERO_TOL:
            continue
        checked_one_site_terms += 1
        transfer = _operator_tz_transfer(operator_name)
        if transfer not in (0,):
            violations.append(
                {
                    "kind": "one_site",
                    "coefficient": float(np.real_if_close(coefficient)),
                    "operator": operator_name,
                    "tz_transfer": transfer,
                }
            )
    return {
        "all_terms_tz_neutral": len(violations) == 0,
        "checked_two_site_terms": int(checked_two_site_terms),
        "checked_one_site_terms": int(checked_one_site_terms),
        "violating_terms": violations,
        "convention": "Orbital ladder transfers use local 2*Tz charges: Tp=+2, Tm=-2.",
    }


def validate_quimb_peps_tz_neutrality(hamiltonian: Any) -> Dict[str, Any]:
    """Return the Tz-neutrality metadata attached to a quimb PEPS Hamiltonian."""
    report = getattr(hamiltonian, "yl_tz_neutrality", None)
    if isinstance(report, dict):
        return dict(report)
    metadata = getattr(hamiltonian, "yl_metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("tz_neutrality"), dict):
        return dict(metadata["tz_neutrality"])
    return {
        "all_terms_tz_neutral": False,
        "checked_two_site_terms": 0,
        "checked_one_site_terms": 0,
        "violating_terms": [],
        "reason": "Hamiltonian does not carry PEPS Tz-neutrality metadata.",
    }


def _u1_tz_sector_preservation_report(
    symmetry_report: Dict[str, Any],
    neutrality_report: Dict[str, Any],
) -> Dict[str, Any]:
    physical_request = bool(symmetry_report.get("physical_u1_tz_requested", False))
    charge_map_active = bool(symmetry_report.get("use_u1_tz", False))
    block_sparse = bool(
        charge_map_active
        and symmetry_report.get("backend_supports_block_sparse_u1_tz", False)
        and not symmetry_report.get("dense_fallback_used", False)
    )
    neutral_terms = bool(neutrality_report.get("all_terms_tz_neutral", False))
    return {
        "u1_tz_requested": physical_request,
        "u1_tz_charge_map_active": charge_map_active,
        "hamiltonian_terms_tz_neutral": neutral_terms,
        "symmetric_tensor_blocks_enforced": block_sparse,
        "block_sparse_tensor_blocks_enforced": block_sparse,
        "fixed_total_tz_sector_enforced": bool(physical_request and block_sparse and neutral_terms),
        "dense_fallback_used": bool(symmetry_report.get("dense_fallback_used", False)),
        "note": (
            "All gates are Tz-neutral and symmetric PEPS tensors enforce the fixed U(1)_Tz sector."
            if physical_request and block_sparse and neutral_terms
            else (
                "All gates are Tz-neutral, but dense PEPS tensors do not restrict the variational "
                "state to a single total Tz sector."
                if physical_request and neutral_terms
                else "U(1)_Tz PEPS symmetry is not active for this calculation."
            )
        ),
    }


def resolve_quimb_peps_symmetry_report(
    *,
    backend: str,
    requested_mode: str | None,
    model_spec: Any,
    external_field_terms: Any = None,
    symmetry_reductions: Any = None,
    strict: bool = True,
    allow_dense_fallback: bool = True,
    unit_cell_kind: str | None = None,
    legacy_use_sz_conserved: bool = False,
) -> Dict[str, Any]:
    """Resolve PEPS/iPEPS tensor-symmetry requests without ever using S^z."""
    mode = _normalise_peps_symmetry_mode(requested_mode)
    charge_metadata = _local_tz_charge_metadata(model_spec)
    quimb_capabilities = _quimb_symmetric_tensor_capabilities()
    field_class = _field_class_from_shared_reductions(symmetry_reductions)
    if field_class is None or field_class == "none":
        inserted_class = _field_class_from_inserted_spin_terms(external_field_terms)
        if inserted_class != "none":
            field_class = inserted_class
    field_class = str(field_class or "none")
    warnings: List[str] = []
    errors: List[str] = []
    use_u1_tz_requested = False
    use_s_z2_requested = False
    is_yao_lee_half = _is_yao_lee_spin_orbital_half(model_spec)

    if legacy_use_sz_conserved:
        warnings.append("PEPS/iPEPS tensor symmetry handling does not use total S^z; dropping legacy Sz request.")

    if mode == "auto":
        use_u1_tz_requested = bool(is_yao_lee_half)
    elif mode == "u1_tz":
        use_u1_tz_requested = True
    elif mode == "u1_tz_z2":
        use_u1_tz_requested = True
        use_s_z2_requested = True

    if use_u1_tz_requested and not is_yao_lee_half:
        warnings.append(
            "U(1)_Tz PEPS/iPEPS symmetry is only defined for model_family='yao_lee' with orbital_rep='1/2'; "
            "using dense tensors."
        )
        use_u1_tz_requested = False
        use_s_z2_requested = False

    z2_generator = _physical_s_z2_generator_for_field(field_class) if use_s_z2_requested else None
    reported_z2_generator = z2_generator
    if use_s_z2_requested and z2_generator is None:
        if field_class == "h111":
            warnings.append("Pure spin-sector Z2 is not conserved for H_[111]; using U1_Tz only.")
        else:
            warnings.append(f"Pure spin-sector Z2 is not conserved for field_class={field_class}; using U1_Tz only.")
        use_s_z2_requested = False
    elif use_s_z2_requested and z2_generator in ("Rx_pi", "Ry_pi"):
        message = (
            f"Spin-sector Z2 generator {z2_generator} would require rotated/non-diagonal PEPS tensors; "
            "only Rz_pi is supported by the first implementation."
        )
        if bool(allow_dense_fallback):
            warnings.append(message)
            use_s_z2_requested = False
        else:
            errors.append(message)
    elif use_s_z2_requested and z2_generator == "Rz_pi":
        message = "Current PEPS/iPEPS backend does not support finite-group/Z2 symmetric tensors for Rz_pi."
        if bool(strict):
            errors.append(message)
        else:
            warnings.append(f"{message} Using U1_Tz only.")
            use_s_z2_requested = False

    backend_supports_symmetric_tensors = bool(
        quimb_capabilities.get("backend_supports_symmetric_tensors", False)
    )
    backend_supports_u1_tz = bool(quimb_capabilities.get("backend_supports_u1_tz", False))
    backend_supports_s_z2 = bool(quimb_capabilities.get("backend_supports_s_z2", False))
    dense_fallback_used = False
    use_u1_tz = False
    use_s_z2 = False
    accepted_mode = "none"

    if use_u1_tz_requested:
        if backend_supports_u1_tz:
            use_u1_tz = True
            accepted_mode = "u1_tz"
        elif bool(allow_dense_fallback):
            warnings.append(
                f"{DENSE_PEPS_SYMMETRY_WARNING} "
                f"Prepared physical charge_map for q=2*Tz, but SimpleUpdate will not enforce tensor blocks: "
                f"{quimb_capabilities.get('u1_tz_support_reason', 'no SimpleUpdate U1_Tz support detected')}"
            )
            dense_fallback_used = True
            use_u1_tz = True
            accepted_mode = "u1_tz_dense_charge_neutral"
        else:
            errors.append(
                f"{DENSE_PEPS_SYMMETRY_WARNING} "
                f"{quimb_capabilities.get('u1_tz_support_reason', 'no SimpleUpdate U1_Tz support detected')}"
            )

    if use_s_z2_requested and backend_supports_s_z2:
        use_s_z2 = True
        accepted_mode = "u1_tz_z2" if use_u1_tz else "z2"

    if errors:
        raise NotImplementedError("; ".join(_dedupe_messages(errors)))

    report: Dict[str, Any] = {
        "requested_mode": mode,
        "accepted_mode": accepted_mode,
        "use_u1_tz": bool(use_u1_tz),
        "use_s_z2": bool(use_s_z2),
        "z2_generator": reported_z2_generator,
        "physical_u1_tz_requested": bool(use_u1_tz_requested),
        "physical_s_z2_requested": bool(use_s_z2_requested),
        "requested_s_z2": bool(mode == "u1_tz_z2"),
        "field_class": field_class,
        "backend": str(backend),
        "backend_supports_symmetric_tensors": bool(backend_supports_symmetric_tensors),
        "backend_supports_u1_tz": bool(backend_supports_u1_tz),
        "backend_supports_block_sparse_u1_tz": bool(backend_supports_u1_tz),
        "backend_supports_s_z2": bool(backend_supports_s_z2),
        "u1_tz_reduces_tensor_cost": bool(backend_supports_u1_tz and use_u1_tz),
        "u1_tz_charge_map_active": bool(use_u1_tz),
        "dense_fallback_used": bool(dense_fallback_used),
        "quimb_capabilities": quimb_capabilities,
        "warnings": _dedupe_messages(warnings),
        "errors": [],
        "local_physical_basis": list(LOCAL_SPIN_ORBITAL_BASIS),
        "u1_tz_physical_charges": list(LOCAL_TZ_CHARGES),
        "u1_tz_charge_map": charge_metadata,
        "prepared_u1_tz_charge_map": bool(use_u1_tz_requested and is_yao_lee_half),
        "simple_update_preserves_fixed_tz_sector": bool(use_u1_tz and not dense_fallback_used),
        "simple_update_sector_note": (
            "Symmetric tensors enforce a fixed U(1)_Tz sector and reduce tensor cost."
            if use_u1_tz and not dense_fallback_used
            else (
                "Dense SimpleUpdate now uses the U(1)_Tz charge map and verifies Tz-neutral gates, "
                "but this quimb API does not expose block-sparse tensors, so it does not reduce "
                "tensor cost or strictly project random tensors into one total Tz sector."
            )
        ),
        "tensor_charge_convention": "If symmetric tensors are enabled later: q_s + q_l + q_u - q_r - q_d = 0.",
    }
    if unit_cell_kind is not None:
        report["unit_cell_kind"] = str(unit_cell_kind)
        report["unit_cell_candidates"] = list(IPEPS_UNIT_CELL_CANDIDATES)
    return report


def _real_or_complex_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.size == 0:
        return array
    if float(np.max(np.abs(np.imag(array)))) <= 1.0e-13:
        return np.asarray(np.real(array), dtype=np.float64)
    return array


def _physical_dim_from_model_spec(
    model_spec: Any,
    site_ops: Dict[str, np.ndarray] | None = None,
) -> int:
    """Return the one-site Hilbert dimension for the selected model."""
    value = getattr(model_spec, "physical_dim", None)
    if value is not None:
        parsed = int(value)
        if parsed > 0:
            return parsed
    if site_ops is None:
        site_ops = build_site_ops(model_spec)
    ident = np.asarray(site_ops.get("Id"))
    if ident.ndim == 2 and ident.shape[0] == ident.shape[1] and ident.shape[0] > 0:
        return int(ident.shape[0])
    raise ValueError("Could not infer a positive PEPS physical dimension from model_spec.")


def _validate_site_operator_dimensions(site_ops: Dict[str, np.ndarray], local_dim: int) -> None:
    expected = (int(local_dim), int(local_dim))
    for name, operator in site_ops.items():
        shape = np.asarray(operator).shape
        if shape != expected:
            raise ValueError(
                f"Local operator '{name}' has shape {shape}, expected {expected} "
                f"for PEPS physical dimension d={int(local_dim)}."
            )


def _two_site_matrix_for_quimb(operator: np.ndarray, local_dim: int, label: str) -> np.ndarray:
    """Normalize a two-site operator to quimb's flattened ``(d^2, d^2)`` matrix.

    quimb's ``LocalHam2D`` and ``SimpleUpdate`` reliably accept two-site
    Hamiltonian terms as matrices.  The matrix convention used here is
    ``(site1_out, site2_out, site1_in, site2_in)`` when reshaped to rank 4.
    Diagnostic gate tensors are produced separately in the input-first order
    expected by quimb's lower-level gate routines.
    """
    d = int(local_dim)
    matrix_shape = (d * d, d * d)
    tensor_shape = (d, d, d, d)
    array = np.asarray(operator, dtype=np.complex128)
    if array.shape == matrix_shape:
        return _real_or_complex_matrix(array)
    if array.shape == tensor_shape:
        # Treat explicit rank-4 input as (site1_in, site2_in, site1_out, site2_out),
        # then convert back to the standard matrix convention.
        return _real_or_complex_matrix(np.transpose(array, (2, 3, 0, 1)).reshape(matrix_shape))
    raise ValueError(
        f"{label} has shape {array.shape}; expected {matrix_shape} or {tensor_shape} "
        f"for PEPS physical dimension d={d}."
    )


def _gate_tensor_input_first_from_matrix(matrix: np.ndarray, local_dim: int) -> np.ndarray:
    """Return a rank-4 diagnostic gate tensor in quimb input-first axis order.

    Axis order: ``(site1_in, site2_in, site1_out, site2_out)``.
    """
    d = int(local_dim)
    matrix_shape = (d * d, d * d)
    array = np.asarray(matrix, dtype=np.complex128)
    if array.shape != matrix_shape:
        array = _two_site_matrix_for_quimb(array, d, "diagnostic two-site gate")
    return array.reshape(d, d, d, d).transpose(2, 3, 0, 1)


def _two_site_gate_for_quimb(operator: np.ndarray, local_dim: int, label: str) -> np.ndarray:
    """Normalize a two-site operator to quimb's explicit input-first gate shape."""
    matrix = _two_site_matrix_for_quimb(operator, local_dim, label)
    return _real_or_complex_matrix(_gate_tensor_input_first_from_matrix(matrix, local_dim))


def _two_site_dim_from_shape(shape: Tuple[int, ...]) -> int | None:
    if len(shape) == 4 and len(set(int(value) for value in shape)) == 1:
        dim = int(shape[0])
        return dim if dim > 0 else None
    if len(shape) == 2 and int(shape[0]) == int(shape[1]):
        dim = int(round(np.sqrt(int(shape[0]))))
        if dim > 0 and dim * dim == int(shape[0]):
            return dim
    return None


def _hamiltonian_term_mapping(hamiltonian: Any) -> Dict[Any, Any]:
    for attr_name in ("yl_h2_terms", "terms"):
        terms = getattr(hamiltonian, attr_name, None)
        if isinstance(terms, dict) and terms:
            return terms
    values = getattr(hamiltonian, "values", None)
    if callable(values):
        return {index: term for index, term in enumerate(values())}
    return {}


def _physical_dim_from_hamiltonian(hamiltonian: Any) -> int:
    metadata = getattr(hamiltonian, "yl_metadata", {})
    model_spec = getattr(hamiltonian, "yl_model_spec", None) or metadata.get("model_spec")
    if model_spec is not None:
        return _physical_dim_from_model_spec(model_spec)
    value = metadata.get("local_dim")
    if value is not None:
        parsed = int(value)
        if parsed > 0:
            return parsed
    for term in _hamiltonian_term_mapping(hamiltonian).values():
        dim = _two_site_dim_from_shape(tuple(np.asarray(term).shape))
        if dim is not None:
            return int(dim)
    raise ValueError("Could not infer PEPS physical dimension from model_spec or Hamiltonian terms.")


def _validate_hamiltonian_term_dimensions(terms: Dict[Any, Any], local_dim: int) -> None:
    for key, term in terms.items():
        shape = tuple(np.asarray(term).shape)
        dim = _two_site_dim_from_shape(shape)
        if dim != int(local_dim):
            raise ValueError(
                f"PEPS Hamiltonian term {key!r} has shape {shape}; expected a two-site "
                f"operator compatible with physical dimension d={int(local_dim)}."
            )


def _validate_quimb_where_terms(
    terms: Dict[Any, Any],
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check explicit two-site terms and record periodic long-range gates."""
    lx = int(metadata.get("Lx", 0))
    ly = int(metadata.get("Ly", 0))
    coord_to_site = {
        tuple(coord): int(site)
        for coord, site in dict(metadata.get("coord_to_site", {})).items()
    }
    cyclic_x, cyclic_y = tuple(metadata.get("cyclic", (False, False)))
    diagnostics: List[Dict[str, Any]] = []
    long_range_terms: List[Dict[str, Any]] = []
    for where in terms:
        if where is None:
            continue
        if not isinstance(where, tuple) or len(where) != 2:
            raise ValueError(f"quimb PEPS Hamiltonian key {where!r} is not a two-site where tuple.")
        coords = tuple(tuple(int(part) for part in coord) for coord in where)
        for coord in coords:
            if coord not in coord_to_site:
                raise ValueError(
                    f"quimb PEPS Hamiltonian where={where!r} references coordinate {coord!r}, "
                    "which is not present in the PEPS state coordinate map."
                )
            x_coord, y_coord = int(coord[0]), int(coord[1])
            if x_coord < 0 or y_coord < 0 or x_coord >= lx or y_coord >= ly:
                raise ValueError(
                    f"quimb PEPS Hamiltonian where={where!r} is outside PEPS bounds "
                    f"Lx={lx}, Ly={ly}."
                )
        dx = abs(int(coords[0][0]) - int(coords[1][0]))
        dy = abs(int(coords[0][1]) - int(coords[1][1]))
        if bool(cyclic_x) and lx > 0:
            dx = min(dx, lx - dx)
        if bool(cyclic_y) and ly > 0:
            dy = min(dy, ly - dy)
        manhattan_distance = int(dx + dy)
        if manhattan_distance != 1:
            long_range_terms.append(
                {
                    "where": [list(coord) for coord in coords],
                    "sites": [coord_to_site.get(coord) for coord in coords],
                    "grid_delta": [int(dx), int(dy)],
                    "manhattan_distance": manhattan_distance,
                }
            )
        diagnostics.append(
            {
                "where": [list(coord) for coord in coords],
                "sites": [coord_to_site[coord] for coord in coords],
                "manhattan_distance": int(manhattan_distance),
                "uses_long_range_swap_gate": bool(manhattan_distance != 1),
            }
        )
    metadata["long_range_terms"] = long_range_terms
    metadata["long_range_gate_strategy"] = (
        "quimb SimpleUpdate long_range_use_swaps=True"
        if long_range_terms
        else "nearest_neighbour_gates"
    )
    return diagnostics


def _quimb_gate_diagnostics(
    state: Any,
    hamiltonian: Any,
    terms: Dict[Any, Any],
    local_dim: int,
) -> Dict[str, Any]:
    """Build concise diagnostics for quimb gate shape and where consistency."""
    metadata = getattr(hamiltonian, "yl_metadata", {})
    where_records = _validate_quimb_where_terms(terms, metadata)
    sample_key = next((key for key in terms if key is not None), None)
    if sample_key is None:
        sample_key = next(iter(terms), None)
    sample_term = np.asarray(
        terms.get(sample_key, np.zeros((local_dim * local_dim, local_dim * local_dim))),
        dtype=np.complex128,
    )
    sample_matrix = _two_site_matrix_for_quimb(sample_term, local_dim, f"PEPS sample term {sample_key}")
    sample_gate = _gate_tensor_input_first_from_matrix(sample_matrix, local_dim)
    try:
        state_phys_dim = int(state.phys_dim(0, 0))
    except Exception:
        state_phys_dim = int(local_dim)
    if state_phys_dim != int(local_dim):
        raise ValueError(
            f"quimb PEPS state physical dimension d={state_phys_dim} does not match "
            f"Hamiltonian local dimension d={int(local_dim)}."
        )
    try:
        state_site_labels = list(getattr(state, "site_labels"))
    except Exception:
        try:
            state_site_labels = [tuple(coo) for coo in state.gen_site_coos()]
        except Exception:
            state_site_labels = []
    physical_site_labels = list(metadata.get("physical_site_labels", []))
    coord_to_label = dict(metadata.get("coord_to_label", {}))
    long_range_terms = list(metadata.get("long_range_terms", []))
    return {
        "phys_dim": int(local_dim),
        "state_phys_dim": int(state_phys_dim),
        "term_count": int(len(terms)),
        "explicit_where_count": int(len(where_records)),
        "long_range_term_count": int(len(long_range_terms)),
        "long_range_terms_sample": long_range_terms[: min(3, len(long_range_terms))],
        "long_range_gate_strategy": metadata.get("long_range_gate_strategy", "nearest_neighbour_gates"),
        "state_site_labels_sample": [tuple(label) for label in state_site_labels[:8]],
        "physical_site_labels_sample": [tuple(label) for label in physical_site_labels[:8]],
        "ham_terms_key_sample": list(terms.keys())[:8],
        "coord_to_label_sample": [
            {"coord": tuple(coord), "label": tuple(label)}
            for coord, label in list(coord_to_label.items())[:8]
        ],
        "sample_where": (
            None
            if sample_key is None
            else [[int(value) for value in coord] for coord in sample_key]
            if sample_key is not None
            and isinstance(sample_key, tuple)
            and len(sample_key) == 2
            else str(sample_key)
        ),
        "sample_matrix_shape": tuple(int(value) for value in sample_matrix.shape),
        "sample_gate_tensor_shape": tuple(int(value) for value in sample_gate.shape),
        "sample_gate_axis_order": "site1_in, site2_in, site1_out, site2_out",
        "where_sample": where_records[: min(3, len(where_records))],
    }


def _format_quimb_gate_diagnostic_message(diagnostics: Dict[str, Any]) -> str:
    return (
        f"phys_dim={diagnostics.get('phys_dim')}, "
        f"state_phys_dim={diagnostics.get('state_phys_dim')}, "
        f"sample_where={diagnostics.get('sample_where')}, "
        f"matrix_shape={diagnostics.get('sample_matrix_shape')}, "
        f"gate_tensor_shape={diagnostics.get('sample_gate_tensor_shape')}, "
        f"axis_order={diagnostics.get('sample_gate_axis_order')}, "
        f"state_labels_sample={diagnostics.get('state_site_labels_sample')}, "
        f"ham_keys_sample={diagnostics.get('ham_terms_key_sample')}, "
        f"long_range_terms={diagnostics.get('long_range_term_count')}, "
        f"gate_strategy={diagnostics.get('long_range_gate_strategy')}"
    )


def _combine_operator_terms(
    terms: Iterable[Tuple[complex, str, str]],
    site_ops: Dict[str, np.ndarray],
    local_dim: int,
) -> np.ndarray:
    h2 = np.zeros((local_dim * local_dim, local_dim * local_dim), dtype=np.complex128)
    for coefficient, op_i, op_j in terms:
        coeff = complex(coefficient)
        if abs(coeff) <= ZERO_TOL:
            continue
        try:
            left = np.asarray(site_ops[str(op_i)], dtype=np.complex128)
            right = np.asarray(site_ops[str(op_j)], dtype=np.complex128)
        except KeyError as exc:
            raise KeyError(f"Unknown local operator in PEPS Hamiltonian term: {exc}") from exc
        h2 = h2 + coeff * np.kron(left, right)
    return _real_or_complex_matrix(h2)


def _combine_onsite_terms(
    external_field_terms: Any,
    site_ops: Dict[str, np.ndarray],
    local_dim: int,
) -> np.ndarray | None:
    h1 = np.zeros((local_dim, local_dim), dtype=np.complex128)
    used = False
    for item in list(external_field_terms or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        coefficient, op_name = item[0], str(item[1])
        coeff = complex(coefficient)
        if abs(coeff) <= ZERO_TOL:
            continue
        if op_name not in site_ops:
            raise KeyError(f"Unknown local operator in PEPS onsite term: {op_name}")
        h1 = h1 + coeff * np.asarray(site_ops[op_name], dtype=np.complex128)
        used = True
    return _real_or_complex_matrix(h1) if used else None


def _infer_lattice_name(geometry: Any, lattice_name: str | None = None) -> str:
    if lattice_name is not None:
        text = str(lattice_name).strip().lower()
        if text:
            return text
    sublattices = [int(value) for value in getattr(geometry, "sublattice_indices", [])]
    if sublattices and max(sublattices) > 0:
        return "honeycomb"
    return "square"


def _site_coordinate_maps(
    geometry: Any,
    lattice_name: str,
) -> Tuple[int, int, Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Map project geometry site ids onto quimb PEPS coordinates."""
    lattice_key = str(lattice_name).strip().lower()
    length_x = int(getattr(geometry, "length_x", 0) or 0)
    length_y = int(getattr(geometry, "length_y", 0) or 0)
    if length_x <= 0 or length_y <= 0:
        cell_indices = list(getattr(geometry, "cell_indices", []))
        if cell_indices:
            length_x = max(int(cell[0]) for cell in cell_indices) + 1
            length_y = max(int(cell[1]) for cell in cell_indices) + 1
    if length_x <= 0 or length_y <= 0:
        raise ValueError("Geometry must provide positive length_x and length_y for PEPS mapping.")

    site_to_coord: Dict[int, Tuple[int, int]] = {}
    if lattice_key == "honeycomb":
        # Use a skew brick-wall embedding:
        #   B(x, y) -> (x - y + shift, x + y)
        #   A(x, y) -> (x - y + shift, x + y + 1)
        # With the geometry convention A=sub=0, B=sub=1 this makes all three
        # honeycomb bonds nearest-neighbour PEPS bonds:
        #   z: A(x,y)-B(x,y), y: A(x,y)-B(x,y+1), x: A(x,y)-B(x+1,y).
        # The original unique honeycomb label is still stored as (x, y, sub).
        shift = length_y - 1
        for site, (cell, sublattice) in enumerate(
            zip(getattr(geometry, "cell_indices", []), getattr(geometry, "sublattice_indices", []))
        ):
            x_cell = int(cell[0])
            y_cell = int(cell[1])
            sub = int(sublattice)
            row = x_cell - y_cell + shift
            col = x_cell + y_cell + (1 if sub == 0 else 0)
            site_to_coord[int(site)] = (row, col)
        lx = length_x + length_y - 1
        ly = length_x + length_y
    elif lattice_key == "square":
        for site, cell in enumerate(getattr(geometry, "cell_indices", [])):
            site_to_coord[int(site)] = (int(cell[0]), int(cell[1]))
        lx = length_x
        ly = length_y
    else:
        raise ValueError(
            f"quimb_ipeps currently supports {', '.join(SUPPORTED_IPEPS_LATTICES)} lattices; "
            f"received lattice='{lattice_name}'."
        )

    if len(site_to_coord) != int(getattr(geometry, "number_of_sites", len(site_to_coord))):
        raise ValueError("Geometry site metadata is incomplete; cannot map all sites to PEPS coordinates.")
    coord_to_site = {coord: site for site, coord in site_to_coord.items()}
    if len(coord_to_site) != len(site_to_coord):
        raise ValueError("PEPS coordinate mapping is not one-to-one.")
    return int(lx), int(ly), site_to_coord, coord_to_site


def _site_label_maps(
    geometry: Any,
    lattice_name: str,
) -> Tuple[Dict[int, Tuple[int, ...]], Dict[Tuple[int, ...], int]]:
    """Return unique physical site labels, preserving honeycomb sublattice."""
    lattice_key = str(lattice_name).strip().lower()
    site_to_label: Dict[int, Tuple[int, ...]] = {}
    if lattice_key == "honeycomb":
        for site, (cell, sublattice) in enumerate(
            zip(getattr(geometry, "cell_indices", []), getattr(geometry, "sublattice_indices", []))
        ):
            site_to_label[int(site)] = (int(cell[0]), int(cell[1]), int(sublattice))
    else:
        for site, cell in enumerate(getattr(geometry, "cell_indices", [])):
            site_to_label[int(site)] = (int(cell[0]), int(cell[1]))
    label_to_site = {label: site for site, label in site_to_label.items()}
    if len(label_to_site) != len(site_to_label):
        raise ValueError("PEPS physical site labels are not one-to-one.")
    return site_to_label, label_to_site


def _ipeps_unit_cell_geometry(
    geometry: Any,
    lattice_name: str,
    unit_cell_kind: str | None,
) -> Any:
    """Choose the variational iPEPS unit cell independently of ED geometry."""
    lattice_key = str(lattice_name).strip().lower()
    kind = str(unit_cell_kind or "auto").strip().lower()
    if lattice_key == "honeycomb" and kind in ("auto", "minimal", "two_sublattice"):
        return build_lattice_geometry(
            "honeycomb",
            2,
            length_y=2,
            circumference_x=True,
            circumference_y=True,
        )
    return geometry


def _canonical_pair(
    coord_i: Tuple[int, int],
    coord_j: Tuple[int, int],
    h2_ij: np.ndarray,
) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray]:
    if tuple(coord_i) <= tuple(coord_j):
        return (tuple(coord_i), tuple(coord_j)), h2_ij
    local_dim2 = int(round(np.sqrt(h2_ij.shape[0])))
    h2_tensor = np.asarray(h2_ij).reshape(local_dim2, local_dim2, local_dim2, local_dim2)
    swapped = np.transpose(h2_tensor, (1, 0, 3, 2)).reshape(h2_ij.shape)
    return (tuple(coord_j), tuple(coord_i)), _real_or_complex_matrix(swapped)


def _local_hamiltonian_metadata(
    *,
    model_spec: Any,
    geometry: Any,
    lattice_name: str,
    lx: int,
    ly: int,
    local_dim: int,
    site_to_coord: Dict[int, Tuple[int, int]],
    coord_to_site: Dict[Tuple[int, int], int],
    site_to_label: Dict[int, Tuple[int, ...]],
    label_to_site: Dict[Tuple[int, ...], int],
    parameters: Dict[str, Any],
    ctm_chi: int,
) -> Dict[str, Any]:
    coord_to_label = {
        tuple(site_to_coord[site]): tuple(label)
        for site, label in site_to_label.items()
        if site in site_to_coord
    }
    return {
        "geometry": geometry,
        "model_spec": model_spec,
        "lattice": str(lattice_name),
        "Lx": int(lx),
        "Ly": int(ly),
        "local_dim": int(local_dim),
        "number_of_sites": int(getattr(geometry, "number_of_sites", len(site_to_coord))),
        "site_to_coord": {int(site): tuple(coord) for site, coord in site_to_coord.items()},
        "coord_to_site": {tuple(coord): int(site) for coord, site in coord_to_site.items()},
        "site_to_label": {int(site): tuple(label) for site, label in site_to_label.items()},
        "label_to_site": {tuple(label): int(site) for label, site in label_to_site.items()},
        "coord_to_label": {tuple(coord): tuple(label) for coord, label in coord_to_label.items()},
        "label_to_coord": {
            tuple(label): tuple(site_to_coord[site])
            for site, label in site_to_label.items()
            if site in site_to_coord
        },
        "physical_site_labels": [tuple(site_to_label[site]) for site in sorted(site_to_label)],
        "parameters": dict(parameters),
        "ctm_chi": int(ctm_chi),
        "cyclic": (
            bool(getattr(geometry, "circumference_x", False)),
            bool(getattr(geometry, "circumference_y", False)),
        ),
    }


def build_2d_local_hamiltonian(
    model_spec: Any,
    geometry: Any,
    *,
    lattice_name: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    coupling_j: float = DEFAULT_COUPLING_J,
    jx: float = DEFAULT_JX,
    jy: float = DEFAULT_JY,
    jz: float = DEFAULT_JZ,
    external_field_terms: Any = None,
    ctm_chi: int = DEFAULT_CTM_CHI,
) -> Any:
    """Build a ``qtn.LocalHam2D`` from the shared model and geometry objects.

    Honeycomb geometries are embedded as a brick-wall PEPS unit cell with
    coordinates ``A(x, y) -> (x, 2y)`` and ``B(x, y) -> (x, 2y + 1)``.  The
    resulting ``LocalHam2D`` gets a zero default nearest-neighbor term so the
    square-grid bonds that are not honeycomb bonds contribute exactly zero.
    """
    lattice_key = _infer_lattice_name(geometry, lattice_name)
    if not quimb_ipeps_supports_lattice(lattice_key):
        raise ValueError(
            f"quimb_ipeps currently supports {', '.join(SUPPORTED_IPEPS_LATTICES)} lattices; "
            f"received lattice='{lattice_key}'."
    )
    lx, ly, site_to_coord, coord_to_site = _site_coordinate_maps(geometry, lattice_key)
    site_to_label, label_to_site = _site_label_maps(geometry, lattice_key)
    site_ops = build_site_ops(model_spec)
    local_dim = _physical_dim_from_model_spec(model_spec, site_ops)
    _validate_site_operator_dimensions(site_ops, local_dim)
    zero_h2 = np.zeros((local_dim, local_dim, local_dim, local_dim), dtype=np.float64)
    h2_terms: Dict[Any, np.ndarray] = {None: zero_h2}
    source_bond_terms: List[Tuple[int, int, str, List[Tuple[complex, str, str]]]] = []
    h1 = _combine_onsite_terms(external_field_terms, site_ops, local_dim)
    identity = np.eye(local_dim, dtype=np.complex128)
    site_degrees: Dict[int, int] = {}
    for bond in getattr(geometry, "bond_list", []):
        site_i = int(getattr(bond, "i", getattr(bond, "site_i", -1)))
        site_j = int(getattr(bond, "j", getattr(bond, "site_j", -1)))
        site_degrees[site_i] = int(site_degrees.get(site_i, 0) + 1)
        site_degrees[site_j] = int(site_degrees.get(site_j, 0) + 1)

    for bond in getattr(geometry, "bond_list", []):
        site_i = int(getattr(bond, "i", getattr(bond, "site_i", -1)))
        site_j = int(getattr(bond, "j", getattr(bond, "site_j", -1)))
        gamma = str(getattr(bond, "gamma", getattr(bond, "bond_type", "z"))).lower()
        if site_i not in site_to_coord or site_j not in site_to_coord:
            raise ValueError(f"Bond ({site_i}, {site_j}) references a site outside the PEPS coordinate map.")
        bond_terms = two_site_operator_terms_for_bond(
            gamma,
            model_spec,
            float(alpha),
            float(beta),
            float(coupling_j),
            jx=float(jx),
            jy=float(jy),
            jz=float(jz),
        )
        source_bond_terms.append((site_i, site_j, gamma, list(bond_terms)))
        h2_ij = _combine_operator_terms(bond_terms, site_ops, local_dim)
        if h1 is not None:
            deg_i = max(1, int(site_degrees.get(site_i, 1)))
            deg_j = max(1, int(site_degrees.get(site_j, 1)))
            h2_ij = _real_or_complex_matrix(
                np.asarray(h2_ij, dtype=np.complex128)
                + np.kron(np.asarray(h1, dtype=np.complex128) / float(deg_i), identity)
                + np.kron(identity, np.asarray(h1, dtype=np.complex128) / float(deg_j))
            )
        pair_key, pair_term = _canonical_pair(site_to_coord[site_i], site_to_coord[site_j], h2_ij)
        pair_term = _two_site_gate_for_quimb(pair_term, local_dim, f"PEPS two-site term {pair_key}")
        if pair_key in h2_terms:
            h2_terms[pair_key] = _real_or_complex_matrix(np.asarray(h2_terms[pair_key]) + np.asarray(pair_term))
        else:
            h2_terms[pair_key] = pair_term

    cyclic = (
        bool(getattr(geometry, "circumference_x", False)),
        bool(getattr(geometry, "circumference_y", False)),
    )
    hamiltonian = qtn.LocalHam2D(
        int(lx),
        int(ly),
        H2=h2_terms,
        H1=None,
        cyclic=cyclic,
    )
    hamiltonian.yl_h2_terms = h2_terms
    hamiltonian.yl_metadata = _local_hamiltonian_metadata(
        model_spec=model_spec,
        geometry=geometry,
        lattice_name=lattice_key,
        lx=lx,
        ly=ly,
        local_dim=local_dim,
        site_to_coord=site_to_coord,
        coord_to_site=coord_to_site,
        site_to_label=site_to_label,
        label_to_site=label_to_site,
        parameters={
            "alpha": float(alpha),
            "beta": float(beta),
            "coupling_j": float(coupling_j),
            "jx": float(jx),
            "jy": float(jy),
            "jz": float(jz),
            "external_field_terms": list(external_field_terms or []),
            "orbital_is_trivial": bool(is_trivial_orbital(model_spec)),
        },
        ctm_chi=int(ctm_chi),
    )
    hamiltonian.yl_metadata["u1_tz_charge_map"] = _local_tz_charge_metadata(model_spec, site_ops)
    hamiltonian.yl_tz_neutrality = _tz_neutrality_report_for_terms(
        bond_terms_by_bond=source_bond_terms,
        external_field_terms=external_field_terms,
    )
    hamiltonian.yl_metadata["tz_neutrality"] = dict(hamiltonian.yl_tz_neutrality)
    hamiltonian.yl_model_spec = model_spec
    return hamiltonian


def optimize_ipeps_simple_update(
    hamiltonian: Any,
    D: int,
    tau: float | Iterable[float],
    steps: int | Iterable[int],
    *,
    seed: int | None = None,
    chi: int | None = None,
    cutoff: float = 1.0e-10,
    progbar: bool = True,
    symmetry_report: Dict[str, Any] | None = None,
) -> Any:
    """Optimize a random PEPS/iPEPS unit cell with Simple Update.

    quimb's ``SimpleUpdate`` operates on a finite PEPS unit cell.  With cyclic
    PEPS boundaries this is the usual practical representation of an iPEPS
    unit cell; the boundary contraction used for energy measurement is governed
    by ``chi``.
    """
    metadata = getattr(hamiltonian, "yl_metadata", {})
    lx = int(getattr(hamiltonian, "Lx", metadata.get("Lx", 0)))
    ly = int(getattr(hamiltonian, "Ly", metadata.get("Ly", 0)))
    local_dim = _physical_dim_from_hamiltonian(hamiltonian)
    cyclic = metadata.get("cyclic", False)
    max_bond = max(1, int(D))
    su_chi = int(chi if chi is not None else max(8, max_bond * max_bond))
    tau_values = [float(value) for value in (tau if isinstance(tau, (list, tuple)) else [tau])]
    step_values = _as_step_list(steps, len(tau_values))
    if len(step_values) < len(tau_values):
        step_values.extend([step_values[-1]] * (len(tau_values) - len(step_values)))

    state = qtn.PEPS.rand(
        lx,
        ly,
        bond_dim=max_bond,
        phys_dim=local_dim,
        dtype="complex128",
        seed=seed,
        cyclic=cyclic,
    )
    try:
        state.site_labels = [tuple(coo) for coo in state.gen_site_coos()]
    except Exception:
        state.site_labels = [(i, j) for i in range(lx) for j in range(ly)]
    state.yl_physical_site_labels = list(metadata.get("physical_site_labels", []))
    state.yl_coord_to_label = dict(metadata.get("coord_to_label", {}))
    charge_map = metadata.get("u1_tz_charge_map")
    if isinstance(charge_map, dict):
        state.yl_u1_tz_charge_map = dict(charge_map)
    if isinstance(symmetry_report, dict):
        state.yl_symmetry_report = dict(symmetry_report)
    hamiltonian_terms = _hamiltonian_term_mapping(hamiltonian)
    if hamiltonian_terms:
        _validate_hamiltonian_term_dimensions(hamiltonian_terms, local_dim)
    if hamiltonian_terms:
        debug_terms = dict(hamiltonian_terms)
    else:
        debug_terms = {None: np.zeros((local_dim * local_dim, local_dim * local_dim), dtype=np.float64)}
    gate_diagnostics = _quimb_gate_diagnostics(state, hamiltonian, debug_terms, local_dim)
    hamiltonian.yl_gate_diagnostics = gate_diagnostics
    if bool(progbar):
        print(
            "[quimb] gate diagnostic: "
            f"{_format_quimb_gate_diagnostic_message(gate_diagnostics)}"
        )
    su = qtn.SimpleUpdate(
        state,
        hamiltonian,
        tau=tau_values[0] if tau_values else DEFAULT_TAU,
        D=max_bond,
        chi=su_chi,
        long_range_use_swaps=bool(gate_diagnostics.get("long_range_term_count", 0)),
    )
    su.progbar = False
    su.compute_energy_final = False
    total_steps = int(sum(max(0, int(value)) for value in step_values))
    progress_bar = _make_progress_bar(
        enabled=bool(progbar) and total_steps > 0,
        total=total_steps,
        desc="quimb PEPS/iPEPS GS",
        unit="step",
        leave=False,
    )
    try:
        for tau_value, step_count in zip(tau_values, step_values):
            step_count = int(step_count)
            if step_count <= 0:
                continue
            tau_float = float(tau_value)
            if progress_bar is None:
                su.evolve(step_count, tau=tau_float, progbar=False)
                continue
            progress_bar.set_postfix({"tau": f"{tau_float:.3g}", "D": int(max_bond), "chi": int(su_chi)})
            for _step in range(step_count):
                su.evolve(1, tau=tau_float, progbar=False)
                progress_bar.update(1)
    except ValueError as exc:
        message = str(exc)
        if "axes" in message.lower() and "match" in message.lower():
            raise ValueError(
                "quimb SimpleUpdate gate application failed with an axis mismatch. "
                "The PEPS backend now passes LocalHam2D two-site terms as explicit "
                "rank-4 tensors in input-first order "
                "(site1_in, site2_in, site1_out, site2_out). "
                f"Gate diagnostics: {_format_quimb_gate_diagnostic_message(gate_diagnostics)}. "
                f"Original quimb error: {message}"
            ) from exc
        if "wrong number of inds" in message.lower() or "too many values to unpack" in message.lower():
            raise ValueError(
                "quimb SimpleUpdate gate application failed while applying a honeycomb bond. "
                "This usually means a Hamiltonian term key does not match the PEPS state labels "
                "or a bond was embedded as a long-range path. "
                "Honeycomb sites are now labelled by unique physical labels (x, y, sublattice) "
                "and embedded onto nearest-neighbour quimb coordinates. "
                f"Gate diagnostics: {_format_quimb_gate_diagnostic_message(gate_diagnostics)}. "
                f"Original quimb error: {message}"
            ) from exc
        raise
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if isinstance(getattr(su, "best", None), dict) and su.best.get("state") is not None:
        state = su.best["state"]
    else:
        state = su.state
    if isinstance(charge_map, dict):
        state.yl_u1_tz_charge_map = dict(charge_map)
    if isinstance(symmetry_report, dict):
        state.yl_symmetry_report = dict(symmetry_report)
    state.yl_simple_update = {
        "D": int(max_bond),
        "chi": int(su_chi),
        "tau_schedule": [float(value) for value in tau_values],
        "steps_schedule": [int(value) for value in step_values],
        "steps_done": int(getattr(su, "n", sum(step_values))),
        "best_energy_per_site": (
            float(su.best["energy"])
            if isinstance(getattr(su, "best", None), dict) and su.best.get("energy") is not None
            else None
        ),
        "energies": [float(value) for value in getattr(su, "energies", [])],
        "energy_iterations": [int(value) for value in getattr(su, "its", getattr(su, "energy_ns", []))],
        "gate_diagnostics": gate_diagnostics,
    }
    return state


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(np.real(value))
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _terms_without_default(terms: Dict[Any, Any]) -> Dict[Any, Any]:
    return {key: value for key, value in dict(terms).items() if key is not None}


def _kron_many(operators: Iterable[np.ndarray]) -> np.ndarray:
    iterator = iter(operators)
    try:
        result = np.asarray(next(iterator), dtype=np.complex128)
    except StopIteration:
        raise ValueError("At least one operator is required.")
    for op in iterator:
        result = np.kron(result, np.asarray(op, dtype=np.complex128))
    return _real_or_complex_matrix(result)


def _local_expectation_for_region(
    ipeps_state: Any,
    operator: np.ndarray,
    where: Tuple[Tuple[int, int], ...],
    max_bond: int,
) -> Any:
    """Contract one multi-site local operator using quimb's PEPS environments."""
    if hasattr(ipeps_state, "local_expectation"):
        try:
            return ipeps_state.local_expectation(
                operator,
                where,
                max_bond=int(max_bond),
                optimize="auto-hq",
                normalized=True,
            )
        except TypeError:
            return ipeps_state.local_expectation(
                operator,
                where,
                int(max_bond),
                "auto-hq",
                normalized=True,
            )

    try:
        return ipeps_state.compute_local_expectation(
            {where: operator},
            max_bond=int(max_bond),
            normalized=True,
        )
    except Exception:
        rho = ipeps_state.partial_trace(where, max_bond=int(max_bond), normalized=True)
        rho_matrix = np.asarray(rho, dtype=np.complex128).reshape(operator.shape)
        return np.trace(rho_matrix @ np.asarray(operator, dtype=np.complex128))


def _compute_energy_per_site(ipeps_state: Any, hamiltonian: Any, max_bond: int) -> Tuple[float | None, float | None]:
    terms = _terms_without_default(_hamiltonian_term_mapping(hamiltonian))
    if len(terms) == 0:
        return 0.0, 0.0
    energy_total = ipeps_state.compute_local_expectation(
        terms,
        max_bond=int(max_bond),
        normalized=True,
    )
    energy_total_float = _finite_float(energy_total)
    n_sites = int(getattr(hamiltonian, "nsites", 0) or getattr(hamiltonian, "yl_metadata", {}).get("number_of_sites", 1))
    energy_per_site = None if energy_total_float is None else energy_total_float / float(max(1, n_sites))
    return energy_per_site, energy_total_float


def _compute_honeycomb_plaquette_flux(ipeps_state: Any, hamiltonian: Any, max_bond: int) -> Dict[str, Any]:
    metadata = getattr(hamiltonian, "yl_metadata", {})
    lattice = str(metadata.get("lattice", "")).lower()
    if lattice != "honeycomb":
        return {
            "available": False,
            "value": None,
            "W_p": None,
            "reason": "Plaquette flux W_p is defined here only for honeycomb hexagons.",
        }
    geometry = metadata.get("geometry")
    if geometry is None:
        return {"available": False, "value": None, "W_p": None, "reason": "Missing source geometry."}
    plaquettes = honeycomb_plaquette_flux_operators(geometry)
    if len(plaquettes) == 0:
        return {"available": False, "value": None, "W_p": None, "reason": "No honeycomb plaquettes found."}
    site_to_coord = metadata.get("site_to_coord", {})
    site_ops = build_site_ops_from_hamiltonian_metadata(hamiltonian)
    all_fluxes: Dict[str, float] = {}
    plaquette_records: List[Dict[str, Any]] = []
    for plaquette in plaquettes:
        sites = [int(site) for site in plaquette["sites"]]
        coords = tuple(tuple(site_to_coord[int(site)]) for site in sites)
        operator = _kron_many(site_ops[str(op_name)] for op_name in plaquette["operator_names"])
        raw_value = _local_expectation_for_region(ipeps_state, operator, coords, max_bond)
        normalized_value = _finite_float(raw_value)
        if normalized_value is None:
            continue
        normalized_value *= float(plaquette.get("normalization", 1.0))
        key = str(int(plaquette["plaquette_index"]))
        all_fluxes[key] = float(normalized_value)
        plaquette_records.append(
            {
                "plaquette_index": int(plaquette["plaquette_index"]),
                "sites": sites,
                "coords": [list(coord) for coord in coords],
                "axes": [str(axis) for axis in plaquette["axes"]],
                "operators": [str(op_name) for op_name in plaquette["operator_names"]],
                "W_p": float(normalized_value),
                "value": float(normalized_value),
                "target": 1.0,
                "normalization": float(plaquette.get("normalization", 1.0)),
                "close_to_target": plaquette_flux_close_to_target(normalized_value),
            }
        )
    if not all_fluxes:
        return {
            "available": False,
            "value": None,
            "W_p": None,
            "reason": "CTMRG/boundary contraction did not return a finite plaquette flux.",
        }
    center_record = plaquette_records[len(plaquette_records) // 2]
    return {
        "available": True,
        "plaquette_index": int(center_record["plaquette_index"]),
        "sites": center_record["sites"],
        "axes": center_record["axes"],
        "operators": center_record["operators"],
        "W_p": float(center_record["W_p"]),
        "value": float(center_record["W_p"]),
        "target": 1.0,
        "normalization": float(center_record["normalization"]),
        "close_to_target": bool(center_record["close_to_target"]),
        "plaquettes": plaquette_records,
        "all_plaquette_fluxes": all_fluxes,
        "plaquette_flux_map": all_fluxes,
    }


def build_site_ops_from_hamiltonian_metadata(hamiltonian: Any) -> Dict[str, np.ndarray]:
    model_spec = getattr(getattr(hamiltonian, "yl_metadata", {}).get("geometry", None), "model_spec", None)
    if model_spec is not None:
        return build_site_ops(model_spec)
    model_spec = getattr(hamiltonian, "yl_model_spec", None)
    if model_spec is None:
        model_spec = getattr(hamiltonian, "model_spec", None)
    if model_spec is None:
        model_spec = getattr(hamiltonian, "yl_metadata", {}).get("model_spec")
    if model_spec is None:
        raise ValueError("Hamiltonian metadata is missing model_spec; cannot build plaquette-flux operators.")
    return build_site_ops(model_spec)


def evaluate_ipeps_observables(
    ipeps_state: Any,
    hamiltonian: Any,
    *,
    ctm_chi: int | None = None,
    contraction_method: str | None = "ctmrg",
) -> Dict[str, Any]:
    """Measure energy density and honeycomb plaquette flux using quimb.

    quimb's PEPS ``compute_local_expectation`` builds the needed local
    plaquette/boundary environments, the same contraction layer used by its
    CTMRG-style two-dimensional routines.
    """
    metadata = getattr(hamiltonian, "yl_metadata", {})
    max_bond = int(ctm_chi if ctm_chi is not None else metadata.get("ctm_chi", DEFAULT_CTM_CHI))
    contraction_key = _normalise_ipeps_contraction_method(contraction_method)
    energy_per_site, energy_total = _compute_energy_per_site(ipeps_state, hamiltonian, max_bond)
    try:
        plaquette_flux = _compute_honeycomb_plaquette_flux(ipeps_state, hamiltonian, max_bond)
    except Exception as exc:
        plaquette_flux = {
            "available": False,
            "value": None,
            "W_p": None,
            "reason": str(exc) or exc.__class__.__name__,
        }
    output = {
        "status": "completed",
        "backend": "quimb_ipeps",
        "energy_per_site": energy_per_site,
        "ground_state_energy_per_site": energy_per_site,
        "energy_total": energy_total,
        "plaquette_flux": plaquette_flux,
        "all_plaquette_fluxes": plaquette_flux.get("all_plaquette_fluxes", {}),
        "contraction": {
            "method": contraction_key,
            "engine": "quimb PEPS.compute_local_expectation boundary/CTMRG-style environments",
            "ctm_chi": int(max_bond),
            "normalized": True,
            "note": (
                "ctmrg/crtg selects quimb's boundary environment contraction with CTMRG-style chi control; "
                "boundary is kept as an explicit alias for the same dense contraction path in this backend."
            ),
        },
    }
    simple_update = getattr(ipeps_state, "yl_simple_update", None)
    if isinstance(simple_update, dict):
        output["simple_update"] = simple_update
        gate_diagnostics = simple_update.get("gate_diagnostics")
        if isinstance(gate_diagnostics, dict):
            output["gate_diagnostics"] = gate_diagnostics
    hamiltonian_gate_diagnostics = getattr(hamiltonian, "yl_gate_diagnostics", None)
    if isinstance(hamiltonian_gate_diagnostics, dict):
        output["gate_diagnostics"] = hamiltonian_gate_diagnostics
    return output


def _coord_for_site(hamiltonian: Any, site: int) -> Tuple[int, int]:
    metadata = getattr(hamiltonian, "yl_metadata", {})
    site_to_coord = metadata.get("site_to_coord", {})
    try:
        return tuple(site_to_coord[int(site)])
    except KeyError as exc:
        raise KeyError(f"PEPS Hamiltonian metadata has no coordinate for site {site}.") from exc


def _one_site_expectation(
    peps_state: Any,
    hamiltonian: Any,
    operator: np.ndarray,
    site: int,
    max_bond: int,
) -> complex:
    coord = _coord_for_site(hamiltonian, site)
    value = _local_expectation_for_region(peps_state, np.asarray(operator), (coord,), max_bond)
    return complex(value)


def _two_site_expectation(
    peps_state: Any,
    hamiltonian: Any,
    op_i: np.ndarray,
    site_i: int,
    op_j: np.ndarray,
    site_j: int,
    max_bond: int,
) -> complex:
    if int(site_i) == int(site_j):
        return _one_site_expectation(peps_state, hamiltonian, np.asarray(op_i) @ np.asarray(op_j), site_i, max_bond)
    coord_i = _coord_for_site(hamiltonian, site_i)
    coord_j = _coord_for_site(hamiltonian, site_j)
    operator = np.kron(np.asarray(op_i, dtype=np.complex128), np.asarray(op_j, dtype=np.complex128))
    value = _local_expectation_for_region(peps_state, operator, (coord_i, coord_j), max_bond)
    return complex(value)


def collect_local_observables_from_peps(
    peps_state: Any,
    hamiltonian: Any,
    *,
    model_spec: Any | None = None,
    ctm_chi: int | None = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Measure local spin and orbital vectors on every PEPS/iPEPS site."""
    metadata = getattr(hamiltonian, "yl_metadata", {})
    if model_spec is None:
        model_spec = getattr(hamiltonian, "yl_model_spec", None) or metadata.get("model_spec")
    if model_spec is None:
        raise ValueError("model_spec is required for PEPS local observables.")
    n_sites = int(metadata.get("number_of_sites", getattr(hamiltonian, "nsites", 0)))
    max_bond = int(ctm_chi if ctm_chi is not None else metadata.get("ctm_chi", DEFAULT_CTM_CHI))
    site_ops = build_site_ops(model_spec)
    axes = ("x", "y", "z")
    site_to_label = dict(metadata.get("site_to_label", {}))
    rows: List[Dict[str, Any]] = []
    spin_sum = np.zeros(3, dtype=float)
    orbital_sum = np.zeros(3, dtype=float)
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=max(0, n_sites) * 6,
        desc="PEPS local observables",
        unit="op",
        leave=False,
    )
    for site in range(n_sites):
        spin_vec: List[float] = []
        orbital_vec: List[float] = []
        for axis_index, axis in enumerate(axes):
            s_value = _finite_float(_one_site_expectation(peps_state, hamiltonian, site_ops[f"S{axis}"], site, max_bond))
            t_value = _finite_float(_one_site_expectation(peps_state, hamiltonian, site_ops[f"T{axis}"], site, max_bond))
            spin_component = 0.0 if s_value is None else float(s_value)
            orbital_component = 0.0 if t_value is None else float(t_value)
            spin_vec.append(spin_component)
            orbital_vec.append(orbital_component)
            spin_sum[axis_index] += spin_component
            orbital_sum[axis_index] += orbital_component
            if progress_bar is not None:
                progress_bar.update(2)
        rows.append(
            {
                "site": int(site),
                "label": list(site_to_label.get(int(site), (int(site),))),
                "S": {"x": spin_vec[0], "y": spin_vec[1], "z": spin_vec[2]},
                "T": {"x": orbital_vec[0], "y": orbital_vec[1], "z": orbital_vec[2]},
                "spin_vector": spin_vec,
                "orbital_vector": orbital_vec,
            }
        )
    if progress_bar is not None:
        progress_bar.close()
    denom = float(max(1, n_sites))
    return {
        "sites": rows,
        "uniform_spin_vector": [float(value / denom) for value in spin_sum],
        "uniform_orbital_vector": [float(value / denom) for value in orbital_sum],
        "spin_z_per_site": float(spin_sum[2] / denom),
        "orbital_z_per_site": float(orbital_sum[2] / denom),
        "operator_convention": "<S> and <T> are local spin/orbital expectation values.",
    }


def collect_resolved_bond_observables_from_peps(
    peps_state: Any,
    hamiltonian: Any,
    *,
    model_spec: Any | None = None,
    ctm_chi: int | None = None,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """Measure resolved spin/orbital dot products and bond energy per geometry bond."""
    metadata = getattr(hamiltonian, "yl_metadata", {})
    geometry = metadata.get("geometry")
    if geometry is None:
        raise ValueError("Hamiltonian metadata is missing geometry; cannot measure PEPS bond observables.")
    if model_spec is None:
        model_spec = getattr(hamiltonian, "yl_model_spec", None) or metadata.get("model_spec")
    if model_spec is None:
        raise ValueError("model_spec is required for PEPS bond observables.")
    max_bond = int(ctm_chi if ctm_chi is not None else metadata.get("ctm_chi", DEFAULT_CTM_CHI))
    site_ops = build_site_ops(model_spec)
    site_to_coord = dict(metadata.get("site_to_coord", {}))
    terms = _hamiltonian_term_mapping(hamiltonian)
    rows: List[Dict[str, Any]] = []
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=max(0, len(getattr(geometry, "bond_list", []))) * 8,
        desc="PEPS bond observables",
        unit="op",
        leave=False,
    )
    for bond_index, bond in enumerate(getattr(geometry, "bond_list", [])):
        site_i = int(getattr(bond, "i", getattr(bond, "site_i", -1)))
        site_j = int(getattr(bond, "j", getattr(bond, "site_j", -1)))
        gamma = str(getattr(bond, "gamma", getattr(bond, "bond_type", "z"))).lower()
        spin_components: Dict[str, float] = {}
        orbital_components: Dict[str, float] = {}
        for axis in ("x", "y", "z"):
            s_value = _finite_float(
                _two_site_expectation(
                    peps_state,
                    hamiltonian,
                    site_ops[f"S{axis}"],
                    site_i,
                    site_ops[f"S{axis}"],
                    site_j,
                    max_bond,
                )
            )
            t_value = _finite_float(
                _two_site_expectation(
                    peps_state,
                    hamiltonian,
                    site_ops[f"T{axis}"],
                    site_i,
                    site_ops[f"T{axis}"],
                    site_j,
                    max_bond,
                )
            )
            spin_components[axis] = 0.0 if s_value is None else float(s_value)
            orbital_components[axis] = 0.0 if t_value is None else float(t_value)
            if progress_bar is not None:
                progress_bar.update(2)
        coord_i = tuple(site_to_coord[int(site_i)])
        coord_j = tuple(site_to_coord[int(site_j)])
        pair_key = (coord_i, coord_j) if coord_i <= coord_j else (coord_j, coord_i)
        energy_value = None
        if pair_key in terms:
            raw_energy = _local_expectation_for_region(
                peps_state,
                np.asarray(terms[pair_key]),
                pair_key,
                max_bond,
            )
            energy_value = _finite_float(raw_energy)
            if progress_bar is not None:
                progress_bar.update(1)
        spin_dot = float(sum(spin_components.values()))
        orbital_dot = float(sum(orbital_components.values()))
        row = {
            "bond_index": int(bond_index),
            "i": int(site_i),
            "j": int(site_j),
            "gamma": gamma,
            "S": spin_dot,
            "T": orbital_dot,
            "spin_dot": spin_dot,
            "orbital_dot": orbital_dot,
            "spin_components": spin_components,
            "orbital_components": orbital_components,
            "spin_gamma": float(spin_components.get(gamma, 0.0)),
            "orbital_gamma": float(orbital_components.get(gamma, 0.0)),
            "total": None if energy_value is None else float(energy_value),
            "energy": None if energy_value is None else float(energy_value),
            "where": [list(coord_i), list(coord_j)],
            "measurement": "PEPS/iPEPS environment contraction",
        }
        rows.append(row)
    if progress_bar is not None:
        progress_bar.close()
    return rows


def _correlation_operator_channels(model_spec: Any) -> List[Tuple[str, str, str]]:
    channels: List[Tuple[str, str, str]] = []
    for axis in ("x", "y", "z"):
        channels.append((f"S{axis}_S{axis}", f"S{axis}", f"S{axis}"))
        channels.append((f"T{axis}_T{axis}", f"T{axis}", f"T{axis}"))
        channels.append((f"ST{axis}_ST{axis}", f"ST{axis}", f"ST{axis}"))
    if not is_trivial_orbital(model_spec):
        for spin_axis in ("x", "y", "z"):
            for orbital_axis in ("x", "y", "z"):
                channels.append(
                    (
                        f"S{spin_axis}T{orbital_axis}_S{spin_axis}T{orbital_axis}",
                        f"S{spin_axis}T{orbital_axis}",
                        f"S{spin_axis}T{orbital_axis}",
                    )
                )
    seen: set[str] = set()
    unique: List[Tuple[str, str, str]] = []
    for channel in channels:
        if channel[0] not in seen:
            unique.append(channel)
            seen.add(channel[0])
    return unique


def collect_correlation_matrices_from_peps(
    peps_state: Any,
    hamiltonian: Any,
    *,
    model_spec: Any | None = None,
    ctm_chi: int | None = None,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Collect the same two-point channels consumed by DMRG/ED analysis."""
    metadata = getattr(hamiltonian, "yl_metadata", {})
    if model_spec is None:
        model_spec = getattr(hamiltonian, "yl_model_spec", None) or metadata.get("model_spec")
    if model_spec is None:
        raise ValueError("model_spec is required for PEPS correlation channels.")
    n_sites = int(metadata.get("number_of_sites", getattr(hamiltonian, "nsites", 0)))
    site_ops = build_site_ops(model_spec)
    max_bond = int(ctm_chi if ctm_chi is not None else metadata.get("ctm_chi", DEFAULT_CTM_CHI))
    channels = _correlation_operator_channels(model_spec)
    correlations: Dict[str, np.ndarray] = {
        key: np.zeros((n_sites, n_sites), dtype=np.complex128)
        for key, _op_i, _op_j in channels
    }
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(channels) * n_sites * n_sites,
        desc="PEPS correlations",
        unit="term",
        leave=False,
    )
    for key, op_i_name, op_j_name in channels:
        op_i = site_ops[op_i_name]
        op_j = site_ops[op_j_name]
        matrix = correlations[key]
        for site_i in range(n_sites):
            for site_j in range(n_sites):
                matrix[site_i, site_j] = _two_site_expectation(
                    peps_state,
                    hamiltonian,
                    op_i,
                    site_i,
                    op_j,
                    site_j,
                    max_bond,
                )
                if progress_bar is not None:
                    progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return correlations


def collect_uniform_z_observables_from_peps(
    peps_state: Any,
    hamiltonian: Any,
    *,
    model_spec: Any | None = None,
    ctm_chi: int | None = None,
) -> Dict[str, float]:
    metadata = getattr(hamiltonian, "yl_metadata", {})
    if model_spec is None:
        model_spec = getattr(hamiltonian, "yl_model_spec", None) or metadata.get("model_spec")
    if model_spec is None:
        raise ValueError("model_spec is required for PEPS uniform observables.")
    n_sites = int(metadata.get("number_of_sites", getattr(hamiltonian, "nsites", 0)))
    max_bond = int(ctm_chi if ctm_chi is not None else metadata.get("ctm_chi", DEFAULT_CTM_CHI))
    site_ops = build_site_ops(model_spec)
    spin_z = 0.0j
    orbital_z = 0.0j
    for site in range(n_sites):
        spin_z += _one_site_expectation(peps_state, hamiltonian, site_ops["Sz"], site, max_bond)
        orbital_z += _one_site_expectation(peps_state, hamiltonian, site_ops["Tz"], site, max_bond)
    return {
        "spin_z_per_site": float(np.real(spin_z) / float(max(1, n_sites))),
        "orbital_z_per_site": float(np.real(orbital_z) / float(max(1, n_sites))),
    }


def _peps_dense_state_vector(peps_state: Any) -> np.ndarray:
    for method_name in ("to_dense", "to_dense_vector", "contract"):
        method = getattr(peps_state, method_name, None)
        if not callable(method):
            continue
        try:
            dense = method()
        except TypeError:
            continue
        array = np.asarray(dense, dtype=np.complex128).reshape(-1)
        if array.size > 0:
            norm = np.linalg.norm(array)
            return array if norm <= 0 else array / norm
    raise RuntimeError("quimb PEPS object does not expose a dense-state conversion method.")


def compute_peps_entropy_profile(
    peps_state: Any,
    *,
    n_sites: int,
    local_dim: int,
    orders: Tuple[int, ...] = (1, 2, 3, 4),
    max_dense_dimension: int = 262144,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Best-effort finite-PEPS entropy profile via dense contraction for small clusters."""
    hilbert_dim = int(local_dim) ** int(n_sites)
    if hilbert_dim > int(max_dense_dimension):
        return {
            "status": "skipped",
            "method": "PEPS",
            "reason": (
                f"Dense PEPS entropy contraction would require Hilbert dimension {hilbert_dim}, "
                f"above the configured cap {int(max_dense_dimension)}."
            ),
            "context": {"n_sites": int(n_sites), "local_dim": int(local_dim)},
        }
    from analysis import compute_ed_entropy_profile_from_state

    profile = compute_ed_entropy_profile_from_state(
        state=_peps_dense_state_vector(peps_state),
        n_sites=int(n_sites),
        local_dim=int(local_dim),
        orders=orders,
        show_progress=show_progress,
    )
    profile["method"] = "PEPS"
    profile.setdefault("context", {})["backend"] = "quimb_peps"
    profile["status"] = "completed"
    return profile


def run_quimb_peps_calculation(
    *,
    geometry: Any,
    model_spec: Any,
    lattice_name: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    coupling_j: float = DEFAULT_COUPLING_J,
    jx: float = DEFAULT_JX,
    jy: float = DEFAULT_JY,
    jz: float = DEFAULT_JZ,
    external_field_terms: Any = None,
    max_sites: int | None = None,
    max_bond_dimension: int = 4,
    max_sweeps: int = 100,
    truncation_cutoff: float = 1.0e-10,
    tau: float | Iterable[float] = DEFAULT_TAU,
    random_seed: int | None = None,
    initial_state_style: str = "random",
    ctm_chi: int | None = None,
    entanglement_max_dense_dim: int = 262144,
    classifier_thresholds: Dict[str, Any] | None = None,
    compute_correlations: bool = True,
    compute_bond_energies: bool = True,
    compute_structure_factors: bool = True,
    compute_uniform_observables: bool = True,
    compute_entanglement: bool = False,
    entropy_orders: Tuple[int, ...] = (1, 2, 3, 4),
    contraction_method: str | None = "ctmrg",
    show_progress: bool = True,
    ground_state_progress: bool | None = None,
    args: Any = None,
    symmetry_reductions: Any = None,
    use_sz_conserved: bool | None = None,
    symmetric: bool = False,
    peps_symmetry_mode: str | None = None,
    peps_strict_symmetry: bool | None = None,
    peps_allow_dense_fallback: bool | None = None,
) -> Dict[str, Any]:
    """Run a finite quimb PEPS solve and DMRG-shaped observable post-processing."""
    lattice_key = _infer_lattice_name(geometry, lattice_name)
    if not quimb_peps_supports_lattice(lattice_key):
        raise ValueError(
            f"quimb_peps currently supports {', '.join(SUPPORTED_PEPS_LATTICES)} lattices; "
            f"received lattice='{lattice_key}'."
        )
    n_sites = int(getattr(geometry, "number_of_sites", 0))
    if max_sites is not None and n_sites > int(max_sites):
        raise ValueError(
            f"finite quimb PEPS safety cap is N <= {int(max_sites)}, but geometry has N={n_sites}."
        )
    D = max(1, int(max_bond_dimension))
    chi = int(ctm_chi if ctm_chi is not None else max(DEFAULT_CTM_CHI, D * D))
    use_sz_conserved_flag = _parse_use_sz_conserved_flag(
        args=args,
        use_sz_conserved=use_sz_conserved,
        symmetry_reductions=symmetry_reductions,
    )
    peps_symmetry_report = resolve_quimb_peps_symmetry_report(
        backend="quimb_peps",
        requested_mode=(
            peps_symmetry_mode
            if peps_symmetry_mode is not None
            else getattr(args, "peps_symmetry_mode", "auto")
        ),
        model_spec=model_spec,
        external_field_terms=external_field_terms,
        symmetry_reductions=symmetry_reductions,
        strict=(
            bool(peps_strict_symmetry)
            if peps_strict_symmetry is not None
            else bool(getattr(args, "peps_strict_symmetry", True))
        ),
        allow_dense_fallback=(
            bool(peps_allow_dense_fallback)
            if peps_allow_dense_fallback is not None
            else bool(getattr(args, "peps_allow_dense_fallback", True))
        ),
        legacy_use_sz_conserved=bool(use_sz_conserved_flag),
    )
    if show_progress:
        for warning in peps_symmetry_report.get("warnings", []):
            print(f"[peps-symmetry] {warning}")
    symmetric_requested = bool(
        symmetric
        or peps_symmetry_report.get("physical_u1_tz_requested", False)
        or peps_symmetry_report.get("physical_s_z2_requested", False)
    )
    symmetric = bool(
        peps_symmetry_report.get("u1_tz_reduces_tensor_cost", False)
        or peps_symmetry_report.get("use_s_z2", False)
    )
    hamiltonian = build_2d_local_hamiltonian(
        model_spec,
        geometry,
        lattice_name=lattice_key,
        alpha=float(alpha),
        beta=float(beta),
        coupling_j=float(coupling_j),
        jx=float(jx),
        jy=float(jy),
        jz=float(jz),
        external_field_terms=external_field_terms,
        ctm_chi=chi,
    )
    if show_progress:
        metadata = getattr(hamiltonian, "yl_metadata", {})
        hamiltonian_terms = _hamiltonian_term_mapping(hamiltonian)
        state_label_sample = [
            (i, j)
            for i in range(int(metadata.get("Lx", 0)))
            for j in range(int(metadata.get("Ly", 0)))
        ][:8]
        print(f"[quimb] state.site_labels sample: {state_label_sample}")
        print(f"[quimb] physical site labels sample: {list(metadata.get('physical_site_labels', []))[:8]}")
        print(f"[quimb] ham_terms key sample: {list(hamiltonian_terms.keys())[:8]}")
        if peps_symmetry_report.get("physical_u1_tz_requested", False):
            charge_map = metadata.get("u1_tz_charge_map", {}).get("physical_index_charge_map", {})
            print(f"[quimb] U1_Tz physical index charges q=2*Tz: {charge_map}")
    peps_symmetry_report["hamiltonian_tz_neutrality"] = validate_quimb_peps_tz_neutrality(hamiltonian)
    peps_symmetry_report["u1_tz_sector_preservation"] = _u1_tz_sector_preservation_report(
        peps_symmetry_report,
        peps_symmetry_report["hamiltonian_tz_neutrality"],
    )
    if (
        bool(peps_symmetry_report.get("physical_u1_tz_requested", False))
        and not bool(peps_symmetry_report["hamiltonian_tz_neutrality"].get("all_terms_tz_neutral", False))
    ):
        raise ValueError(
            "PEPS U(1)_Tz was requested, but at least one Hamiltonian term is not Tz-neutral: "
            f"{peps_symmetry_report['hamiltonian_tz_neutrality'].get('violating_terms', [])[:3]}"
        )
    state = optimize_ipeps_simple_update(
        hamiltonian,
        D=D,
        tau=tau,
        steps=int(max_sweeps),
        seed=random_seed,
        chi=chi,
        cutoff=float(truncation_cutoff),
        progbar=show_progress if ground_state_progress is None else bool(ground_state_progress),
        symmetry_report=peps_symmetry_report,
    )
    measured = evaluate_ipeps_observables(state, hamiltonian, ctm_chi=chi, contraction_method=contraction_method)
    measured["backend"] = "quimb_peps"
    energy_per_site = measured.get("ground_state_energy_per_site", measured.get("energy_per_site"))
    n_sites = int(getattr(geometry, "number_of_sites", 1))
    energy_total = measured.get("energy_total")
    if energy_total is None and energy_per_site is not None:
        energy_total = float(energy_per_site) * float(max(1, n_sites))

    correlations: Dict[str, np.ndarray] = {}
    scalar_correlations: Dict[str, np.ndarray] = {}
    bond_rows: List[Dict[str, Any]] = []
    resolved_bond_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    uniform_observables: Dict[str, Any] = {}
    local_observables: Dict[str, Any] = {}
    entropy_profile: Dict[str, Any] | None = None
    if compute_bond_energies:
        try:
            resolved_bond_rows = collect_resolved_bond_observables_from_peps(
                state,
                hamiltonian,
                model_spec=model_spec,
                ctm_chi=chi,
                show_progress=show_progress,
            )
        except Exception as exc:
            resolved_bond_rows = [{"status": "failed", "warning": str(exc)}]
    if compute_correlations:
        correlations = collect_correlation_matrices_from_peps(
            state,
            hamiltonian,
            model_spec=model_spec,
            ctm_chi=chi,
            show_progress=show_progress,
        )
        scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
        if compute_bond_energies:
            bond_rows = all_bond_energies(
                geometry,
                correlations,
                model_spec,
                float(alpha),
                float(beta),
                float(coupling_j),
                jx=float(jx),
                jy=float(jy),
                jz=float(jz),
                show_progress=show_progress,
                progress_desc="PEPS bond energies",
            )
            if not bond_rows:
                bond_rows = [row for row in resolved_bond_rows if "warning" not in row]
        if compute_structure_factors:
            structure_rows = all_high_symmetry_structure_factors(
                scalar_correlations,
                geometry,
                lattice=lattice_key,
                show_progress=show_progress,
                progress_desc="PEPS structure factors",
            )
    elif compute_bond_energies:
        bond_rows = [row for row in resolved_bond_rows if "warning" not in row]
    if compute_uniform_observables:
        try:
            local_observables = collect_local_observables_from_peps(
                state,
                hamiltonian,
                model_spec=model_spec,
                ctm_chi=chi,
                show_progress=show_progress,
            )
            uniform_observables = {
                "spin_vector_per_site": local_observables.get("uniform_spin_vector"),
                "orbital_vector_per_site": local_observables.get("uniform_orbital_vector"),
                "spin_z_per_site": local_observables.get("spin_z_per_site"),
                "orbital_z_per_site": local_observables.get("orbital_z_per_site"),
            }
        except Exception as exc:
            uniform_observables = {"warning": f"Failed to compute PEPS uniform z observables: {exc}"}
            local_observables = {"status": "failed", "warning": str(exc)}
    if compute_entanglement:
        try:
            entropy_profile = compute_peps_entropy_profile(
                state,
                n_sites=n_sites,
                local_dim=int(getattr(model_spec, "physical_dim", 0)),
                orders=entropy_orders,
                max_dense_dimension=int(entanglement_max_dense_dim),
                show_progress=show_progress,
            )
        except Exception as exc:
            entropy_profile = {"status": "failed", "method": "PEPS", "warning": str(exc)}

    diagnostics = _phase_observable_diagnostics(
        structure_rows,
        bond_rows,
        n_sites,
        plaquette_flux=measured.get("plaquette_flux") if isinstance(measured.get("plaquette_flux"), dict) else None,
    )
    thresholds = classifier_thresholds or DEFAULT_PHASE_CLASSIFIER_THRESHOLDS
    phase_label = _classify_phase_from_diagnostics(
        diagnostics,
        float(alpha),
        float(beta),
        "quimb_peps",
        thresholds,
    )
    info: Dict[str, Any] = {
        "status": "completed",
        "backend": "quimb_peps",
        "method": "finite_peps_simple_update",
        "library": "quimb.tensor",
        "E": energy_total,
        "ground_state_energy": energy_total,
        "energy_per_site": energy_per_site,
        "ground_state_energy_per_site": energy_per_site,
        "phase_label": phase_label,
        "phase_observables": {
            "plaquette_flux": measured.get("plaquette_flux"),
            "diagnostics": diagnostics,
        },
        "plaquette_flux": measured.get("plaquette_flux"),
        "all_plaquette_fluxes": measured.get("all_plaquette_fluxes", {}),
        "structure_factors": structure_rows,
        "bond_energies": bond_rows,
        "resolved_bond_observables": resolved_bond_rows,
        "uniform_observables": uniform_observables,
        "local_observables": local_observables,
        "simple_update": measured.get("simple_update", {}),
        "gate_diagnostics": measured.get("gate_diagnostics", {}),
        "contraction": measured.get("contraction", {}),
        "peps_symmetry_report": peps_symmetry_report,
        "symmetry": peps_symmetry_report,
        "peps_options": {
            "max_bond_dimension": int(D),
            "max_sites": None if max_sites is None else int(max_sites),
            "max_sweeps": int(max_sweeps),
            "truncation_cutoff": float(truncation_cutoff),
            "ctm_chi": int(chi),
            "contraction_method": _normalise_ipeps_contraction_method(contraction_method),
            "tau": [float(value) for value in (tau if isinstance(tau, (list, tuple)) else [tau])],
            "entanglement_max_dense_dim": int(entanglement_max_dense_dim),
            "initial_state_style": str(initial_state_style),
            "random_seed": None if random_seed is None else int(random_seed),
            "ground_state_progress": (
                bool(show_progress) if ground_state_progress is None else bool(ground_state_progress)
            ),
            "symmetric": bool(symmetric),
            "symmetric_requested": bool(symmetric_requested),
            "use_u1_tz": bool(peps_symmetry_report.get("use_u1_tz", False)),
            "u1_tz_charge_map_active": bool(peps_symmetry_report.get("u1_tz_charge_map_active", False)),
            "u1_tz_reduces_tensor_cost": bool(peps_symmetry_report.get("u1_tz_reduces_tensor_cost", False)),
            "use_s_z2": bool(peps_symmetry_report.get("use_s_z2", False)),
            "z2_generator": peps_symmetry_report.get("z2_generator"),
            "use_sz_conserved_requested": False,
            "legacy_use_sz_conserved_dropped": bool(use_sz_conserved_flag),
            "dense_reason": (
                peps_symmetry_report.get("quimb_capabilities", {}).get(
                    "u1_tz_support_reason",
                    "quimb SimpleUpdate U(1)_Tz tensor-block support was not detected.",
                )
                if peps_symmetry_report.get("dense_fallback_used", False)
                else None
            ),
        },
    }
    if entropy_profile is not None:
        info["entanglement"] = entropy_profile
    return {
        "state": state,
        "hamiltonian": hamiltonian,
        "info": info,
        "correlations": correlations,
        "scalar_correlations": scalar_correlations,
        "bond_rows": bond_rows,
        "resolved_bond_rows": resolved_bond_rows,
        "structure_factor_rows": structure_rows,
        "uniform_observables": uniform_observables,
        "local_observables": local_observables,
        "entanglement": entropy_profile,
    }


def _external_field_terms_payload(external_field_terms: Any) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in list(external_field_terms or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        coefficient, operator_name = item[0], item[1]
        payload.append({"coefficient": float(coefficient), "operator": str(operator_name)})
    return payload


def _row_from_result(
    result: Dict[str, Any],
    *,
    alpha: float,
    beta: float,
    alpha_index: int,
    beta_index: int,
) -> Dict[str, Any]:
    row = dict(result)
    row.update(
        {
            "alpha": float(alpha),
            "beta": float(beta),
            "alpha_index": int(alpha_index),
            "beta_index": int(beta_index),
        }
    )
    return row


def run_quimb_peps_scan(
    geometry: Any,
    alpha_values: Iterable[float] | None = None,
    beta_values: Iterable[float] | None = None,
    coupling_j: float = DEFAULT_COUPLING_J,
    max_sites: int | None = None,
    max_bond_dimension: int = 4,
    max_sweeps: int = 100,
    truncation_cutoff: float = 1.0e-10,
    tau: float | Iterable[float] = DEFAULT_TAU,
    carry_state_between_betas: bool = False,
    classifier_thresholds: Dict[str, Any] | None = None,
    external_field_terms: Any = None,
    show_progress: bool = True,
    *,
    model_spec: Any | None = None,
    lattice_name: str | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    jx: float = DEFAULT_JX,
    jy: float = DEFAULT_JY,
    jz: float = DEFAULT_JZ,
    random_seed: int | None = None,
    initial_state_style: str = "random",
    symmetry_reductions: Any = None,
    args: Any = None,
    use_sz_conserved: bool | None = None,
    symmetric: bool = False,
    peps_symmetry_mode: str | None = None,
    peps_strict_symmetry: bool | None = None,
    peps_allow_dense_fallback: bool | None = None,
    ctm_chi: int | None = None,
    entanglement_max_dense_dim: int = 262144,
) -> Dict[str, Any]:
    """Run a finite quimb PEPS alpha-beta scan with finite-DMRG row fields."""
    if model_spec is None:
        model_spec = build_model_spec("1/2", "1/2", "yao_lee", "z")
    lattice_key = _infer_lattice_name(geometry, lattice_name)
    alpha_grid = _as_float_list(alpha_values, DEFAULT_ALPHA if alpha is None else float(alpha))
    beta_grid = _as_float_list(beta_values, DEFAULT_BETA if beta is None else float(beta))
    n_sites = int(getattr(geometry, "number_of_sites", 0))
    D = max(1, int(max_bond_dimension))
    chi = int(ctm_chi if ctm_chi is not None else max(DEFAULT_CTM_CHI, D * D))
    use_sz_conserved_flag = _parse_use_sz_conserved_flag(
        args=args,
        use_sz_conserved=use_sz_conserved,
        symmetry_reductions=symmetry_reductions,
    )
    peps_symmetry_report = resolve_quimb_peps_symmetry_report(
        backend="quimb_peps",
        requested_mode=(
            peps_symmetry_mode
            if peps_symmetry_mode is not None
            else getattr(args, "peps_symmetry_mode", "auto")
        ),
        model_spec=model_spec,
        external_field_terms=external_field_terms,
        symmetry_reductions=symmetry_reductions,
        strict=(
            bool(peps_strict_symmetry)
            if peps_strict_symmetry is not None
            else bool(getattr(args, "peps_strict_symmetry", True))
        ),
        allow_dense_fallback=(
            bool(peps_allow_dense_fallback)
            if peps_allow_dense_fallback is not None
            else bool(getattr(args, "peps_allow_dense_fallback", True))
        ),
        legacy_use_sz_conserved=bool(use_sz_conserved_flag),
    )
    if show_progress:
        for warning in peps_symmetry_report.get("warnings", []):
            print(f"[peps-symmetry] {warning}")
    rows: List[Dict[str, Any]] = []
    scan_stage = _start_stage("quimb finite-PEPS phase scan", show_progress)
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(alpha_grid) * len(beta_grid),
        desc="quimb peps scan",
        unit="point",
        leave=False,
    )
    base_payload: Dict[str, Any] = {
        "status": "running",
        "backend": "quimb_peps",
        "scan_type": "finite_peps_observable_scan",
        "adiabatic_state_passing": {
            "direction": "alpha",
            "carry_state_between_betas": bool(carry_state_between_betas),
            "supported": False,
            "note": "Simple Update state passing is not enabled for the first finite quimb PEPS backend version.",
        },
        "alpha_values": [float(value) for value in alpha_grid],
        "beta_values": [float(value) for value in beta_grid],
        "library": "quimb.tensor",
        "lattice": lattice_key,
        "supported_lattices": list(SUPPORTED_PEPS_LATTICES),
        "number_of_sites": int(getattr(geometry, "number_of_sites", 0)),
        "number_of_bonds": int(len(getattr(geometry, "bond_list", []))),
        "local_physical_dim": int(getattr(model_spec, "physical_dim", 0)),
        "symmetry_reductions": symmetry_reductions,
        "peps_symmetry_report": peps_symmetry_report,
        "symmetry": peps_symmetry_report,
        "controls": {
            "max_bond_dimension": int(D),
            "max_sites": None if max_sites is None else int(max_sites),
            "max_sweeps": int(max_sweeps),
            "truncation_cutoff": float(truncation_cutoff),
            "tau": [float(value) for value in (tau if isinstance(tau, (list, tuple)) else [tau])],
            "random_seed": None if random_seed is None else int(random_seed),
            "initial_state_style": str(initial_state_style),
            "ctm_chi": int(chi),
            "entanglement_max_dense_dim": int(entanglement_max_dense_dim),
            "show_progress": bool(show_progress),
            "ground_state_progress": bool(show_progress),
        },
        "external_field_terms": _external_field_terms_payload(external_field_terms),
    }
    if not quimb_peps_supports_lattice(lattice_key):
        reason = (
            f"quimb_peps currently supports {', '.join(SUPPORTED_PEPS_LATTICES)} lattices; "
            f"received lattice='{lattice_key}'."
        )
        if progress_bar is not None:
            progress_bar.close()
        _end_stage("quimb finite-PEPS phase scan", scan_stage, show_progress)
        base_payload.update(
            {
                "status": "skipped",
                "reason": reason,
                "rows": [],
                "completed_points": 0,
                "failed_points": 0,
                "skipped_points": int(len(alpha_grid) * len(beta_grid)),
                "energy_per_site": None,
                "ground_state_energy_per_site": None,
                "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": reason},
                "phase_label": "Weak/undetermined",
                "all_plaquette_fluxes": {},
            }
        )
        return base_payload
    if max_sites is not None and n_sites > int(max_sites):
        reason = f"finite quimb PEPS safety cap is N <= {int(max_sites)}, but geometry has N={n_sites}."
        if progress_bar is not None:
            progress_bar.close()
        _end_stage("quimb finite-PEPS phase scan", scan_stage, show_progress)
        base_payload.update(
            {
                "status": "skipped",
                "reason": reason,
                "rows": [],
                "completed_points": 0,
                "failed_points": 0,
                "skipped_points": int(len(alpha_grid) * len(beta_grid)),
                "energy_per_site": None,
                "ground_state_energy_per_site": None,
                "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": reason},
                "phase_label": "Weak/undetermined",
                "all_plaquette_fluxes": {},
            }
        )
        return base_payload

    point_index = 0
    for beta_index, beta_value in enumerate(beta_grid):
        for alpha_index, alpha_value in enumerate(alpha_grid):
            point_seed = None if random_seed is None else int(random_seed) + int(point_index)
            try:
                result = run_quimb_peps_calculation(
                    geometry=geometry,
                    model_spec=model_spec,
                    lattice_name=lattice_key,
                    alpha=float(alpha_value),
                    beta=float(beta_value),
                    coupling_j=float(coupling_j),
                    jx=float(jx),
                    jy=float(jy),
                    jz=float(jz),
                    external_field_terms=external_field_terms,
                    max_sites=max_sites,
                    max_bond_dimension=int(D),
                    max_sweeps=int(max_sweeps),
                    truncation_cutoff=float(truncation_cutoff),
                    tau=tau,
                    random_seed=point_seed,
                    initial_state_style=initial_state_style,
                    ctm_chi=chi,
                    entanglement_max_dense_dim=int(entanglement_max_dense_dim),
                    classifier_thresholds=classifier_thresholds,
                    compute_correlations=True,
                    compute_bond_energies=True,
                    compute_structure_factors=True,
                    compute_uniform_observables=False,
                    compute_entanglement=False,
                    show_progress=False,
                    ground_state_progress=show_progress,
                    args=args,
                    symmetry_reductions=symmetry_reductions,
                    use_sz_conserved=use_sz_conserved,
                    symmetric=symmetric,
                    peps_symmetry_mode=peps_symmetry_report.get("requested_mode", peps_symmetry_mode),
                    peps_strict_symmetry=(
                        bool(peps_strict_symmetry)
                        if peps_strict_symmetry is not None
                        else bool(getattr(args, "peps_strict_symmetry", True))
                    ),
                    peps_allow_dense_fallback=(
                        bool(peps_allow_dense_fallback)
                        if peps_allow_dense_fallback is not None
                        else bool(getattr(args, "peps_allow_dense_fallback", True))
                    ),
                )
                info = dict(result["info"])
                energy_total = info.get("ground_state_energy", info.get("E"))
                energy_per_site = info.get("ground_state_energy_per_site", info.get("energy_per_site"))
                row = {
                    "status": "completed",
                    "backend": "quimb_peps",
                    "alpha_index": int(alpha_index),
                    "beta_index": int(beta_index),
                    "alpha": float(alpha_value),
                    "beta": float(beta_value),
                    "energy": energy_total,
                    "energy_per_site": energy_per_site,
                    "ground_state_energy_per_site": energy_per_site,
                    "phase_label": info.get("phase_label"),
                    "diagnostics": (info.get("phase_observables") or {}).get("diagnostics", {}),
                    "structure_factors": info.get("structure_factors", []),
                    "bond_energies": info.get("bond_energies", []),
                    "plaquette_flux": info.get("plaquette_flux"),
                    "all_plaquette_fluxes": info.get("all_plaquette_fluxes", {}),
                    "gate_diagnostics": info.get("gate_diagnostics", {}),
                    "peps_symmetry_report": info.get("peps_symmetry_report", peps_symmetry_report),
                    "peps_info": info,
                    "random_seed": None if point_seed is None else int(point_seed),
                }
            except Exception as exc:
                error_text = str(exc) or exc.__class__.__name__
                row = {
                    "status": "failed",
                    "backend": "quimb_peps",
                    "alpha_index": int(alpha_index),
                    "beta_index": int(beta_index),
                    "alpha": float(alpha_value),
                    "beta": float(beta_value),
                    "energy": None,
                    "energy_per_site": None,
                    "ground_state_energy_per_site": None,
                    "phase_label": "Weak/undetermined",
                    "diagnostics": {"warning": error_text},
                    "structure_factors": [],
                    "bond_energies": [],
                    "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": error_text},
                    "all_plaquette_fluxes": {},
                    "error": error_text,
                    "peps_symmetry_report": peps_symmetry_report,
                    "random_seed": None if point_seed is None else int(point_seed),
                }
            rows.append(row)
            point_index += 1
            if progress_bar is not None:
                progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    _end_stage("quimb finite-PEPS phase scan", scan_stage, show_progress)
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    representative = completed_rows[0] if completed_rows else (rows[0] if rows else {})
    status = "completed" if completed_rows and not failed_rows else "completed_with_warnings"
    if not completed_rows and failed_rows:
        status = "failed"
    base_payload.update(
        {
            "status": status,
            "rows": rows,
            "completed_points": int(len(completed_rows)),
            "failed_points": int(len(failed_rows)),
            "skipped_points": 0,
            "energy_per_site": representative.get("energy_per_site"),
            "ground_state_energy_per_site": representative.get("ground_state_energy_per_site"),
            "plaquette_flux": representative.get("plaquette_flux"),
            "phase_label": representative.get("phase_label"),
            "diagnostics": representative.get("diagnostics", {}),
            "all_plaquette_fluxes": representative.get("all_plaquette_fluxes", {}),
            "gate_diagnostics": representative.get("gate_diagnostics", {}),
            "peps_symmetry_report": representative.get("peps_symmetry_report", peps_symmetry_report),
        }
    )
    if representative.get("error") is not None:
        base_payload["error"] = representative["error"]
    return base_payload


def run_quimb_ipeps_scan(
    geometry: Any,
    alpha_values: Iterable[float] | None = None,
    beta_values: Iterable[float] | None = None,
    coupling_j: float = DEFAULT_COUPLING_J,
    max_unit_cell_sites: int | None = None,
    max_bond_dimension: int = 4,
    max_iterations: int = 100,
    truncation_cutoff: float = 1.0e-10,
    carry_state_between_betas: bool = False,
    classifier_thresholds: Dict[str, Any] | None = None,
    external_field_terms: Any = None,
    show_progress: bool = True,
    *,
    model_spec: Any | None = None,
    lattice_name: str | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    jx: float = DEFAULT_JX,
    jy: float = DEFAULT_JY,
    jz: float = DEFAULT_JZ,
    random_seed: int | None = None,
    initial_state_style: str = "random",
    symmetry_reductions: Any = None,
    args: Any = None,
    use_sz_conserved: bool | None = None,
    symmetric: bool = False,
    ipeps_symmetry_mode: str | None = None,
    ipeps_strict_symmetry: bool | None = None,
    ipeps_allow_dense_fallback: bool | None = None,
    unit_cell_kind: str | None = None,
    use_translation_symmetry: bool = True,
    tau: float | Iterable[float] = DEFAULT_TAU,
    ctm_chi: int | None = None,
    contraction_method: str | None = "ctmrg",
) -> Dict[str, Any]:
    """Run a quimb iPEPS observable scan with the TeNPy iDMRG scan shape.

    The positional arguments intentionally match
    ``tenpy_backend.run_alpha_beta_idmrg_observable_scan``. Extra keyword-only
    arguments carry the general model construction data needed by quimb.
    """
    if model_spec is None:
        model_spec = build_model_spec("1/2", "1/2", "yao_lee", "z")
    lattice_key = _infer_lattice_name(geometry, lattice_name)
    selected_unit_cell_kind = str(
        unit_cell_kind
        if unit_cell_kind is not None
        else getattr(args, "ipeps_unit_cell_kind", "auto")
    )
    source_geometry = geometry
    contraction_key = _normalise_ipeps_contraction_method(contraction_method)
    if bool(use_translation_symmetry):
        geometry = _ipeps_unit_cell_geometry(geometry, lattice_key, selected_unit_cell_kind)
    else:
        selected_unit_cell_kind = "translation_disabled"
    alpha_grid = _as_float_list(alpha_values, DEFAULT_ALPHA if alpha is None else float(alpha))
    beta_grid = _as_float_list(beta_values, DEFAULT_BETA if beta is None else float(beta))
    n_sites = int(getattr(geometry, "number_of_sites", 0))
    D = max(1, int(max_bond_dimension))
    chi = int(ctm_chi if ctm_chi is not None else max(DEFAULT_CTM_CHI, D * D))
    thresholds = classifier_thresholds or DEFAULT_PHASE_CLASSIFIER_THRESHOLDS
    use_sz_conserved_flag = _parse_use_sz_conserved_flag(
        args=args,
        use_sz_conserved=use_sz_conserved,
        symmetry_reductions=symmetry_reductions,
    )
    ipeps_symmetry_report = resolve_quimb_peps_symmetry_report(
        backend="quimb_ipeps",
        requested_mode=(
            ipeps_symmetry_mode
            if ipeps_symmetry_mode is not None
            else getattr(args, "ipeps_symmetry_mode", "auto")
        ),
        model_spec=model_spec,
        external_field_terms=external_field_terms,
        symmetry_reductions=symmetry_reductions,
        strict=(
            bool(ipeps_strict_symmetry)
            if ipeps_strict_symmetry is not None
            else bool(getattr(args, "ipeps_strict_symmetry", True))
        ),
        allow_dense_fallback=(
            bool(ipeps_allow_dense_fallback)
            if ipeps_allow_dense_fallback is not None
            else bool(getattr(args, "ipeps_allow_dense_fallback", True))
        ),
        unit_cell_kind=selected_unit_cell_kind,
        legacy_use_sz_conserved=bool(use_sz_conserved_flag),
    )
    if show_progress:
        for warning in ipeps_symmetry_report.get("warnings", []):
            print(f"[ipeps-symmetry] {warning}")
    symmetric_requested = bool(
        symmetric
        or ipeps_symmetry_report.get("physical_u1_tz_requested", False)
        or ipeps_symmetry_report.get("physical_s_z2_requested", False)
    )
    symmetric = bool(
        ipeps_symmetry_report.get("u1_tz_reduces_tensor_cost", False)
        or ipeps_symmetry_report.get("use_s_z2", False)
    )
    rows: List[Dict[str, Any]] = []
    scan_stage = _start_stage("quimb iPEPS phase scan", show_progress)
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(alpha_grid) * len(beta_grid),
        desc="quimb ipeps scan",
        unit="point",
        leave=False,
    )
    base_payload: Dict[str, Any] = {
        "status": "running",
        "backend": "quimb_ipeps",
        "scan_type": "ipeps_observable_scan",
        "adiabatic_state_passing": {
            "direction": "alpha",
            "carry_state_between_betas": bool(carry_state_between_betas),
            "supported": False,
            "note": "Simple Update state passing is not enabled for the first quimb iPEPS backend version.",
        },
        "alpha_values": [float(value) for value in alpha_grid],
        "beta_values": [float(value) for value in beta_grid],
        "library": "quimb.tensor",
        "lattice": lattice_key,
        "supported_lattices": list(SUPPORTED_IPEPS_LATTICES),
        "number_of_sites": int(getattr(geometry, "number_of_sites", 0)),
        "number_of_bonds": int(len(getattr(geometry, "bond_list", []))),
        "source_finite_geometry": {
            "number_of_sites": int(getattr(source_geometry, "number_of_sites", 0)),
            "number_of_bonds": int(len(getattr(source_geometry, "bond_list", []))),
            "length_x": int(getattr(source_geometry, "length_x", 0) or 0),
            "length_y": int(getattr(source_geometry, "length_y", 0) or 0),
        },
        "unit_cell_geometry": {
            "number_of_sites": int(getattr(geometry, "number_of_sites", 0)),
            "number_of_bonds": int(len(getattr(geometry, "bond_list", []))),
            "length_x": int(getattr(geometry, "length_x", 0) or 0),
            "length_y": int(getattr(geometry, "length_y", 0) or 0),
            "kind": selected_unit_cell_kind,
            "translation_symmetry_enabled": bool(use_translation_symmetry),
            "primitive_honeycomb_sites": 2 if lattice_key == "honeycomb" else None,
            "note": (
                "Internal iPEPS unit cell is separate from the finite ED reference cluster. "
                "For honeycomb, the physical primitive cell has two sublattice sites; "
                "quimb's finite PEPS SimpleUpdate/CTMRG path uses a compatible periodic 2x2 "
                "honeycomb supercell so x/y/z bonds do not collapse into hyperedges."
            ),
        },
        "local_physical_dim": int(getattr(model_spec, "physical_dim", 0)),
        "model_family": str(getattr(model_spec, "model_family", "")),
        "spin_rep": str(getattr(model_spec, "spin_rep", "")),
        "orbital_rep": str(getattr(model_spec, "orbital_rep", "")),
        "ising_axis": str(getattr(model_spec, "ising_axis", "")),
        "quimb_tensor_module": str(getattr(qtn, "__name__", "quimb.tensor")),
        "symmetry": {
            "use_sz_conserved_requested": False,
            "legacy_use_sz_conserved_dropped": bool(use_sz_conserved_flag),
            "symmetric_requested": bool(symmetric_requested),
            "symmetric": bool(symmetric),
            "use_u1_tz": bool(ipeps_symmetry_report.get("use_u1_tz", False)),
            "u1_tz_charge_map_active": bool(ipeps_symmetry_report.get("u1_tz_charge_map_active", False)),
            "u1_tz_reduces_tensor_cost": bool(ipeps_symmetry_report.get("u1_tz_reduces_tensor_cost", False)),
            "use_s_z2": bool(ipeps_symmetry_report.get("use_s_z2", False)),
            "z2_generator": ipeps_symmetry_report.get("z2_generator"),
            "dense_reason": (
                ipeps_symmetry_report.get("quimb_capabilities", {}).get(
                    "u1_tz_support_reason",
                    "quimb SimpleUpdate U(1)_Tz tensor-block support was not detected.",
                )
                if ipeps_symmetry_report.get("dense_fallback_used", False)
                else None
            ),
            "symmetry_reductions": symmetry_reductions,
            "translation_symmetry": {
                "enabled": bool(use_translation_symmetry),
                "implemented_as": "periodic repeated iPEPS unit cell",
            },
        },
        "ipeps_symmetry_report": ipeps_symmetry_report,
        "controls": {
            "max_bond_dimension": int(D),
            "max_unit_cell_sites": None if max_unit_cell_sites is None else int(max_unit_cell_sites),
            "max_iterations": int(max_iterations),
            "truncation_cutoff": float(truncation_cutoff),
            "random_seed": None if random_seed is None else int(random_seed),
            "initial_state_style": str(initial_state_style),
            "tau": [float(value) for value in (tau if isinstance(tau, (list, tuple)) else [tau])],
            "ctm_chi": int(chi),
            "show_progress": bool(show_progress),
            "classifier_thresholds": thresholds,
            "unit_cell_kind": selected_unit_cell_kind,
            "use_translation_symmetry": bool(use_translation_symmetry),
            "translation_symmetry": {
                "enabled": bool(use_translation_symmetry),
                "implemented_as": "periodic repeated iPEPS unit cell",
            },
            "contraction_method": contraction_key,
            "ctmrg_enabled": contraction_key == "ctmrg",
        },
        "external_field_terms": _external_field_terms_payload(external_field_terms),
    }
    if not bool(use_translation_symmetry):
        reason = "quimb iPEPS requires a translated/repeated unit cell; use finite PEPS or enable iPEPS translation symmetry."
        if progress_bar is not None:
            progress_bar.close()
        _end_stage("quimb iPEPS phase scan", scan_stage, show_progress)
        base_payload.update(
            {
                "status": "skipped",
                "reason": reason,
                "rows": [],
                "completed_points": 0,
                "failed_points": 0,
                "skipped_points": int(len(alpha_grid) * len(beta_grid)),
                "energy_per_site": None,
                "ground_state_energy_per_site": None,
                "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": reason},
                "phase_label": "Weak/undetermined",
                "all_plaquette_fluxes": {},
            }
        )
        return base_payload
    if not quimb_ipeps_supports_lattice(lattice_key):
        reason = (
            f"quimb_ipeps currently supports {', '.join(SUPPORTED_IPEPS_LATTICES)} lattices; "
            f"received lattice='{lattice_key}'."
        )
        if progress_bar is not None:
            progress_bar.close()
        _end_stage("quimb iPEPS phase scan", scan_stage, show_progress)
        base_payload.update(
            {
                "status": "skipped",
                "reason": reason,
                "rows": [],
                "completed_points": 0,
                "failed_points": 0,
                "skipped_points": int(len(alpha_grid) * len(beta_grid)),
                "energy_per_site": None,
                "ground_state_energy_per_site": None,
                "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": reason},
                "phase_label": "Weak/undetermined",
                "all_plaquette_fluxes": {},
            }
        )
        return base_payload
    if max_unit_cell_sites is not None and n_sites > int(max_unit_cell_sites):
        reason = f"quimb iPEPS unit-cell safety cap is N <= {int(max_unit_cell_sites)}, but geometry has N={n_sites}."
        if progress_bar is not None:
            progress_bar.close()
        _end_stage("quimb iPEPS phase scan", scan_stage, show_progress)
        base_payload.update(
            {
                "status": "skipped",
                "reason": reason,
                "rows": [],
                "completed_points": 0,
                "failed_points": 0,
                "skipped_points": int(len(alpha_grid) * len(beta_grid)),
                "energy_per_site": None,
                "ground_state_energy_per_site": None,
                "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": reason},
                "phase_label": "Weak/undetermined",
                "all_plaquette_fluxes": {},
            }
        )
        return base_payload

    for beta_index, beta_value in enumerate(beta_grid):
        for alpha_index, alpha_value in enumerate(alpha_grid):
            point_seed = None if random_seed is None else int(random_seed) + len(rows)
            try:
                hamiltonian = build_2d_local_hamiltonian(
                    model_spec,
                    geometry,
                    lattice_name=lattice_key,
                    alpha=float(alpha_value),
                    beta=float(beta_value),
                    coupling_j=float(coupling_j),
                    jx=float(jx),
                    jy=float(jy),
                    jz=float(jz),
                    external_field_terms=external_field_terms,
                    ctm_chi=chi,
                )
                if show_progress and len(rows) == 0:
                    metadata = getattr(hamiltonian, "yl_metadata", {})
                    hamiltonian_terms = _hamiltonian_term_mapping(hamiltonian)
                    state_label_sample = [
                        (i, j)
                        for i in range(int(metadata.get("Lx", 0)))
                        for j in range(int(metadata.get("Ly", 0)))
                    ][:8]
                    print(f"[quimb] iPEPS state.site_labels sample: {state_label_sample}")
                    print(f"[quimb] iPEPS physical site labels sample: {list(metadata.get('physical_site_labels', []))[:8]}")
                    print(f"[quimb] iPEPS ham_terms key sample: {list(hamiltonian_terms.keys())[:8]}")
                    if ipeps_symmetry_report.get("physical_u1_tz_requested", False):
                        charge_map = metadata.get("u1_tz_charge_map", {}).get("physical_index_charge_map", {})
                        print(f"[quimb] iPEPS U1_Tz physical index charges q=2*Tz: {charge_map}")
                point_symmetry_report = dict(ipeps_symmetry_report)
                point_symmetry_report["hamiltonian_tz_neutrality"] = validate_quimb_peps_tz_neutrality(hamiltonian)
                point_symmetry_report["u1_tz_sector_preservation"] = _u1_tz_sector_preservation_report(
                    point_symmetry_report,
                    point_symmetry_report["hamiltonian_tz_neutrality"],
                )
                if (
                    bool(point_symmetry_report.get("physical_u1_tz_requested", False))
                    and not bool(
                        point_symmetry_report["hamiltonian_tz_neutrality"].get("all_terms_tz_neutral", False)
                    )
                ):
                    raise ValueError(
                        "iPEPS U(1)_Tz was requested, but at least one Hamiltonian term is not Tz-neutral: "
                        f"{point_symmetry_report['hamiltonian_tz_neutrality'].get('violating_terms', [])[:3]}"
                    )
                state = optimize_ipeps_simple_update(
                    hamiltonian,
                    D=D,
                    tau=tau,
                    steps=int(max_iterations),
                    seed=point_seed,
                    chi=chi,
                    cutoff=float(truncation_cutoff),
                    progbar=show_progress,
                    symmetry_report=point_symmetry_report,
                )
                measured = evaluate_ipeps_observables(
                    state,
                    hamiltonian,
                    ctm_chi=chi,
                    contraction_method=contraction_key,
                )
                plaquette_flux = measured.get("plaquette_flux")
                try:
                    local_observables = collect_local_observables_from_peps(
                        state,
                        hamiltonian,
                        model_spec=model_spec,
                        ctm_chi=chi,
                        show_progress=False,
                    )
                except Exception as exc:
                    local_observables = {"status": "failed", "warning": str(exc)}
                try:
                    bond_rows = collect_resolved_bond_observables_from_peps(
                        state,
                        hamiltonian,
                        model_spec=model_spec,
                        ctm_chi=chi,
                        show_progress=False,
                    )
                except Exception as exc:
                    bond_rows = [{"status": "failed", "warning": str(exc)}]
                diagnostics = _phase_observable_diagnostics(
                    [],
                    [row for row in bond_rows if isinstance(row, dict) and "warning" not in row],
                    int(getattr(geometry, "number_of_sites", 1)),
                    plaquette_flux=plaquette_flux if isinstance(plaquette_flux, dict) else None,
                )
                phase_label = _classify_phase_from_diagnostics(
                    diagnostics,
                    float(alpha_value),
                    float(beta_value),
                    "quimb_ipeps",
                    thresholds,
                )
                energy_density = measured.get("ground_state_energy_per_site", measured.get("energy_per_site"))
                observables = {
                    "plaquette_flux": plaquette_flux,
                    "all_plaquette_fluxes": measured.get("all_plaquette_fluxes", {}),
                    "W_p": plaquette_flux.get("W_p", plaquette_flux.get("value")) if isinstance(plaquette_flux, dict) else None,
                    "local_order_parameters": local_observables,
                    "local_observables": local_observables,
                    "bond_observables": bond_rows,
                }
                ipeps_options = {
                    "max_bond_dimension": int(D),
                    "ctm_chi": int(chi),
                    "truncation_cutoff": float(truncation_cutoff),
                    "max_iterations": int(max_iterations),
                    "tau": [float(value) for value in (tau if isinstance(tau, (list, tuple)) else [tau])],
                    "symmetric": bool(symmetric),
                    "use_u1_tz": bool(point_symmetry_report.get("use_u1_tz", False)),
                    "u1_tz_charge_map_active": bool(point_symmetry_report.get("u1_tz_charge_map_active", False)),
                    "u1_tz_reduces_tensor_cost": bool(point_symmetry_report.get("u1_tz_reduces_tensor_cost", False)),
                    "use_s_z2": bool(point_symmetry_report.get("use_s_z2", False)),
                    "z2_generator": point_symmetry_report.get("z2_generator"),
                    "use_sz_conserved_requested": False,
                    "legacy_use_sz_conserved_dropped": bool(use_sz_conserved_flag),
                    "unit_cell_kind": selected_unit_cell_kind,
                    "use_translation_symmetry": bool(use_translation_symmetry),
                    "contraction_method": contraction_key,
                    "ctmrg_enabled": contraction_key == "ctmrg",
                }
                result = {
                    "status": "completed",
                    "backend": "quimb_ipeps",
                    "alpha_index": int(alpha_index),
                    "beta_index": int(beta_index),
                    "alpha": float(alpha_value),
                    "beta": float(beta_value),
                    "energy_per_site": energy_density,
                    "ground_state_energy_per_site": energy_density,
                    "energy_per_unit_cell": (
                        None
                        if energy_density is None
                        else float(energy_density) * float(max(1, int(getattr(geometry, "number_of_sites", 1))))
                    ),
                    "used_adiabatic_initial_state": False,
                    "observables": observables,
                    "plaquette_flux": plaquette_flux,
                    "all_plaquette_fluxes": measured.get("all_plaquette_fluxes", {}),
                    "dmrg_options": ipeps_options,
                    "ipeps_options": ipeps_options,
                    "ipeps_symmetry_report": point_symmetry_report,
                    "symmetry": point_symmetry_report,
                    "phase_label": phase_label,
                    "diagnostics": diagnostics,
                    "structure_factors": [],
                    "bond_energies": bond_rows,
                    "local_observables": local_observables,
                    "contraction": measured.get("contraction", {}),
                    "simple_update": measured.get("simple_update", {}),
                    "gate_diagnostics": measured.get("gate_diagnostics", {}),
                }
            except Exception as exc:
                error_text = str(exc) or exc.__class__.__name__
                result = {
                    "status": "failed",
                    "backend": "quimb_ipeps",
                    "alpha_index": int(alpha_index),
                    "beta_index": int(beta_index),
                    "alpha": float(alpha_value),
                    "beta": float(beta_value),
                    "energy_per_site": None,
                    "ground_state_energy_per_site": None,
                    "energy_per_unit_cell": None,
                    "used_adiabatic_initial_state": False,
                    "observables": {
                        "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": error_text},
                        "all_plaquette_fluxes": {},
                        "W_p": None,
                        "local_order_parameters": {},
                    },
                    "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": error_text},
                    "all_plaquette_fluxes": {},
                    "phase_label": "Weak/undetermined",
                    "diagnostics": {"warning": error_text},
                    "structure_factors": [],
                    "bond_energies": [],
                    "error": error_text,
                    "ipeps_symmetry_report": ipeps_symmetry_report,
                }
            rows.append(result)
            if progress_bar is not None:
                progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    _end_stage("quimb iPEPS phase scan", scan_stage, show_progress)

    completed_rows = [row for row in rows if row.get("status") == "completed"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]
    representative = completed_rows[0] if completed_rows else (rows[0] if rows else {})
    status = "completed" if completed_rows and not failed_rows else "completed_with_warnings"
    if not completed_rows and failed_rows:
        status = "failed"
    if not completed_rows and skipped_rows and not failed_rows:
        status = "skipped"
    base_payload.update(
        {
            "status": status,
            "rows": rows,
            "completed_points": int(len(completed_rows)),
            "failed_points": int(len(failed_rows)),
            "skipped_points": int(len(skipped_rows)),
            "energy_per_site": representative.get("energy_per_site"),
            "ground_state_energy_per_site": representative.get("ground_state_energy_per_site"),
            "energy_per_unit_cell": representative.get("energy_per_unit_cell"),
            "observables": representative.get("observables", {}),
            "plaquette_flux": representative.get(
                "plaquette_flux",
                {"available": False, "value": None, "W_p": None, "reason": "No iPEPS rows were produced."},
            ),
            "phase_label": representative.get("phase_label"),
            "diagnostics": representative.get("diagnostics", {}),
            "bond_energies": representative.get("bond_energies", []),
            "local_observables": representative.get("local_observables", {}),
            "all_plaquette_fluxes": representative.get("all_plaquette_fluxes", {}),
            "gate_diagnostics": representative.get("gate_diagnostics", {}),
            "ipeps_symmetry_report": representative.get("ipeps_symmetry_report", ipeps_symmetry_report),
        }
    )
    if representative.get("error") is not None:
        base_payload["error"] = representative["error"]
    return base_payload
