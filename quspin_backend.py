#!/usr/bin/env python3
"""QuSpin exact-diagonalization backend for the spin-orbital Yao-Lee model.

This module mirrors the small-cluster ED entry points in ``ed_backend.py`` for
the spin-1/2, orbital-1/2 Yao-Lee Hilbert space.  The local physical dimension
is d=4, represented as a tensor product of a spin chain and an orbital chain.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sparse

from analysis import profile_stage

_QUSPIN_IMPORT_ERROR: Exception | None = None


def _quspin_basis_api() -> Tuple[Any, Any, Any]:
    """Load QuSpin basis classes only when a QuSpin calculation needs them."""
    global _QUSPIN_IMPORT_ERROR
    try:
        from quspin.basis import spin_basis_1d, spin_basis_general, tensor_basis  # type: ignore
    except Exception as exc:
        _QUSPIN_IMPORT_ERROR = exc
        raise ImportError(
            "The Python package 'quspin' is required only for the QuSpin ED backend. "
            "Install quspin or use --ed-backend standard / --no-run-ed."
        ) from exc
    return spin_basis_1d, spin_basis_general, tensor_basis


def _quspin_hamiltonian_class() -> Any:
    """Load QuSpin's Hamiltonian builder only for actual QuSpin ED work."""
    global _QUSPIN_IMPORT_ERROR
    try:
        from quspin.operators import hamiltonian  # type: ignore
    except Exception as exc:
        _QUSPIN_IMPORT_ERROR = exc
        raise ImportError(
            "The Python package 'quspin' is required only for the QuSpin ED backend. "
            "Install quspin or use --ed-backend standard / --no-run-ed."
        ) from exc
    return hamiltonian


def quspin_package_available() -> Tuple[bool, str | None]:
    """Return whether QuSpin can be imported, without forcing callers to use it."""
    try:
        _quspin_basis_api()
        _quspin_hamiltonian_class()
    except Exception as exc:
        return False, str(exc)
    return True, None


def quspin_combined_c3_api_support_report(
    *,
    model_family: str = "yao_lee",
    phase_scan_requested: bool = False,
) -> Dict[str, Any]:
    """Report whether QuSpin can currently host the physical combined C3.

    The Yao-Lee C3 symmetry is not a pure integer site map.  It is a lattice
    120-degree rotation composed with the local spin rotation
    ``exp[-i(2*pi/3)*(Sx+Sy+Sz)/sqrt(3)]``.  In the working Sz computational
    basis this local spin rotation maps each spin bit to a superposition, so the
    usual QuSpin map interfaces cannot express it as a single integer-state
    representative map.  A future QuSpin implementation would need a carefully
    validated spin-[111] basis encoding or a deeper custom route that can carry
    configuration-dependent complex phases and the transformed Hamiltonian.
    """
    report: Dict[str, Any] = {
        "status": "checked",
        "experimental": True,
        "model_family": str(model_family),
        "physical_combined_c3": (
            "lattice 120-degree rotation times local spin rotation "
            "U_C3=exp[-i(2*pi/3)*(Sx+Sy+Sz)/sqrt(3)]"
        ),
        "pure_site_permutation_c3_supported_by_quspin": False,
        "pure_site_permutation_c3_rejected_for_yao_lee": str(model_family).strip().lower() == "yao_lee",
        "why_not_native_quspin": (
            "spin_basis_general and the current tensor_basis route represent symmetry generators "
            "as basis-state maps. The physical Yao-Lee combined C3 includes a non-diagonal local "
            "spin rotation in the Sz basis, so a pure C3_map would block-diagonalize the wrong "
            "operator."
        ),
        "what_would_be_needed_for_quspin": (
            "Rotate the spin sector to the [111] basis so U_C3 is diagonal, rebuild every Yao-Lee "
            "Hamiltonian term in that basis, encode the resulting configuration-dependent C3 phase "
            "in a packed spin-orbital user_basis, and validate N=8 against the standard projector."
        ),
        "z2_compatibility_note": (
            "The currently used spin_pi_z or spin_flip Z2 label is not treated as an independent "
            "commuting label with true combined C3. C3 cycles spin axes, so the full little group "
            "would need a joint group projector rather than separate QuSpin zblock and C3 labels."
        ),
        "combined_c3_implemented": False,
        "phase_scan_requested": bool(phase_scan_requested),
        "phase_scan_allowed": False,
        "validation_location": "tests/test_projector_ed_symmetry_path.py",
        "requires": [
            "a spin-[111] basis rotation with correctly encoded local C3 phase sectors",
            "a packed spin-orbital user_basis carrying fused-site translation and total Tz filtering",
            "Hamiltonian terms transformed consistently into the spin-[111] basis",
            "N=8 validation against the standard_projector combined-C3 spectrum in the test suite before phase scans",
        ],
    }
    try:
        import quspin.basis as quspin_basis  # type: ignore

        report["quspin_available"] = True
        has_user_basis = bool(hasattr(quspin_basis, "user_basis"))
        report["has_user_basis"] = has_user_basis
        report["has_spin_basis_general"] = bool(hasattr(quspin_basis, "spin_basis_general"))
        report["user_basis_note"] = (
            "user_basis is installed, but no validated packed spin-[111] implementation exists here; "
            "presence of user_basis alone is not enough for non-diagonal combined C3."
            if has_user_basis
            else "user_basis is not installed."
        )
    except Exception as exc:
        report["quspin_available"] = False
        report["api_error"] = str(exc)
        report["has_user_basis"] = False
        report["has_spin_basis_general"] = False
    report["reason"] = (
        "No QuSpin combined-C3 implementation is enabled. A pure C3_map is rejected for Yao-Lee "
        "because it omits the required local spin rotation and would not commute with the "
        "bond-directional Hamiltonian."
    )
    if bool(phase_scan_requested):
        report["phase_scan_rejection_reason"] = (
            "quspin_experimental_c3 cannot be used in phase scans until combined C3 is implemented; "
            "N=8 validation belongs in the test suite, not in runtime options."
        )
    return report


def quspin_fused_translation_api_support_report(
    geometry: Any | None = None,
    *,
    use_tau_z_block: bool = True,
    use_z2_block: bool = False,
    requested: bool = False,
) -> Dict[str, Any]:
    """Report whether QuSpin can host fused spin-orbital translations.

    The physical translation needed for the spin-orbital Yao-Lee model is the
    diagonal operation on the local physical site,
    ``T_fused |s_i,t_i> = |s_{T(i)},t_{T(i)}>``.  QuSpin's current
    ``tensor_basis`` route represents spin and orbital as separate chains, so
    ordinary translation blocks there would be independent factor translations,
    not this fused physical-site translation.  A future QuSpin implementation
    would need a packed ``user_basis`` with local ``sps=4`` states, explicit
    total-Tz filtering, and validated custom maps.
    """
    report: Dict[str, Any] = {
        "status": "checked",
        "experimental": True,
        "requested": bool(requested),
        "physical_translation": "T_fused |s_i,t_i> = |s_{T(i)},t_{T(i)}>",
        "tensor_basis_translation_supported": False,
        "tensor_basis_translation_rejected": True,
        "implemented": False,
        "available": False,
        "use_tau_z_block": bool(use_tau_z_block),
        "use_z2_block": bool(use_z2_block),
        "validation_location": "tests/test_projector_ed_symmetry_path.py",
        "requires": [
            "packed spin-orbital user_basis with sps=4 local states",
            "custom total-Tz state filtering/pcon for the orbital component",
            "custom fused translation maps acting on packed physical sites",
            "optional Z2 maps validated in the same packed basis",
            "N=8 comparison against standard_projector in the test suite before production use",
        ],
    }
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            import quspin.basis as quspin_basis  # type: ignore

        report["quspin_available"] = True
        report["has_user_basis"] = bool(hasattr(quspin_basis, "user_basis"))
        report["has_spin_basis_general"] = bool(hasattr(quspin_basis, "spin_basis_general"))
        user_basis_obj = getattr(quspin_basis, "user_basis", None)
        report["user_basis_object"] = str(user_basis_obj) if user_basis_obj is not None else None
    except Exception as exc:
        report["quspin_available"] = False
        report["api_error"] = str(exc)
        report["has_user_basis"] = False
        report["has_spin_basis_general"] = False

    if geometry is not None:
        try:
            translation_support = quspin_translation_block_support(geometry)
            report["geometry_translation_support"] = translation_support
            report["geometry_supports_requested_maps"] = bool(
                all(
                    axis_report.get("geometry_supported", False)
                    for axis_report in translation_support.values()
                )
            )
        except Exception as exc:
            report["geometry_translation_error"] = str(exc)
            report["geometry_supports_requested_maps"] = False

    report["reason"] = (
        "QuSpin fused-site translation is unavailable in this build. The installed API exposes "
        "user_basis, but this project has not implemented and validated the required packed sps=4 "
        "basis, total-Tz filter, fused translation maps, and optional Z2 maps. "
        "QuSpin tensor_basis translation is not used with Tz because Yao-Lee translation must act "
        "on fused spin-orbital physical sites."
    )
    return report

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


def build_quspin_1d_translation_maps_from_geometry(geometry: Any) -> Dict[str, Dict[str, Any]]:
    """Build raw 1D integer Tx/Ty maps from ``GeometryData`` cell labels.

    These are the maps QuSpin's ``spin_basis_general`` expects for a single
    spin-1/2 chain.  For the spin-orbital Yao-Lee tensor basis they are only a
    validated ingredient: a production translation block must still be the
    diagonal/fused physical-site translation acting on spin and orbital
    together.
    """
    t1, t2 = build_honeycomb_torus_translation_permutations(geometry)
    result: Dict[str, Dict[str, Any]] = {}
    if bool(getattr(geometry, "circumference_x", False)):
        result["x"] = {
            "available": True,
            "map": [int(value) for value in np.asarray(t1, dtype=np.int32).tolist()],
            "quspin_map": [int(value) for value in np.asarray(t1, dtype=np.int32).tolist()],
            "direction": "x",
        }
    else:
        result["x"] = {
            "available": False,
            "direction": "x",
            "reason": "circumference_x is false; no periodic Tx map is available.",
        }
    if bool(getattr(geometry, "circumference_y", False)):
        result["y"] = {
            "available": True,
            "map": [int(value) for value in np.asarray(t2, dtype=np.int32).tolist()],
            "quspin_map": [int(value) for value in np.asarray(t2, dtype=np.int32).tolist()],
            "direction": "y",
        }
    else:
        result["y"] = {
            "available": False,
            "direction": "y",
            "reason": "circumference_y is false; no periodic Ty map is available.",
        }
    return result


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


def _translation_validation_report(
    geometry: Any,
    permutation: np.ndarray,
    *,
    axis: str,
) -> Dict[str, Any]:
    bond_set = {
        (min(int(i), int(j)), max(int(i), int(j)), str(gamma))
        for i, j, gamma in _bond_triplets(geometry)
    }
    missing: List[Dict[str, Any]] = []
    for i, j, gamma in _bond_triplets(geometry):
        mapped_i = int(permutation[int(i)])
        mapped_j = int(permutation[int(j)])
        mapped = (min(mapped_i, mapped_j), max(mapped_i, mapped_j), str(gamma))
        if mapped not in bond_set:
            missing.append(
                {
                    "source_bond": [int(i), int(j), str(gamma)],
                    "mapped_bond": [int(mapped_i), int(mapped_j), str(gamma)],
                }
            )
    return {
        "axis": str(axis),
        "periodic": bool(
            getattr(
                geometry,
                "circumference_x" if str(axis) == "x" else "circumference_y",
                False,
            )
        ),
        "bond_gamma_preserving": len(missing) == 0,
        "missing_or_mismatched_bonds": missing[:8],
        "missing_or_mismatched_bond_count": int(len(missing)),
    }


def quspin_tensor_basis_fused_translation_equivalence(
    *,
    model_family: str = "yao_lee",
    spin_orbital_tensor_basis: bool = True,
) -> Tuple[bool, str]:
    """Whether native QuSpin tensor-basis maps equal fused physical translation.

    For the spin-orbital Yao-Lee representation used here, ``tensor_basis`` is
    a product of a spin chain basis and an orbital chain basis.  Applying the
    same site map to both factors creates independent factor momentum labels,
    not the single diagonal momentum block of the fused physical site
    ``|S_i,T_i>``.  The latter is what the standard projector path implements.
    """
    if str(model_family).strip().lower() == "yao_lee" and bool(spin_orbital_tensor_basis):
        return (
            False,
            "QuSpin tensor_basis would impose spin-chain and orbital-chain translations separately; "
            "this is not equivalent to the fused physical-site translation required for spin-orbital Yao-Lee. "
            "Use ED_SYMMETRY_ENGINE=standard_projector for production Tx/Ty blocks.",
        )
    return (
        False,
        "No validated QuSpin-native fused translation equivalence is registered for this model.",
    )


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
    """Check x/y honeycomb translation maps and production QuSpin support.

    ``geometry_supported=True`` means the raw 1D map exists and preserves the
    honeycomb bond/gamma list. ``supported=True`` is stricter: it means the map
    can be used as a production native QuSpin symmetry for this backend.  The
    current spin-orbital tensor-basis route reports the valid maps but does not
    use them because they are not equivalent to fused physical-site
    translations.
    """
    support: Dict[str, Dict[str, Any]] = {}
    equivalence_ok, equivalence_reason = quspin_tensor_basis_fused_translation_equivalence()
    for axis, use_x, use_y in (("x", True, False), ("y", False, True)):
        try:
            t1, t2 = _validated_translation_blocks(
                geometry,
                use_translation_x_block=use_x,
                use_translation_y_block=use_y,
            )
            permutation = t1 if axis == "x" else t2
            if permutation is None:
                raise ValueError(f"No {axis}-translation map was generated.")
            validation = _translation_validation_report(geometry, permutation, axis=axis)
        except Exception as exc:
            support[axis] = {
                "supported": False,
                "geometry_supported": False,
                "bond_gamma_preserving": False,
                "tensor_basis_equivalent_to_fused_translation": False,
                "reason": str(exc),
            }
        else:
            support[axis] = {
                "supported": bool(equivalence_ok),
                "geometry_supported": True,
                "bond_gamma_preserving": bool(validation["bond_gamma_preserving"]),
                "commutes_with_uniform_yao_lee_hamiltonian_by_bond_check": bool(
                    validation["bond_gamma_preserving"]
                ),
                "tensor_basis_equivalent_to_fused_translation": bool(equivalence_ok),
                "reason": None if equivalence_ok else equivalence_reason,
                "map": [int(value) for value in np.asarray(permutation, dtype=np.int32).tolist()],
                "validation": validation,
            }
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
    equivalence_ok, equivalence_reason = quspin_tensor_basis_fused_translation_equivalence()
    if not equivalence_ok:
        return False, equivalence_reason
    return True, None


def _spin_flip_permutation(n_sites: int) -> np.ndarray:
    # Negative entries request spin inversion in QuSpin's general-basis maps.
    return -(np.arange(int(n_sites), dtype=np.int32) + 1)


def _quspin_zblock_from_parity(z2_target_parity: int) -> int:
    """Map user parity 0/1 to QuSpin spin-inversion zblock +/-1."""
    return 1 if int(z2_target_parity) % 2 == 0 else -1


def _is_spin_flip_z2_generator(z2_generator: str | None) -> bool:
    generator = str(z2_generator or "spin_flip").strip().lower().replace("-", "_")
    return generator in ("spin_flip", "spin_inversion", "zblock")


def _general_spin_basis_with_fallbacks(n_sites: int, **kwargs: Any) -> Any:
    """Build spin_basis_general while keeping compatibility with QuSpin variants."""
    _spin_basis_1d, spin_basis_general, _tensor_basis = _quspin_basis_api()
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
    use_sz_block: bool = False,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_generator: str | None = None,
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
    if bool(use_sz_block):
        raise ValueError(
            "QuSpin Yao-Lee basis cannot use a total-Sz block: total Sz is not conserved "
            "by the bond-directional Yao-Lee Hamiltonian. Use tau_z or full spin basis."
        )
    spin_basis_1d, _spin_basis_general, tensor_basis = _quspin_basis_api()
    if bool(use_reflection_block) or int(reflection_block) != 0:
        raise ValueError(
            "QuSpin reflection/C3 spatial blocks are forbidden for the bond-directional Yao-Lee "
            "Hamiltonian unless a gauge transformation that permutes x/y/z bonds is implemented."
        )
    if bool(use_z2_block):
        if not _is_spin_flip_z2_generator(z2_generator):
            raise ValueError(
                "QuSpin Yao-Lee Z2 currently implements the full-spin spin-flip block "
                f"with z2_generator='spin_flip'; got {z2_generator or 'None'}."
            )
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
    if any_translation_requested:
        if geometry is None:
            raise ValueError("QuSpin translation blocks require the full geometry object.")
        _validated_translation_blocks(
            geometry,
            use_translation_x_block=translation_x_requested,
            use_translation_y_block=translation_y_requested,
        )
        equivalence_ok, equivalence_reason = quspin_tensor_basis_fused_translation_equivalence()
        if not equivalence_ok:
            raise ValueError(equivalence_reason)
    if bool(use_z2_block) and any_translation_requested:
        raise ValueError("QuSpin spin-flip Z2 is not combined with custom 2D translation blocks.")

    # pauli=False makes x, y, z represent spin-1/2 operators S and tau rather
    # than Pauli matrices sigma.
    spin_kwargs: Dict[str, Any] = {"pauli": 0}
    orbital_kwargs: Dict[str, Any] = {"pauli": 0}
    if bool(use_sz_block):
        spin_kwargs["Nup"] = _nup_from_total_m2(n_sites, int(target_sz2), "spin")
    if bool(use_tau_z_block):
        orbital_kwargs["Nup"] = _nup_from_total_m2(n_sites, int(target_tz2), "orbital")
    if bool(use_z2_block):
        # This is the spin-sector global spin inversion supported directly by
        # QuSpin's spin_basis_1d. It acts on the full S Hilbert space and does
        # not require, or imply, total-Sz conservation.
        spin_kwargs["zblock"] = _quspin_zblock_from_parity(z2_target_parity)

    if any_translation_requested:
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
    if any_translation_requested:
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
    spin_dot_coefficient = float(coupling_j) * float(alpha) * float(beta)
    spin_gamma_coefficient = -2.0 * float(coupling_j) * float(beta)
    orbital_dot_coefficient = float(coupling_j) * float(beta)
    spin_dot_orbital_dot_coefficient = -float(coupling_j) * float(alpha)
    spin_gamma_orbital_dot_coefficient = 2.0 * float(coupling_j)
    constant_coefficient = -float(coupling_j) * float(beta) * float(beta)
    axis_pair = {"x": "xx", "y": "yy", "z": "zz"}
    orbital_dot_terms = (("+-", 0.5), ("-+", 0.5), ("zz", 1.0))

    for i, j, gamma in _bond_triplets(geometry):
        for spin_pair in ("xx", "yy", "zz"):
            _append_term(static, f"{spin_pair}|I", [spin_dot_coefficient, i, j, i])
            for orbital_pair, orbital_factor in orbital_dot_terms:
                _append_term(
                    static,
                    f"{spin_pair}|{orbital_pair}",
                    [spin_dot_orbital_dot_coefficient * orbital_factor, i, j, i, j],
                )
        spin_gamma_pair = axis_pair[gamma]
        _append_term(static, f"{spin_gamma_pair}|I", [spin_gamma_coefficient, i, j, i])
        for orbital_pair, orbital_factor in orbital_dot_terms:
            _append_term(static, f"I|{orbital_pair}", [orbital_dot_coefficient * orbital_factor, i, i, j])
            _append_term(
                static,
                f"{spin_gamma_pair}|{orbital_pair}",
                [spin_gamma_orbital_dot_coefficient * orbital_factor, i, j, i, j],
            )
        _append_term(static, "I|I", [constant_coefficient, i, i])
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
    use_sz_block: bool = False,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_generator: str | None = None,
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
    with profile_stage("QuSpin basis construction"):
        basis = build_quspin_yao_lee_basis(
            n_sites,
            geometry=geometry,
            use_sz_block=use_sz_block,
            target_sz2=target_sz2,
            use_tau_z_block=use_tau_z_block,
            target_tz2=target_tz2,
            use_z2_block=use_z2_block,
            z2_generator=z2_generator,
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
    with profile_stage("QuSpin Hamiltonian construction"):
        static = build_quspin_yao_lee_static_terms(
            geometry=geometry,
            alpha=alpha,
            beta=beta,
            coupling_j=coupling_j,
            external_field_terms=external_field_terms,
        )
        hamiltonian = _quspin_hamiltonian_class()
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
        with profile_stage("diagonalization"):
            matrix = quspin_hamiltonian_as_sparse_matrix(hamiltonian_operator)
        return (
            np.asarray([float(np.real(matrix[0, 0]))], dtype=float),
            np.ones((1, 1), dtype=np.complex128),
        )
    requested_count = max(1, int(eigenstate_count))
    if dimension <= 2 or requested_count >= dimension - 1:
        with profile_stage("diagonalization"):
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
    with profile_stage("diagonalization"):
        eigenvalues, eigenvectors = hamiltonian_operator.eigsh(k=k, which="SA")
    if show_progress:
        print(f"[quspin-ed] {label} eigsh finished in {time.perf_counter() - start:.2f}s")
    order = np.argsort(np.real(eigenvalues))
    return (
        np.asarray(np.real(eigenvalues[order]), dtype=float),
        np.asarray(eigenvectors[:, order], dtype=np.complex128),
    )


def _sanitize_quspin_yao_lee_blocks(
    *,
    use_tau_z_block: bool,
    use_z2_block: bool,
    z2_generator: str | None = None,
    use_translation_block: bool,
    use_translation_x_block: bool | None,
    use_translation_y_block: bool | None,
) -> Tuple[bool, bool, bool, bool | None, bool | None, List[str]]:
    """Keep only tested QuSpin block combinations for spin-orbital Yao-Lee ED."""
    warnings: List[str] = []
    z2 = bool(use_z2_block)
    translation = bool(use_translation_block)
    tx = use_translation_x_block
    ty = use_translation_y_block
    if z2:
        generator = str(z2_generator or "").strip()
        if not _is_spin_flip_z2_generator(generator):
            z2 = False
            warnings.append(
                "Dropped QuSpin Z2 because this backend currently implements only the spin_flip zblock generator."
            )
    translation_requested = bool(translation or bool(tx) or bool(ty))
    if translation_requested:
        _equivalence_ok, equivalence_reason = quspin_tensor_basis_fused_translation_equivalence()
        translation = False
        tx = False
        ty = False
        warnings.append(f"Dropped QuSpin translation blocks: {equivalence_reason}")
    return bool(use_tau_z_block), bool(z2), bool(translation), tx, ty, warnings


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
    use_sz_block: bool = False,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_generator: str | None = None,
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

    (
        use_tau_z_block,
        use_z2_block,
        use_translation_block,
        use_translation_x_block,
        use_translation_y_block,
        block_warnings,
    ) = _sanitize_quspin_yao_lee_blocks(
        use_tau_z_block=use_tau_z_block,
        use_z2_block=use_z2_block,
        z2_generator=z2_generator,
        use_translation_block=use_translation_block,
        use_translation_x_block=use_translation_x_block,
        use_translation_y_block=use_translation_y_block,
    )
    if show_progress:
        for warning in block_warnings:
            print(f"[quspin-ed] {warning}")
    field_terms = _field_terms(external_field_terms)
    if bool(use_sz_block):
        if show_progress:
            print("[quspin-ed] total Sz is not conserved by the Yao-Lee Hamiltonian; using the full spin basis.")
        use_sz_block = False
    transverse_field = _has_transverse_spin_field_terms(field_terms)
    scan_sz_sectors = bool(use_sz_block and _has_sz_zeeman_terms(field_terms) and not transverse_field)
    if transverse_field and bool(use_sz_block):
        if show_progress:
            print("[quspin-ed] transverse field breaks total Sz; using the full spin basis.")
        use_sz_block = False
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
                z2_generator=z2_generator,
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
    use_sz_block: bool = False,
    target_sz2: int = 0,
    use_tau_z_block: bool = False,
    target_tz2: int = 0,
    use_z2_block: bool = False,
    z2_generator: str | None = None,
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
    requested_z2_generator = z2_generator
    (
        use_tau_z_block,
        use_z2_block,
        use_translation_block,
        use_translation_x_block,
        use_translation_y_block,
        block_warnings,
    ) = _sanitize_quspin_yao_lee_blocks(
        use_tau_z_block=use_tau_z_block,
        use_z2_block=use_z2_block,
        z2_generator=z2_generator,
        use_translation_block=use_translation_block,
        use_translation_x_block=use_translation_x_block,
        use_translation_y_block=use_translation_y_block,
    )
    if show_progress:
        for warning in block_warnings:
            print(f"[quspin-ed] {warning}")
    field_terms = _field_terms(external_field_terms)
    if bool(use_sz_block):
        if show_progress:
            print("[quspin-ed] total Sz is not conserved by the Yao-Lee Hamiltonian; using the full spin basis.")
        use_sz_block = False
    transverse_field = _has_transverse_spin_field_terms(field_terms)
    scan_sz_sectors = bool(use_sz_block and _has_sz_zeeman_terms(field_terms) and not transverse_field)
    if transverse_field and bool(use_sz_block):
        if show_progress:
            print("[quspin-ed] transverse field breaks total Sz; using the full spin basis.")
        use_sz_block = False
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
                z2_generator=z2_generator,
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
    spin_basis_label = "spin_flip_z2" if bool(use_z2_block) else ("fixed Sz" if bool(use_sz_block) else "full")
    orbital_basis_label = "fixed tau_z" if bool(use_tau_z_block) else "full"
    basis_type = (
        "quspin_tensor_spin_z2_orbital_tz"
        if bool(use_z2_block) and bool(use_tau_z_block)
        else (
            "quspin_tensor_spin_z2_orbital_full"
            if bool(use_z2_block)
            else (
                "quspin_tensor_spin_u1_block_orbital_tz"
                if bool(use_sz_block) and bool(use_tau_z_block)
                else (
                    "quspin_tensor_spin_full_orbital_tz"
                    if bool(use_tau_z_block)
                    else "quspin_tensor_spin_full_orbital_full"
                )
            )
        )
    )
    spectrum: Dict[str, Any] = {
        "backend": "quspin",
        "symmetry_engine": "quspin_native",
        "basis": f"tensor_basis(spin={spin_basis_label}, orbital={orbital_basis_label})",
        "basis_type": basis_type,
        "use_sz_block": bool(use_sz_block),
        "target_sz2": int(target_sz2),
        "requested_use_sz_block": bool(requested_use_sz_block),
        "requested_target_sz2": int(requested_target_sz2),
        "use_tau_z_block": bool(use_tau_z_block),
        "target_tz2": int(target_tz2),
        "use_z2_block": bool(use_z2_block),
        "requested_use_z2_block": bool(requested_use_z2_block),
        "z2_generator": "spin_flip" if bool(use_z2_block) else None,
        "z2_kind": "spin_flip" if bool(use_z2_block) else None,
        "quspin_zblock": _quspin_zblock_from_parity(z2_target_parity) if bool(use_z2_block) else None,
        "requested_z2_generator": requested_z2_generator,
        "block_warnings": list(block_warnings),
        "z2_target_parity": int(z2_target_parity) % 2,
        "native_supported_symmetries": {
            "u1_tz": True,
            "spin_flip_z2_zero_field": True,
            "spin_pi_z": False,
            "translation": False,
            "combined_c3": False,
            "reason": (
                "QuSpin native Yao-Lee uses tensor_basis with a full spin basis and optional orbital Tz. "
                "The only tested Z2 is zero-field spin_flip. Fused translations and true combined "
                "spin-lattice C3 remain in the standard_projector path."
            ),
        },
        "use_translation_block": bool(translation_x_used or translation_y_used),
        "use_translation_x_block": bool(translation_x_used),
        "use_translation_y_block": bool(translation_y_used),
        "momentum_block_1": int(kx),
        "momentum_block_2": int(ky),
        "momentum_x_block": int(kx),
        "momentum_y_block": int(ky),
        "use_reflection_block": bool(use_reflection_block),
        "reflection_block": int(reflection_block),
        "block_warnings": list(block_warnings),
        "formula": (
            "H = -J sum_<ij>_gamma [alpha S_i.S_j - 2 S_i^gamma S_j^gamma - beta]"
            "[T_i.T_j - beta]"
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


# ----------------------------------------------------------------------
# Opt-in QuSpin environment validation
# ----------------------------------------------------------------------

_COMPAT_PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "quspin",
    "tenpy",
    "tenax",
    "quimb",
    "numba",
    "llvmlite",
)


def _compat_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _compat_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_compat_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _compat_json_safe(item())
        except Exception:
            pass
    return str(value)


def _compat_package_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    for package_name in _COMPAT_PACKAGE_NAMES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = None
        except Exception as exc:
            versions[package_name] = f"unavailable: {exc}"
    return versions


def _compat_run_step(name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        payload = fn()
        if not isinstance(payload, dict):
            payload = {"result": payload}
        payload.setdefault("status", "passed")
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
    payload["wall_time_seconds"] = float(time.perf_counter() - start)
    payload["name"] = str(name)
    return payload


def _validate_required_quspin_api() -> Dict[str, Any]:
    from quspin.basis import spin_basis_1d, spin_basis_general, tensor_basis  # type: ignore
    from quspin.operators import hamiltonian  # type: ignore
    import quspin.basis as quspin_basis  # type: ignore

    return {
        "spin_basis_1d": str(spin_basis_1d),
        "spin_basis_general": str(spin_basis_general),
        "tensor_basis": str(tensor_basis),
        "hamiltonian": str(hamiltonian),
        "has_user_basis": bool(hasattr(quspin_basis, "user_basis")),
        "user_basis": str(getattr(quspin_basis, "user_basis", None)),
        "has_basis_general": bool(hasattr(quspin_basis, "basis_general")),
    }


def _validate_spin_chain_translation() -> Dict[str, Any]:
    from quspin.basis import spin_basis_general  # type: ignore
    from quspin.operators import hamiltonian  # type: ignore

    length = 4
    translation = np.asarray([(site + 1) % length for site in range(length)], dtype=np.int32)
    basis = spin_basis_general(length, kblock=(translation, 0), pauli=0)
    zz_terms = [[1.0, site, (site + 1) % length] for site in range(length)]
    x_terms = [[0.25, site] for site in range(length)]
    ham = hamiltonian(
        [["zz", zz_terms], ["x", x_terms]],
        [],
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    eigenvalues = np.linalg.eigvalsh(ham.toarray())
    return {
        "length": int(length),
        "translation_map": [int(value) for value in translation.tolist()],
        "basis_dimension": int(basis.Ns),
        "ground_state_energy": float(eigenvalues[0]),
        "native_translation_maps_supported": True,
    }


def _validate_zblock_spin_flip() -> Dict[str, Any]:
    from quspin.basis import spin_basis_1d  # type: ignore
    from quspin.operators import hamiltonian  # type: ignore

    length = 4
    dimensions: Dict[str, int] = {}
    energies: Dict[str, float] = {}
    for parity in (-1, 1):
        basis = spin_basis_1d(L=length, zblock=parity, pauli=0)
        ham = hamiltonian(
            [["x", [[1.0, site] for site in range(length)]]],
            [],
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        dimensions[str(parity)] = int(basis.Ns)
        energies[str(parity)] = float(np.linalg.eigvalsh(ham.toarray())[0])
    return {
        "length": int(length),
        "parity_dimensions": dimensions,
        "parity_ground_state_energies": energies,
        "zblock_spin_flip_supported": True,
    }


def _validate_tensor_basis() -> Dict[str, Any]:
    from quspin.basis import spin_basis_1d, tensor_basis  # type: ignore

    length = 2
    spin_basis = spin_basis_1d(L=length, pauli=0)
    orbital_basis = spin_basis_1d(L=length, pauli=0)
    basis = tensor_basis(spin_basis, orbital_basis)
    return {
        "length": int(length),
        "spin_basis_dimension": int(spin_basis.Ns),
        "orbital_basis_dimension": int(orbital_basis.Ns),
        "tensor_basis_dimension": int(basis.Ns),
        "tensor_basis_supported": True,
    }


def _validate_backend_supported_spin_orbital_blocks() -> Dict[str, Any]:
    from models import build_lattice_geometry, build_model_spec

    geometry = build_lattice_geometry(
        "honeycomb",
        1,
        length_y=2,
        circumference_x=False,
        circumference_y=True,
    )
    model_spec = build_model_spec("1/2", "1/2", "yao_lee", "z")
    hamiltonian_operator, basis, static = build_quspin_yao_lee_hamiltonian(
        geometry=geometry,
        alpha=0.7,
        beta=0.2,
        coupling_j=1.0,
        use_sz_block=False,
        use_tau_z_block=True,
        target_tz2=0,
        use_z2_block=True,
        z2_generator="spin_flip",
        z2_target_parity=0,
        use_translation_block=False,
        external_field_terms=[],
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    return {
        "package_available": quspin_package_available(),
        "fused_translation_report": quspin_fused_translation_api_support_report(
            geometry,
            use_tau_z_block=True,
            use_z2_block=False,
            requested=True,
        ),
        "combined_c3_report": quspin_combined_c3_api_support_report(
            model_family=model_spec.model_family,
            phase_scan_requested=False,
        ),
        "supported_backend_basis": "tensor_basis(spin=spin_flip_z2, orbital=fixed tau_z)",
        "basis_dimension": int(basis.Ns),
        "static_term_count": int(len(static)),
        "sparse_shape": [int(value) for value in hamiltonian_operator.tocsr().shape],
    }


def _validate_yao_lee_quspin_vs_standard_ed() -> Dict[str, Any]:
    import ed_backend
    from models import build_lattice_geometry, build_model_spec

    geometry = build_lattice_geometry(
        "honeycomb",
        1,
        length_y=2,
        circumference_x=False,
        circumference_y=True,
    )
    model_spec = build_model_spec("1/2", "1/2", "yao_lee", "z")
    alpha = 0.7
    beta = 0.2
    coupling_j = 1.0
    standard_spectrum, _standard_vectors = ed_backend.run_small_cluster_exact_spectrum(
        geometry=geometry,
        model_spec=model_spec,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        external_field_terms=[],
        show_progress=False,
        solver="dense",
    )
    quspin_operator, quspin_basis, _static = build_quspin_yao_lee_hamiltonian(
        geometry=geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        use_sz_block=False,
        use_tau_z_block=False,
        use_z2_block=False,
        use_translation_block=False,
        external_field_terms=[],
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    quspin_ground_energy = float(
        np.linalg.eigvalsh(quspin_hamiltonian_as_sparse_matrix(quspin_operator).toarray())[0]
    )
    standard_ground_energy = float(standard_spectrum["ground_state_energy"])
    difference = abs(quspin_ground_energy - standard_ground_energy)
    tolerance = 1.0e-8
    return {
        "status": "passed" if difference <= tolerance else "failed",
        "geometry": {
            "lattice": "honeycomb",
            "length_x": 1,
            "length_y": 2,
            "circumference_x": False,
            "circumference_y": True,
            "number_of_sites": int(geometry.number_of_sites),
        },
        "parameters": {
            "alpha": float(alpha),
            "beta": float(beta),
            "coupling_j": float(coupling_j),
        },
        "standard_ed_ground_energy": standard_ground_energy,
        "quspin_ground_energy": quspin_ground_energy,
        "absolute_difference": float(difference),
        "tolerance": float(tolerance),
        "quspin_basis_dimension": int(quspin_basis.Ns),
    }


def run_quspin_compatibility_validation() -> Dict[str, Any]:
    """Validate QuSpin APIs and tiny Yao-Lee ED parity without changing packages."""
    report: Dict[str, Any] = {
        "validator": "quspin_backend.run_quspin_compatibility_validation",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "package_versions": _compat_package_versions(),
        "steps": {},
    }
    steps: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
        ("import_required_quspin_api", _validate_required_quspin_api),
        ("spin_chain_translation_block", _validate_spin_chain_translation),
        ("zblock_spin_flip", _validate_zblock_spin_flip),
        ("tensor_basis", _validate_tensor_basis),
        ("yao_lee_backend_supported_spin_orbital_blocks", _validate_backend_supported_spin_orbital_blocks),
        ("yao_lee_quspin_vs_standard_ed", _validate_yao_lee_quspin_vs_standard_ed),
    ]
    for name, fn in steps:
        report["steps"][name] = _compat_run_step(name, fn)
    report["passed"] = all(
        str(report["steps"].get(name, {}).get("status")) == "passed"
        for name, _fn in steps
    )
    return report


def _pip_freeze(timeout_seconds: float = 60.0) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "command": [sys.executable, "-m", "pip", "freeze"],
        "timeout_seconds": float(timeout_seconds),
        "status": "not_run",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": None,
        "timed_out": False,
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
        result.update(
            {
                "status": "completed",
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "status": "timeout",
                "timed_out": True,
                "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
                "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
                "error": f"timed out after {float(timeout_seconds):.1f}s",
            }
        )
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
    return result


def _write_text_file(output_folder: str, filename: str, text: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text)
    return filepath


def _requirements_header(report: Dict[str, Any], *, tested: bool) -> str:
    lines = [
        "# Generated by quspin_backend.run_quspin_compatibility_validation",
        "# No packages were installed, upgraded, pinned, or changed by this code.",
        f"# Python executable: {sys.executable}",
        f"# Validation passed: {bool(report.get('passed', False))}",
    ]
    if tested:
        lines.append("# This file was written only because the QuSpin compatibility validation passed.")
    else:
        lines.append("# Current environment freeze; this is not a recommendation.")
    return "\n".join(lines) + "\n"


def write_quspin_compatibility_requirement_files(
    report: Dict[str, Any],
    output_folder: str,
    *,
    write_current_freeze: bool,
    write_tested_freeze: bool,
) -> Dict[str, Any]:
    """Write optional requirements snapshots from the active environment."""
    files: Dict[str, Any] = {}
    if not (write_current_freeze or write_tested_freeze):
        return files
    freeze = _pip_freeze()
    files["pip_freeze"] = {
        "status": freeze.get("status"),
        "returncode": freeze.get("returncode"),
        "stderr": freeze.get("stderr"),
        "error": freeze.get("error"),
        "timed_out": freeze.get("timed_out"),
    }
    freeze_text = str(freeze.get("stdout") or "")
    if freeze.get("status") != "completed" or int(freeze.get("returncode") or 1) != 0:
        freeze_text = (
            "# pip freeze did not complete successfully.\n"
            f"# status: {freeze.get('status')}\n"
            f"# returncode: {freeze.get('returncode')}\n"
            f"# error: {freeze.get('error')}\n"
            "# stderr:\n"
            f"{freeze.get('stderr') or ''}\n"
            "# stdout:\n"
            f"{freeze.get('stdout') or ''}\n"
        )
    if write_current_freeze:
        files["requirements_current_freeze"] = _write_text_file(
            output_folder,
            "requirements-current-freeze.txt",
            _requirements_header(report, tested=False) + freeze_text,
        )
    if write_tested_freeze:
        if bool(report.get("passed", False)):
            files["requirements_quspin_tested"] = _write_text_file(
                output_folder,
                "requirements-quspin-tested.txt",
                _requirements_header(report, tested=True) + freeze_text,
            )
        else:
            files["requirements_quspin_tested"] = {
                "status": "not_written",
                "reason": "QuSpin compatibility validation did not pass.",
            }
    return files


def quspin_compatibility_cli_main(argv: Sequence[str] | None = None) -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Validate the active QuSpin environment against the Yao-Lee ED backend. "
            "This never installs or changes packages."
        )
    )
    parser.add_argument("--output-folder", default=os.path.join(script_dir, "outputs", "profiling"))
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--write-current-freeze", action="store_true")
    parser.add_argument("--write-tested-freeze", action="store_true")
    parser.add_argument("--write-files", action="store_true")
    args = parser.parse_args(argv)
    output_folder = os.path.abspath(os.path.expanduser(str(args.output_folder)))
    report = run_quspin_compatibility_validation()
    report["output_folder"] = output_folder
    report["files"] = write_quspin_compatibility_requirement_files(
        report,
        output_folder,
        write_current_freeze=bool(args.write_current_freeze or args.write_files),
        write_tested_freeze=bool(args.write_tested_freeze or args.write_files),
    )
    if args.json_path:
        json_path = os.path.abspath(os.path.expanduser(str(args.json_path)))
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(_compat_json_safe(report), file, indent=2, sort_keys=True)
        report["json_report"] = json_path
    print(json.dumps(_compat_json_safe(report), indent=2, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(quspin_compatibility_cli_main())
