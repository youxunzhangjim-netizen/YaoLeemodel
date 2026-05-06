#!/usr/bin/env python3
"""Tenax backend for the Yao-Lee driver.

This module owns Tenax AutoMPO construction, finite DMRG, iDMRG, and Tenax
correlation collection. General model/geometry physics stays in ``models.py``
and CLI orchestration stays in ``ylmodel_main.py``.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from analysis import (
    ENTROPY_ORDERS,
    _end_stage,
    _get_tqdm,
    _make_progress_bar,
    _start_stage,
    compute_tenax_infinite_mps_entropy_profile,
    find_dmrg_excited_state,
    get_tenax_api,
)
from models import (
    GeometryData,
    ModelSpec,
    _get_z2_symmetry_object,
    _normalize_symmetry_mode,
    _u1_charge_encoding_summary,
    _u1_basis_charge_table_for_model,
    _u1_encoded_phys_charges_for_model,
    _u1_encoded_target_charge,
    _validate_symmetry_conserving_terms,
    _z2_basis_charge_table_for_model,
    _z2_phys_charges_for_model,
    auto_mpo_pair_terms_for_bond_terms,
    build_site_ops,
    model_terms_for_bond,
    nonzero_auto_mpo_terms,
)

SYMMETRY_MODE_DEFAULT = "none"
U1_TARGET_TOTAL_SZ2_DEFAULT = 0
U1_TARGET_TOTAL_TZ2_DEFAULT = 0
Z2_TARGET_PARITY_DEFAULT = 0
STRICT_SYMMETRY_SELECTION_RULES_DEFAULT = True
SYMMETRY_MODE = SYMMETRY_MODE_DEFAULT
U1_TARGET_TOTAL_SZ2 = U1_TARGET_TOTAL_SZ2_DEFAULT
U1_TARGET_TOTAL_TZ2 = U1_TARGET_TOTAL_TZ2_DEFAULT
Z2_TARGET_PARITY = Z2_TARGET_PARITY_DEFAULT
STRICT_SYMMETRY_SELECTION_RULES = STRICT_SYMMETRY_SELECTION_RULES_DEFAULT


def _required_site_operator_names(terms: List[Tuple[Any, ...]]) -> List[str]:
    required = {"Id"}
    for term in terms:
        args = term[1:]
        for idx in range(0, len(args), 2):
            required.add(str(args[idx]))
    return sorted(required)


def _coerce_auto_mpo_coefficients(
    terms: List[Tuple[Any, ...]],
    *,
    complex_coefficients: bool,
) -> List[Tuple[Any, ...]]:
    coerced: List[Tuple[Any, ...]] = []
    for term in terms:
        if len(term) == 0:
            continue
        coeff = complex(term[0])
        if complex_coefficients:
            coerced.append((np.complex128(coeff), *term[1:]))
        elif abs(coeff.imag) <= 1e-12:
            coerced.append((float(coeff.real), *term[1:]))
        else:
            coerced.append((np.complex128(coeff), *term[1:]))
    return coerced


def _site_ops_subset(
    site_ops: Dict[str, np.ndarray],
    required_ops: List[str],
    dtype: Any,
) -> Dict[str, np.ndarray]:
    subset: Dict[str, np.ndarray] = {}
    for op_name in required_ops:
        if op_name not in site_ops:
            continue
        subset[op_name] = np.asarray(site_ops[op_name], dtype=dtype)
    return subset


def _real_site_ops_subset_if_possible(
    site_ops: Dict[str, np.ndarray],
    required_ops: List[str],
) -> Dict[str, np.ndarray] | None:
    subset: Dict[str, np.ndarray] = {}
    for op_name in required_ops:
        if op_name not in site_ops:
            continue
        op_array = np.asarray(site_ops[op_name])
        if np.max(np.abs(np.imag(op_array))) > 1e-12:
            return None
        subset[op_name] = np.asarray(np.real(op_array), dtype=np.float64)
    return subset


def _expand_sy_ty_pair_terms_to_real_ladder_ops(
    terms: List[Tuple[Any, ...]],
) -> List[Tuple[Any, ...]]:
    expanded: List[Tuple[Any, ...]] = []
    for term in terms:
        if len(term) == 5 and str(term[1]) == str(term[3]) and str(term[1]) in ("Sy", "Ty"):
            coeff = complex(term[0])
            op_plus, op_minus = ("Sp", "Sm") if str(term[1]) == "Sy" else ("Tp", "Tm")
            i_site = int(term[2])
            j_site = int(term[4])
            expanded.extend(
                [
                    (-0.25 * coeff, op_plus, i_site, op_plus, j_site),
                    (0.25 * coeff, op_plus, i_site, op_minus, j_site),
                    (0.25 * coeff, op_minus, i_site, op_plus, j_site),
                    (-0.25 * coeff, op_minus, i_site, op_minus, j_site),
                ]
            )
            continue
        expanded.append(term)
    return nonzero_auto_mpo_terms(expanded)


def _looks_like_tenax_complex_cast_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "Cannot cast ufunc 'add' output" in text
        and "complex128" in text
        and "float64" in text
    )


def _build_auto_mpo_from_terms(
    terms: List[Tuple[Any, ...]],
    length: int,
    site_ops: Dict[str, np.ndarray],
    *,
    symmetry_mode: str = "none",
    phys_charges: np.ndarray | None = None,
    strict_charge_conservation: bool = True,
) -> Any:
    terms = nonzero_auto_mpo_terms(terms)
    if len(terms) == 0:
        raise ValueError(
            "No non-zero AutoMPO terms were provided. Check coupling_j, external field "
            "settings, and symmetry filtering before building the Tenax MPO."
        )

    mode = _normalize_symmetry_mode(symmetry_mode)
    if mode == "z2":
        raise RuntimeError(
            "Tenax 0.2 AutoMPO cannot build a true Z2 block-sparse MPO: its "
            "symmetric AutoMPO path checks raw integer U1 charge transfers. "
            "Use symmetry_mode=none for full Yao-Lee x/y bonds, or use "
            "symmetry_mode=u1 with a U1-conserving Hamiltonian."
        )
    api = get_tenax_api()
    build_auto_mpo = api["build_auto_mpo"]
    auto_mpo_cls = api["AutoMPO"]
    use_symmetric_tensors = (mode != "none")
    mpo_dtype = np.float64 if mode == "u1" else np.complex128
    terms_for_build = terms
    required_ops = _required_site_operator_names(terms)
    site_ops_for_build = _site_ops_subset(site_ops, required_ops, np.complex128)
    if mode == "u1":
        real_subset = _real_site_ops_subset_if_possible(site_ops, required_ops)
        if real_subset is None:
            complex_ops = [
                op_name
                for op_name in required_ops
                if op_name in site_ops
                and np.max(np.abs(np.imag(np.asarray(site_ops[op_name])))) > 1e-12
            ]
            raise ValueError(
                "U1 symmetric real Tenax DMRG cannot use complex operators: "
                f"{', '.join(complex_ops)}."
            )
        site_ops_for_build = real_subset
        terms_for_build = _coerce_auto_mpo_coefficients(terms, complex_coefficients=False)
    else:
        real_candidate_terms = _coerce_auto_mpo_coefficients(
            _expand_sy_ty_pair_terms_to_real_ladder_ops(terms),
            complex_coefficients=False,
        )
        real_candidate_required_ops = _required_site_operator_names(real_candidate_terms)
        real_candidate_site_ops = _real_site_ops_subset_if_possible(
            site_ops,
            real_candidate_required_ops,
        )
        real_candidate_coefficients = all(
            abs(complex(term[0]).imag) <= 1e-12
            for term in real_candidate_terms
        )
        if real_candidate_site_ops is not None and real_candidate_coefficients:
            terms_for_build = real_candidate_terms
            required_ops = real_candidate_required_ops
            site_ops_for_build = real_candidate_site_ops
            mpo_dtype = np.float64
        else:
            terms_for_build = _coerce_auto_mpo_coefficients(terms, complex_coefficients=True)

    if use_symmetric_tensors:
        if phys_charges is None:
            raise ValueError(f"{mode.upper()} symmetric MPO build requires explicit phys_charges.")
        if strict_charge_conservation:
            _validate_symmetry_conserving_terms(terms, site_ops, phys_charges, mode)

    def _build_with_inputs(
        input_terms: List[Tuple[Any, ...]],
        input_site_ops: Dict[str, np.ndarray],
        input_dtype: Any,
    ) -> Any:
        if build_auto_mpo is None:
            raise RuntimeError("Tenax build_auto_mpo unavailable.")
        signature = inspect.signature(build_auto_mpo)
        kwargs: Dict[str, Any] = {"L": length}
        local_dim = int(next(iter(input_site_ops.values())).shape[0])
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
                kwargs["site_ops"] = input_site_ops
            # Our local operators include Sy/Ty, so terms are complex.
            # Dense mode stays complex-safe; U1 mode uses real Sp/Sm and z terms
            # because Tenax 0.2 symmetric DMRG casts Lanczos inner products to float.
            if "dtype" in signature.parameters:
                kwargs["dtype"] = input_dtype
            if use_symmetric_tensors:
                kwargs["symmetric"] = True
                kwargs["phys_charges"] = np.asarray(phys_charges, dtype=np.int32)
            return build_auto_mpo(input_terms, **kwargs)
        if auto_mpo_cls is None:
            raise RuntimeError(
                "The installed Tenax build_auto_mpo does not expose symmetric/phys_charges "
                "arguments, and AutoMPO fallback is unavailable for U1/Z2 construction."
            )
        raise RuntimeError("Tenax build_auto_mpo does not support the requested arguments.")

    def _build_with_autompo_class(
        input_terms: List[Tuple[Any, ...]],
        input_site_ops: Dict[str, np.ndarray],
        input_dtype: Any,
    ) -> Any:
        if auto_mpo_cls is None:
            raise RuntimeError("Tenax AutoMPO unavailable.")
        local_dim = int(next(iter(input_site_ops.values())).shape[0])
        signature = inspect.signature(auto_mpo_cls)
        kwargs: Dict[str, Any] = {}
        if "L" in signature.parameters:
            kwargs["L"] = length
        if "d" in signature.parameters:
            kwargs["d"] = local_dim
        if "site_ops" in signature.parameters:
            kwargs["site_ops"] = input_site_ops
        if "dtype" in signature.parameters:
            kwargs["dtype"] = input_dtype
        try:
            auto = auto_mpo_cls(**kwargs) if kwargs else auto_mpo_cls(L=length, d=local_dim)
        except TypeError:
            auto = auto_mpo_cls(length)
        for term in input_terms:
            auto += term
        try:
            if use_symmetric_tensors:
                return auto.to_mpo(
                    compress=True,
                    symmetric=True,
                    phys_charges=np.asarray(phys_charges, dtype=np.int32),
                    dtype=input_dtype,
                )
            return auto.to_mpo(compress=True, dtype=input_dtype)
        except TypeError:
            if use_symmetric_tensors:
                return auto.to_mpo(
                    symmetric=True,
                    phys_charges=np.asarray(phys_charges, dtype=np.int32),
                    dtype=input_dtype,
                )
            try:
                return auto.to_mpo(dtype=input_dtype)
            except TypeError:
                return auto.to_mpo()

    def _build_primary() -> Any:
        if build_auto_mpo is not None:
            try:
                return _build_with_inputs(terms_for_build, site_ops_for_build, mpo_dtype)
            except RuntimeError as exc:
                if auto_mpo_cls is None:
                    raise
                if "does not support the requested arguments" not in str(exc):
                    raise
        if auto_mpo_cls is not None:
            return _build_with_autompo_class(terms_for_build, site_ops_for_build, mpo_dtype)
        raise RuntimeError("Tenax provides neither build_auto_mpo nor AutoMPO.")

    def _build_real_ladder_fallback(original_exc: Exception) -> Any:
        real_terms = _coerce_auto_mpo_coefficients(
            _expand_sy_ty_pair_terms_to_real_ladder_ops(terms),
            complex_coefficients=False,
        )
        real_required_ops = _required_site_operator_names(real_terms)
        real_site_ops = _real_site_ops_subset_if_possible(site_ops, real_required_ops)
        if real_site_ops is None:
            raise RuntimeError(
                "Tenax hit its complex AutoMPO cast bug, and this Hamiltonian still "
                "requires complex one-site operators after the Sy/Ty ladder rewrite. "
                "Use no y-axis Zeeman field or patch Tenax AutoMPO to allocate complex buffers."
            ) from original_exc
        try:
            if build_auto_mpo is not None:
                return _build_with_inputs(real_terms, real_site_ops, np.float64)
            if auto_mpo_cls is not None:
                return _build_with_autompo_class(real_terms, real_site_ops, np.float64)
        except Exception as retry_exc:
            raise RuntimeError(
                "Tenax AutoMPO failed even after rewriting Sy/Ty pair terms to real "
                "ladder-operator form."
            ) from retry_exc
        raise RuntimeError("Tenax provides neither build_auto_mpo nor AutoMPO.") from original_exc

    try:
        return _build_primary()
    except TypeError as exc:
        if mode == "none" and _looks_like_tenax_complex_cast_error(exc):
            return _build_real_ladder_fallback(exc)
        if auto_mpo_cls is None or build_auto_mpo is None:
            raise
        try:
            return _build_with_autompo_class(terms_for_build, site_ops_for_build, mpo_dtype)
        except Exception:
            raise
    except Exception as exc:
        if mode != "none" or not _looks_like_tenax_complex_cast_error(exc):
            raise
        return _build_real_ladder_fallback(exc)

    raise RuntimeError("Tenax provides neither build_auto_mpo nor AutoMPO.")


def _empty_tenax_hamiltonian_message(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    field_terms: List[Tuple[float, str]],
) -> str:
    return (
        "No non-zero Hamiltonian terms were generated for the Tenax MPO. "
        f"sites={int(geometry.number_of_sites)}, bonds={len(geometry.bond_list)}, "
        f"model_family={model_spec.model_family}, spin_rep={model_spec.spin_rep}, "
        f"orbital_rep={model_spec.orbital_rep}, alpha={float(alpha):g}, "
        f"beta={float(beta):g}, coupling_j={float(coupling_j):g}, "
        f"external_field_terms={field_terms}. "
        "Set --coupling-j to a non-zero value, or use "
        "--external-field-treatment hamiltonian with a non-zero field."
    )


def build_tenax_model_mpo(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    symmetry_mode: str = "none",
    symmetry_phys_charges: np.ndarray | None = None,
    strict_charge_conservation: bool = True,
    show_progress: bool = True,
) -> Any:
    length = geometry.number_of_sites
    custom_ops = build_site_ops(model_spec)
    field_terms = list(external_field_terms or [])
    terms: List[Tuple[Any, ...]] = []

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(geometry.bond_list) + (length if field_terms else 0),
        desc="Tenax MPO terms",
        unit="item",
        leave=False,
    )
    for bond in geometry.bond_list:
        i, j, gamma = bond.i, bond.j, bond.gamma.lower()
        bond_terms = model_terms_for_bond(
            gamma,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx=jx,
            jy=jy,
            jz=jz,
        )
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

    for site in range(length):
        for coefficient, op_name in field_terms:
            terms.append((coefficient, op_name, site))
        if field_terms and progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    terms = nonzero_auto_mpo_terms(terms)
    if len(terms) == 0:
        raise ValueError(
            _empty_tenax_hamiltonian_message(
                geometry,
                model_spec,
                alpha,
                beta,
                coupling_j,
                field_terms,
            )
        )

    return _build_auto_mpo_from_terms(
        terms,
        length,
        custom_ops,
        symmetry_mode=symmetry_mode,
        phys_charges=symmetry_phys_charges,
        strict_charge_conservation=bool(strict_charge_conservation),
    )


# Backward-compat alias for older imports.
build_tenax_yao_lee_mpo = build_tenax_model_mpo


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
        tensor_dtype = np.float64
    elif mode == "z2":
        symmetry = _get_z2_symmetry_object()
        charge_modulus = 2
        target_charge = int(target_charge) % 2
        phys = np.asarray(phys_charges, dtype=np.int32) % 2
        tensor_dtype = np.float64
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

        left_label = "v_-1_0" if site == 0 else f"v{site - 1}_{site}"
        right_label = f"v{site}_{site + 1}"
        indices = (
            TensorIndex(symmetry, left, FlowDirection.IN, label=left_label),
            TensorIndex(symmetry, phys, FlowDirection.IN, label=f"p{site}"),
            TensorIndex(symmetry, right, FlowDirection.OUT, label=right_label),
        )
        tensor = SymmetricTensor.random_normal(
            indices,
            key=subkey,
            dtype=tensor_dtype,
            target=site_target,
        )
        tensors.append(tensor)

    mps = FiniteMPS.from_tensors(tensors, target_charge=int(target_charge))
    try:
        return mps.right_canonicalize()
    except Exception:
        return mps


def run_tenax_cylindrical_dmrg(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    max_bond_dimension: int,
    max_sweeps: int,
    random_seed: int,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    symmetry_mode: str = SYMMETRY_MODE,
    u1_target_total_sz2: int = U1_TARGET_TOTAL_SZ2,
    u1_target_total_tz2: int = U1_TARGET_TOTAL_TZ2,
    z2_target_parity: int = Z2_TARGET_PARITY,
    strict_symmetry_selection_rules: bool = STRICT_SYMMETRY_SELECTION_RULES,
    num_states: int = 1,
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

        mpo = build_tenax_model_mpo(
            geometry,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx=jx,
            jy=jy,
            jz=jz,
            external_field_terms=external_field_terms,
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
        if "num_states" in config_signature.parameters:
            config_kwargs["num_states"] = max(1, int(num_states))
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
        if sym_mode != "none":
            if show_progress:
                print(
                    f"[symmetry] Requested symmetry_mode={sym_mode} could not be imposed; "
                    "retrying Tenax DMRG with symmetry_mode=none."
                )
            mps_retry, mpo_retry, retry_info = run_tenax_cylindrical_dmrg(
                geometry=geometry,
                model_spec=model_spec,
                alpha=alpha,
                beta=beta,
                coupling_j=coupling_j,
                max_bond_dimension=max_bond_dimension,
                max_sweeps=max_sweeps,
                random_seed=random_seed,
                jx=jx,
                jy=jy,
                jz=jz,
                external_field_terms=external_field_terms,
                symmetry_mode="none",
                u1_target_total_sz2=u1_target_total_sz2,
                u1_target_total_tz2=u1_target_total_tz2,
                z2_target_parity=z2_target_parity,
                strict_symmetry_selection_rules=strict_symmetry_selection_rules,
                num_states=num_states,
                show_progress=show_progress,
            )
            retry_info["requested_symmetry_mode"] = sym_mode
            retry_info["symmetry_fallback_to_dense"] = True
            retry_info["symmetry_fallback_reason"] = str(exc)
            retry_info["strict_symmetry_selection_rules_requested"] = bool(strict_symmetry_selection_rules)
            if sym_mode == "u1":
                retry_info["requested_u1_target_sector"] = {
                    "total_Sz_times_2": int(u1_target_total_sz2),
                    "total_Tz_times_2": int(u1_target_total_tz2),
                }
                retry_info["u1_charge_encoding"] = _u1_charge_encoding_summary()
            if sym_mode == "z2":
                retry_info["requested_z2_target_sector"] = {
                    "global_parity": int(z2_target_parity) % 2
                }
            return mps_retry, mpo_retry, retry_info
        raise

    mps_out, dmrg_info = _extract_dmrg_result(result, mps)
    dmrg_info["symmetry_mode"] = sym_mode
    dmrg_info["symmetry_enabled"] = bool(symmetry_enabled)
    dmrg_info["u1_symmetry_enabled"] = bool(sym_mode == "u1")
    dmrg_info["z2_symmetry_enabled"] = bool(sym_mode == "z2")
    dmrg_info["num_states_requested"] = max(1, int(num_states))
    if symmetry_enabled and symmetry_phys_charges is not None:
        dmrg_info["symmetry_phys_charges"] = [
            int(val) for val in list(np.asarray(symmetry_phys_charges, dtype=np.int32))
        ]
        dmrg_info["symmetry_target_charge"] = int(symmetry_target_charge)
        if symmetry_basis_table is not None:
            dmrg_info["symmetry_basis_charge_table"] = symmetry_basis_table
        if sym_mode == "u1":
            dmrg_info["u1_charge_encoding"] = _u1_charge_encoding_summary()
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


def run_tenax_ed_guided_dmrg_excited_search(
    geometry: GeometryData,
    model_spec: ModelSpec,
    alpha: float,
    beta: float,
    coupling_j: float,
    reference_ground_energy: float,
    ed_ground_state_degeneracy: int | None,
    ed_ground_state_degeneracy_tolerance: float | None,
    ed_first_excited_energy: float | None,
    max_trials: int,
    max_bond_dimension: int,
    max_sweeps: int,
    random_seed: int,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_terms: List[Tuple[float, str]] | None = None,
    symmetry_mode: str = SYMMETRY_MODE,
    u1_target_total_sz2: int = U1_TARGET_TOTAL_SZ2,
    u1_target_total_tz2: int = U1_TARGET_TOTAL_TZ2,
    z2_target_parity: int = Z2_TARGET_PARITY,
    strict_symmetry_selection_rules: bool = STRICT_SYMMETRY_SELECTION_RULES,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run extra finite-DMRG starts, using ED degeneracy to choose the target level.

    Tenax 0.2 exposes ``DMRGConfig.num_states`` but returns a single optimized
    MPS/energy. This helper therefore records the requested ED-guided target
    state count and only reports a DMRG first-excited candidate when a restart
    actually converges above the ED-guided ground-manifold tolerance.
    """

    try:
        guide_degeneracy = int(ed_ground_state_degeneracy)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {
            "status": "skipped",
            "reason": "ED ground-state degeneracy was not resolved, so no DMRG excited-state guide is available.",
            "ground_state_degeneracy_check_enabled": True,
            "ground_state_degeneracy_status": "not_resolved",
        }
    if guide_degeneracy <= 0:
        return {
            "status": "skipped",
            "reason": f"Invalid ED ground-state degeneracy guide: {guide_degeneracy}.",
            "ground_state_degeneracy_check_enabled": True,
            "ground_state_degeneracy_status": "not_resolved",
        }

    target_state_count = max(2, guide_degeneracy + 1)
    trial_count = max(1, min(int(max_trials), target_state_count + 1))
    reference_ground = float(reference_ground_energy)
    degeneracy_tolerance = max(
        1e-8 * max(1.0, abs(reference_ground)),
        float(ed_ground_state_degeneracy_tolerance or 0.0),
    )

    stage_start = _start_stage("ED-guided finite-DMRG excited search", show_progress)
    trial_records: List[Dict[str, Any]] = []
    for trial_index in range(trial_count):
        trial_seed = int(random_seed) + 104729 * (trial_index + 1)
        try:
            _trial_mps, _trial_mpo, trial_info = run_tenax_cylindrical_dmrg(
                geometry=geometry,
                model_spec=model_spec,
                alpha=alpha,
                beta=beta,
                coupling_j=coupling_j,
                max_bond_dimension=max_bond_dimension,
                max_sweeps=max_sweeps,
                random_seed=trial_seed,
                jx=jx,
                jy=jy,
                jz=jz,
                external_field_terms=external_field_terms,
                symmetry_mode=symmetry_mode,
                u1_target_total_sz2=u1_target_total_sz2,
                u1_target_total_tz2=u1_target_total_tz2,
                z2_target_parity=z2_target_parity,
                strict_symmetry_selection_rules=strict_symmetry_selection_rules,
                num_states=target_state_count,
                show_progress=show_progress,
            )
            trial_records.append(
                {
                    "trial_index": int(trial_index),
                    "seed": int(trial_seed),
                    "status": "completed",
                    "energy": float(trial_info["E"]),
                    "converged": trial_info.get("converged"),
                    "sweeps_done": trial_info.get("sweeps_done"),
                    "symmetry_mode": trial_info.get("symmetry_mode"),
                    "num_states_requested": trial_info.get("num_states_requested", target_state_count),
                }
            )
        except Exception as exc:
            trial_records.append(
                {
                    "trial_index": int(trial_index),
                    "seed": int(trial_seed),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    successful_energies = [
        float(record["energy"])
        for record in trial_records
        if record.get("status") == "completed" and np.isfinite(float(record["energy"]))
    ]
    if len(successful_energies) == 0:
        _end_stage("ED-guided finite-DMRG excited search", stage_start, show_progress)
        return {
            "status": "failed",
            "reason": "All ED-guided finite-DMRG excited-search trials failed.",
            "ground_state_degeneracy_check_enabled": True,
            "ground_state_degeneracy": int(guide_degeneracy),
            "ground_state_degeneracy_status": "ed_guided",
            "ground_state_degeneracy_tolerance": float(degeneracy_tolerance),
            "target_state_count": int(target_state_count),
            "trials": trial_records,
        }

    all_ground_candidates = [reference_ground] + successful_energies
    best_ground = float(min(all_ground_candidates))
    threshold = best_ground + float(degeneracy_tolerance)
    excited_candidates = [energy for energy in successful_energies if energy > threshold]
    first_excited = None
    if excited_candidates:
        if ed_first_excited_energy is not None and np.isfinite(float(ed_first_excited_energy)):
            ed_e1 = float(ed_first_excited_energy)
            first_excited = float(min(excited_candidates, key=lambda value: abs(value - ed_e1)))
        else:
            first_excited = float(min(excited_candidates))
    spectral_gap = (
        float(first_excited - best_ground)
        if first_excited is not None
        else None
    )
    found_lower_ground = bool(best_ground < reference_ground - degeneracy_tolerance)
    status = "completed" if first_excited is not None else "not_found"
    _end_stage("ED-guided finite-DMRG excited search", stage_start, show_progress)
    return {
        "status": status,
        "search_method": "ed_degeneracy_guided_multistart_dmrg",
        "exact_orthogonal_targeting": False,
        "method_note": (
            "ED degeneracy sets the requested Tenax num_states and the ground-manifold "
            "tolerance. Installed Tenax returns one variational DMRG state per run, so "
            "the reported DMRG first-excited value is a restart candidate, not an exact "
            "orthogonally targeted excitation."
        ),
        "ground_state_energy": float(best_ground),
        "ground_state_energy_per_site": float(best_ground / geometry.number_of_sites),
        "reference_dmrg_ground_state_energy": float(reference_ground),
        "found_lower_ground_energy_than_original_dmrg": found_lower_ground,
        "ground_state_degeneracy_check_enabled": True,
        "ground_state_degeneracy": int(guide_degeneracy),
        "ground_state_degeneracy_status": "ed_guided",
        "ground_state_degeneracy_tolerance": float(degeneracy_tolerance),
        "ground_state_degeneracy_is_lower_bound": False,
        "target_state_count": int(target_state_count),
        "trial_count": int(trial_count),
        "successful_trial_count": int(len(successful_energies)),
        "first_excited_energy": first_excited,
        "first_excited_energy_per_site": (
            float(first_excited / geometry.number_of_sites)
            if first_excited is not None
            else None
        ),
        "spectral_gap": spectral_gap,
        "ed_guided_first_excited_reference": (
            float(ed_first_excited_energy)
            if ed_first_excited_energy is not None
            else None
        ),
        "trials": trial_records,
    }


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
        finite_n_sites_for_profile = int(diagnostics.get("n_sites_finite_mpo", 0))
        entanglement_profile = compute_tenax_infinite_mps_entropy_profile(
            mps=getattr(result, "mps"),
            sites_per_idmrg_site=sites_per_idmrg_site,
            finite_n_sites=finite_n_sites_for_profile if finite_n_sites_for_profile > 0 else None,
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
    if hasattr(mps, "expectation_value"):
        return complex(mps.expectation_value(mpo_ij))
    if hasattr(mps, "expectation"):
        return complex(mps.expectation(mpo_ij))
    api = get_tenax_api()
    exp_fn = api["expectation"]
    if callable(exp_fn):
        return complex(exp_fn(mpo_ij, mps))
    raise RuntimeError(
        "No expectation evaluator found. Tenax must provide expectation(...) or MPS expectation methods."
    )


def collect_uniform_z_observables_from_tenax(
    mps: Any,
    geometry: GeometryData,
    model_spec: ModelSpec,
) -> Dict[str, float]:
    n_sites = int(geometry.number_of_sites)
    custom_ops = build_site_ops(model_spec)
    api = get_tenax_api()
    one_site_expectation = api.get("expectation_value", None)
    totals = {"spin_z_per_site": 0.0j, "orbital_z_per_site": 0.0j}
    for site in range(n_sites):
        if callable(one_site_expectation):
            totals["spin_z_per_site"] += complex(one_site_expectation(mps, custom_ops["Sz"], site))
            totals["orbital_z_per_site"] += complex(one_site_expectation(mps, custom_ops["Tz"], site))
        else:
            spin_mpo = _build_auto_mpo_from_terms([(1.0, "Sz", site)], n_sites, custom_ops)
            orbital_mpo = _build_auto_mpo_from_terms([(1.0, "Tz", site)], n_sites, custom_ops)
            totals["spin_z_per_site"] += evaluate_expectation_value(spin_mpo, mps)
            totals["orbital_z_per_site"] += evaluate_expectation_value(orbital_mpo, mps)
    return {
        key: float(np.real(value) / float(max(1, n_sites)))
        for key, value in totals.items()
    }


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
