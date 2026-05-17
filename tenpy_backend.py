#!/usr/bin/env python3
"""TeNPy site/model template for the spin-orbital Yao-Lee model.

The physical local basis is fixed to

    0: Sup_Oup
    1: Sup_Odown
    2: Sdown_Oup
    3: Sdown_Odown

For the Yao-Lee model the safe conserved TeNPy charge is ``2*Tz`` with local
charges ``[+1, -1, +1, -1]``. Spin operators are neutral under this orbital
U(1), including Hamiltonian Zeeman terms that act only on spin.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, List

import numpy as np

from analysis import (
    DEFAULT_PHASE_CLASSIFIER_THRESHOLDS,
    _end_stage,
    _classify_phase_from_diagnostics,
    _make_progress_bar,
    _phase_observable_diagnostics,
    _start_stage,
)
from tenpy.linalg import np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site

try:
    from models import (
        GeometryData,
        _normalize_symmetry_mode,
        _u1_phys_charges_for_model,
        _validate_symmetry_conserving_terms,
        all_high_symmetry_structure_factors as _models_all_high_symmetry_structure_factors,
        build_honeycomb_cylinder_geometry,
        build_model_spec,
        build_site_ops,
        honeycomb_plaquette_flux_operators,
        plaquette_flux_close_to_target,
        select_honeycomb_plaquette_flux_operator,
        yao_lee_u1_two_site_terms_for_bond,
    )
    from ed_backend import (
        all_bond_energies as _ed_all_bond_energies,
        build_spin_orbital_scalar_correlations as _ed_scalar_correlations,
    )
except Exception:  # pragma: no cover - useful if this file is imported as a package module.
    from .models import (  # type: ignore
        GeometryData,
        _normalize_symmetry_mode,
        _u1_phys_charges_for_model,
        _validate_symmetry_conserving_terms,
        all_high_symmetry_structure_factors as _models_all_high_symmetry_structure_factors,
        build_honeycomb_cylinder_geometry,
        build_model_spec,
        build_site_ops,
        honeycomb_plaquette_flux_operators,
        plaquette_flux_close_to_target,
        select_honeycomb_plaquette_flux_operator,
        yao_lee_u1_two_site_terms_for_bond,
    )
    from .ed_backend import (  # type: ignore
        all_bond_energies as _ed_all_bond_energies,
        build_spin_orbital_scalar_correlations as _ed_scalar_correlations,
    )


DEFAULT_MODEL_SPEC = build_model_spec("1/2", "1/2", "yao_lee", "z")
ENTROPY_ORDERS = (1, 2, 3, 4)
_TENPY_PROGRESS_LOGGING_CONFIGURED = False
_TENPY_SWEEP_PROGRESS_BAR: Any | None = None
_TENPY_SWEEP_PROGRESS_LAST = 0
TENPY_STABLE_MIN_SWEEPS = 60
# For TwoSiteDMRGEngine, TeNPy maps mixer=True to DensityMatrixMixer, so name
# SubspaceExpansion explicitly to get the requested subspace-expansion mixer.
TENPY_STABLE_MIXER = "SubspaceExpansion"
TENPY_STABLE_MIXER_PARAMS = {"amplitude": 1.0e-4, "decay": 1.2, "disable_after": 40}
TENPY_STABLE_TRUNC_CUT = 1.0e-8


class _SuppressTenpyParameterReadFilter(logging.Filter):
    """Hide verbose TeNPy Config 'reading ...' records from stdout."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.name).startswith("tenpy.tools.params")


class _ConciseTenpyProgressFilter(logging.Filter):
    """Keep TeNPy console progress compact while preserving warnings/errors."""

    _energy_pattern = re.compile(r"energy=([^,\n]+)")
    _max_entropy_pattern = re.compile(r"max S=([^,\n]+)")
    _sweep_pattern = re.compile(r"checkpoint after sweep\s+(\d+)")
    _finished_pattern = re.compile(r"finished after\s+(\d+)\s+sweeps")

    def filter(self, record: logging.LogRecord) -> bool:
        global _TENPY_SWEEP_PROGRESS_BAR, _TENPY_SWEEP_PROGRESS_LAST
        name = str(record.name)
        if not name.startswith("tenpy"):
            return True
        if name.startswith("tenpy.tools.params"):
            return False
        if record.levelno >= logging.WARNING:
            return True

        message = record.getMessage()
        if name.startswith("tenpy.algorithms.dmrg"):
            sweep_match = self._sweep_pattern.search(message)
            if sweep_match is not None:
                sweep_index = int(sweep_match.group(1))
                pieces = [f"sweep={sweep_index}"]
                energy_match = self._energy_pattern.search(message)
                if energy_match is not None:
                    pieces.append(f"E={energy_match.group(1).strip()}")
                entropy_match = self._max_entropy_pattern.search(message)
                if entropy_match is not None:
                    pieces.append(f"Smax={entropy_match.group(1).strip()}")
                if _TENPY_SWEEP_PROGRESS_BAR is not None:
                    if (
                        _TENPY_SWEEP_PROGRESS_BAR.total is not None
                        and sweep_index > int(_TENPY_SWEEP_PROGRESS_BAR.total)
                    ):
                        _TENPY_SWEEP_PROGRESS_BAR.total = sweep_index
                    delta = max(0, sweep_index - int(_TENPY_SWEEP_PROGRESS_LAST))
                    if delta > 0:
                        _TENPY_SWEEP_PROGRESS_BAR.update(delta)
                        _TENPY_SWEEP_PROGRESS_LAST = sweep_index
                    postfix = {}
                    if energy_match is not None:
                        postfix["E"] = energy_match.group(1).strip()
                    if entropy_match is not None:
                        postfix["Smax"] = entropy_match.group(1).strip()
                    if postfix:
                        _TENPY_SWEEP_PROGRESS_BAR.set_postfix(postfix, refresh=True)
                    return False
                record.name = "tenpy-dmrg"
                record.msg = "checkpoint: " + ", ".join(pieces)
                record.args = ()
                return True
            if "finished after" in message:
                finished_match = self._finished_pattern.search(message)
                if finished_match is not None and _TENPY_SWEEP_PROGRESS_BAR is not None:
                    sweep_index = int(finished_match.group(1))
                    if (
                        _TENPY_SWEEP_PROGRESS_BAR.total is not None
                        and sweep_index > int(_TENPY_SWEEP_PROGRESS_BAR.total)
                    ):
                        _TENPY_SWEEP_PROGRESS_BAR.total = sweep_index
                    delta = max(0, sweep_index - int(_TENPY_SWEEP_PROGRESS_LAST))
                    if delta > 0:
                        _TENPY_SWEEP_PROGRESS_BAR.update(delta)
                        _TENPY_SWEEP_PROGRESS_LAST = sweep_index
                record.name = "tenpy-dmrg"
                record.msg = message.strip().splitlines()[0]
                record.args = ()
                return True
            return False

        if name.startswith("tenpy.algorithms.mps_common"):
            if "Converged" in message or "Maximum number of sweeps reached" in message:
                record.name = "tenpy-dmrg"
                record.msg = message.strip().splitlines()[0]
                record.args = ()
                return True
            return False

        if name.startswith("tenpy.networks.mps"):
            warning_like = (
                "not" in message.lower()
                or "significantly smaller chi" in message
                or "renormalized the TransferMatrix" in message
            )
            if warning_like:
                record.name = "tenpy-warning"
                record.msg = message.strip().splitlines()[0]
                record.args = ()
                return True
            return False

        return False


def _configure_tenpy_progress_logging(enabled: bool) -> None:
    """Route TeNPy sweep status messages to stdout when progress is enabled."""
    global _TENPY_PROGRESS_LOGGING_CONFIGURED
    if not bool(enabled):
        return
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(name)s] %(message)s",
            stream=sys.stdout,
        )
    root.setLevel(logging.INFO)
    for handler in root.handlers:
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        if handler.level == logging.NOTSET or handler.level > logging.INFO:
            handler.setLevel(logging.INFO)
        if not any(isinstance(item, _SuppressTenpyParameterReadFilter) for item in handler.filters):
            handler.addFilter(_SuppressTenpyParameterReadFilter())
        if not any(isinstance(item, _ConciseTenpyProgressFilter) for item in handler.filters):
            handler.addFilter(_ConciseTenpyProgressFilter())
    for logger_name in (
        "tenpy",
        "tenpy.algorithms",
        "tenpy.algorithms.dmrg",
        "tenpy.algorithms.mps_common",
    ):
        logging.getLogger(logger_name).setLevel(logging.INFO)
    logging.getLogger("tenpy.tools.params").setLevel(logging.WARNING)
    if not _TENPY_PROGRESS_LOGGING_CONFIGURED:
        print("[progress] TeNPy sweep/checkpoint logging enabled; parameter-read details go only to summary data.")
        _TENPY_PROGRESS_LOGGING_CONFIGURED = True


def _start_tenpy_sweep_progress(enabled: bool, total_sweeps: int, desc: str) -> Any | None:
    """Create the live sweep progress bar updated by TeNPy checkpoint logs."""
    global _TENPY_SWEEP_PROGRESS_BAR, _TENPY_SWEEP_PROGRESS_LAST
    if _TENPY_SWEEP_PROGRESS_BAR is not None:
        _TENPY_SWEEP_PROGRESS_BAR.close()
    _TENPY_SWEEP_PROGRESS_LAST = 0
    _TENPY_SWEEP_PROGRESS_BAR = _make_progress_bar(
        enabled=enabled,
        total=max(1, int(total_sweeps)),
        desc=desc,
        unit="sweep",
        leave=True,
    )
    return _TENPY_SWEEP_PROGRESS_BAR


def _finish_tenpy_sweep_progress(progress_bar: Any | None) -> None:
    """Close the active TeNPy sweep progress bar."""
    global _TENPY_SWEEP_PROGRESS_BAR, _TENPY_SWEEP_PROGRESS_LAST
    if progress_bar is not None:
        try:
            if progress_bar.total is not None and int(progress_bar.n) < int(progress_bar.total):
                progress_bar.total = max(1, int(progress_bar.n))
            progress_bar.refresh()
        except Exception:
            pass
        progress_bar.close()
    if progress_bar is _TENPY_SWEEP_PROGRESS_BAR:
        _TENPY_SWEEP_PROGRESS_BAR = None
        _TENPY_SWEEP_PROGRESS_LAST = 0


def _stable_chi_list(max_bond_dimension: int) -> dict[int, int]:
    """Return a conservative TeNPy chi ramp capped by the requested chi_max."""
    chi_max = max(1, int(max_bond_dimension))
    ramp_targets = [32, 64, 96]
    chi_list: dict[int, int] = {}
    for sweep, target in zip((0, 10, 20), ramp_targets):
        chi = min(chi_max, int(target))
        if len(chi_list) == 0 or chi > max(chi_list.values()):
            chi_list[int(sweep)] = int(chi)
    if max(chi_list.values()) < chi_max:
        chi_list[30] = int(chi_max)
    return chi_list


def _stable_tenpy_dmrg_params(
    max_bond_dimension: int,
    requested_sweeps: int,
    truncation_cutoff: float,
    svd_min: float | None = None,
) -> dict[str, Any]:
    """Build DMRG/iDMRG options with the cylinder-stability settings enabled."""
    chi_max = max(1, int(max_bond_dimension))
    max_sweeps = max(TENPY_STABLE_MIN_SWEEPS, int(requested_sweeps))
    svd_min_value = float(truncation_cutoff) if svd_min is None else float(svd_min)
    if not np.isfinite(svd_min_value) or svd_min_value < 0.0:
        raise ValueError(f"svd_min must be a nonnegative finite value; got {svd_min!r}.")
    trunc_cut = max(float(truncation_cutoff), TENPY_STABLE_TRUNC_CUT)
    return {
        "mixer": TENPY_STABLE_MIXER,
        "mixer_params": dict(TENPY_STABLE_MIXER_PARAMS),
        "diag_method": "lanczos",
        "lanczos_params": {"N_min": 2, "N_max": 40},
        "max_trunc_err": None,
        "norm_tol": None,
        "max_sweeps": int(max_sweeps),
        "N_sweeps_check": 1,
        "chi_list": _stable_chi_list(chi_max),
        "trunc_params": {
            "chi_max": int(chi_max),
            "svd_min": float(svd_min_value),
            "trunc_cut": float(trunc_cut),
        },
    }


def _canonicalize_after_dmrg(psi: MPS) -> str | None:
    """Force the optimized MPS back into canonical form after TeNPy returns."""
    try:
        psi.canonical_form()
    except Exception as exc:
        return str(exc)
    return None


def _run_dmrg_with_sweep_progress(
    psi: MPS,
    model: Any,
    options: dict[str, Any],
    *,
    show_progress: bool,
    desc: str,
    expected_sweeps: int,
) -> dict[str, Any]:
    """Run TeNPy DMRG with a tqdm sweep bar driven by checkpoint logs."""
    progress_bar = _start_tenpy_sweep_progress(show_progress, expected_sweeps, desc)
    try:
        info = dmrg.run(psi, model, options)
        canonicalization_warning = _canonicalize_after_dmrg(psi)
        if canonicalization_warning is not None:
            info["post_run_canonical_form_warning"] = canonicalization_warning
        return info
    finally:
        _finish_tenpy_sweep_progress(progress_bar)


def _tenpy_conserve_from_symmetry_reductions(symmetry_reductions: Any | None = None) -> str | None:
    """Map the shared symmetry report onto the local TeNPy site charge."""
    if isinstance(symmetry_reductions, dict):
        if bool(symmetry_reductions.get("use_tau_z_block", False)):
            return "Tz"
        if bool(symmetry_reductions.get("use_sz_block", False)):
            return "Sz"
        return None
    if isinstance(symmetry_reductions, (list, tuple, set)):
        reductions = {str(item).strip().lower() for item in symmetry_reductions}
        if reductions.intersection({"tz", "u1_tz", "tau_z"}):
            return "Tz"
        if reductions.intersection({"sz", "u1_sz", "spin_z"}):
            return "Sz"
    return None


def _tenpy_symmetry_mode_from_conserve(conserve: str | None) -> str:
    if conserve == "Tz":
        return "u1_tz"
    if conserve == "Sz":
        return "u1_sz"
    return "none"


def _yao_lee_auto_terms_for_tenpy_symmetry_check(
    geometry: GeometryData,
    alpha: float,
    beta: float,
    coupling_j: float,
    *,
    infinite_x: bool,
    external_field_terms: list[tuple[float, str]],
) -> list[tuple[Any, ...]]:
    """Build the same Yao-Lee terms used by TeNPy, for shared charge validation."""
    terms: list[tuple[Any, ...]] = []

    def append_bond_terms(i_site: int, j_site: int, gamma: str) -> None:
        for coefficient, op_i, op_j in yao_lee_u1_two_site_terms_for_bond(
            str(gamma),
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=float(coupling_j),
        ):
            if abs(complex(coefficient)) <= 1.0e-14:
                continue
            if int(i_site) <= int(j_site):
                terms.append((coefficient, str(op_i), int(i_site), str(op_j), int(j_site)))
            else:
                terms.append((coefficient, str(op_j), int(j_site), str(op_i), int(i_site)))

    for bond in geometry.bond_list:
        append_bond_terms(int(bond.i), int(bond.j), str(bond.gamma))
    if bool(infinite_x):
        for i_site, j_site, gamma in infinite_x_boundary_bonds(geometry):
            append_bond_terms(int(i_site), int(j_site), str(gamma))

    for site in range(int(geometry.number_of_sites)):
        for coefficient, op_name in external_field_terms:
            if abs(float(coefficient)) > 1.0e-14:
                terms.append((float(coefficient), str(op_name), int(site)))
    return terms


def _validated_tenpy_yao_lee_conserve(
    requested_conserve: str | None,
    geometry: GeometryData,
    alpha: float,
    beta: float,
    coupling_j: float,
    *,
    infinite_x: bool,
    external_field_terms: list[tuple[float, str]],
) -> tuple[str | None, list[str]]:
    """Apply the shared term-level validator to TeNPy's local Yao-Lee charge."""
    mode = _normalize_symmetry_mode(_tenpy_symmetry_mode_from_conserve(requested_conserve))
    if mode == "none":
        return None, []
    if mode == "u1_sz":
        return None, [
            "Requested TeNPy U1_Sz symmetry was dropped: total S^z is not conserved by the Yao-Lee Hamiltonian."
        ]
    terms = _yao_lee_auto_terms_for_tenpy_symmetry_check(
        geometry,
        alpha,
        beta,
        coupling_j,
        infinite_x=bool(infinite_x),
        external_field_terms=external_field_terms,
    )
    try:
        _validate_symmetry_conserving_terms(
            terms,
            build_site_ops(DEFAULT_MODEL_SPEC),
            _u1_phys_charges_for_model(DEFAULT_MODEL_SPEC, mode),
            mode,
        )
    except Exception as exc:
        return None, [
            f"Requested TeNPy {mode} symmetry is not conserved by the Yao-Lee Hamiltonian; "
            f"using dense/no-symmetry tensors instead. Validator detail: {exc}"
        ]
    return requested_conserve, []


class YaoLeeSite(Site):
    """One d=4 spin-1/2 tensor orbital-1/2 site."""

    state_labels = ["Sup_Oup", "Sup_Odown", "Sdown_Oup", "Sdown_Odown"]

    def __init__(self, conserve: str | None = "Tz", sort_charge: bool = False) -> None:
        if conserve in ("Tz", "tz", "tau_z", "tau", "U1_Tz", "u1_tz"):
            conserve_text = "Tz"
        elif conserve in ("Sz", "sz", "U1", "u1", "U1_Sz", "u1_sz", True):
            conserve_text = "Sz"
        else:
            conserve_text = "None"
        if conserve_text == "Tz":
            chinfo = npc.ChargeInfo([1], ["2*Tz"])
            leg = npc.LegCharge.from_qflat(chinfo, [1, -1, 1, -1])
        elif conserve_text == "Sz":
            chinfo = npc.ChargeInfo([1], ["2*Sz"])
            leg = npc.LegCharge.from_qflat(chinfo, [1, 1, -1, -1])
        else:
            leg = npc.LegCharge.from_trivial(4)

        spin_up_down = {
            "Sp": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
            "Sm": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
            "Sz": np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128),
            "Sx": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128),
            "Sy": np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128),
        }
        orbital_up_down = {
            "tau_p": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
            "tau_m": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
            "tau_z": np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128),
            "tau_x": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128),
            "tau_y": np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128),
        }
        spin_id = np.eye(2, dtype=np.complex128)
        orbital_id = np.eye(2, dtype=np.complex128)

        ops: dict[str, np.ndarray] = {
            "Sp": np.kron(spin_up_down["Sp"], orbital_id),
            "Sm": np.kron(spin_up_down["Sm"], orbital_id),
            "Sz": np.kron(spin_up_down["Sz"], orbital_id),
            "tau_p": np.kron(spin_id, orbital_up_down["tau_p"]),
            "tau_m": np.kron(spin_id, orbital_up_down["tau_m"]),
            "tau_z": np.kron(spin_id, orbital_up_down["tau_z"]),
        }
        if conserve_text != "Tz":
            ops["tau_x"] = np.kron(spin_id, orbital_up_down["tau_x"])
            ops["tau_y"] = np.kron(spin_id, orbital_up_down["tau_y"])
        if conserve_text != "Sz":
            ops["Sx"] = np.kron(spin_up_down["Sx"], orbital_id)
            ops["Sy"] = np.kron(spin_up_down["Sy"], orbital_id)
        alias_pairs = {
            "Tp": "tau_p",
            "Tm": "tau_m",
            "Tz": "tau_z",
        }
        if conserve_text != "Tz":
            alias_pairs.update(
                {
                    "Tx": "tau_x",
                    "Ty": "tau_y",
                }
            )
        for alias, source in alias_pairs.items():
            ops[alias] = ops[source]

        for spin_name in ("Sp", "Sm", "Sz", "Sx", "Sy"):
            if spin_name not in ops:
                continue
            orbital_aliases = ("Tz", "Tp", "Tm") if conserve_text == "Tz" else ("Tx", "Ty", "Tz", "Tp", "Tm")
            for orbital_alias in orbital_aliases:
                ops[f"{spin_name}{orbital_alias}"] = ops[spin_name] @ ops[orbital_alias]
            orbital_names = ("tau_z", "tau_p", "tau_m") if conserve_text == "Tz" else ("tau_x", "tau_y", "tau_z", "tau_p", "tau_m")
            for orbital_name in orbital_names:
                ops[f"{spin_name}_{orbital_name}"] = ops[spin_name] @ ops[orbital_name]

        self.conserve = conserve_text
        super().__init__(leg, self.state_labels, sort_charge=sort_charge, **ops)
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, dtype=int)

    def __repr__(self) -> str:
        return f"YaoLeeSite(conserve='{self.conserve}')"


class YaoLeeModel(CouplingModel):
    """Minimal TeNPy CouplingModel with dense or total-Tz-conserving Yao-Lee sites."""

    def __init__(
        self,
        geometry: GeometryData,
        alpha: float,
        beta: float,
        coupling_j: float = 1.0,
        *,
        bc_MPS: str = "finite",
        sort_charge: bool = False,
        infinite_x: bool = False,
        external_field_terms: list[tuple[float, str]] | None = None,
        symmetry_reductions: dict[str, Any] | None = None,
    ) -> None:
        field_terms = [
            (float(coefficient), str(op_name))
            for coefficient, op_name in list(external_field_terms or [])
            if abs(float(coefficient)) > 1e-14
        ]
        requested_conserve = _tenpy_conserve_from_symmetry_reductions(symmetry_reductions)
        originally_requested_conserve = requested_conserve
        requested_conserve, symmetry_validation_warnings = _validated_tenpy_yao_lee_conserve(
            requested_conserve,
            geometry,
            alpha,
            beta,
            coupling_j,
            infinite_x=bool(infinite_x),
            external_field_terms=field_terms,
        )
        if originally_requested_conserve is not None and requested_conserve is None and not bool(
            (symmetry_reductions or {}).get("allow_dense_fallback", True)
        ):
            raise NotImplementedError(
                "TeNPy could not build the requested Yao-Lee symmetry sector and dense fallback is disabled: "
                + "; ".join(symmetry_validation_warnings or ["unknown symmetry validation failure"])
            )
        site = YaoLeeSite(conserve=requested_conserve, sort_charge=sort_charge)
        self.geometry = geometry
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.coupling_j = float(coupling_j)
        self.infinite_x = bool(infinite_x)
        chain_bc = "periodic" if str(bc_MPS) in ("infinite", "segment") else "open"
        lat = Chain(int(geometry.number_of_sites), site, bc=chain_bc, bc_MPS=bc_MPS)
        super().__init__(lat)

        for bond in geometry.bond_list:
            i = int(bond.i)
            j = int(bond.j)
            self._add_yao_lee_bond(i, j, str(bond.gamma), category=f"YL_{bond.gamma}")

        if self.infinite_x:
            for i, j, gamma in infinite_x_boundary_bonds(geometry):
                self._add_yao_lee_bond(i, j, gamma, category=f"YL_{gamma}_xwrap")

        self.external_field_terms = field_terms
        self.spin_u1_conserved = bool(requested_conserve == "Sz")
        self.tau_z_u1_conserved = bool(requested_conserve == "Tz")
        self.symmetry_mode = _tenpy_symmetry_mode_from_conserve(requested_conserve)
        self.symmetry_validation_warnings = list(symmetry_validation_warnings)
        self.symmetry_reductions = dict(symmetry_reductions or {})
        self.target_tz2 = int(self.symmetry_reductions.get("target_tz2", 0))
        self.target_sz2 = int(self.symmetry_reductions.get("target_sz2", 0))
        if self.tau_z_u1_conserved:
            adjusted_tz2 = _nearest_reachable_spin_half_total_m2(
                int(geometry.number_of_sites),
                self.target_tz2,
            )
            if adjusted_tz2 != self.target_tz2:
                self.symmetry_validation_warnings.append(
                    f"Requested total 2*Tz={self.target_tz2} is unreachable for "
                    f"{int(geometry.number_of_sites)} orbital-1/2 sites; using 2*Tz={adjusted_tz2}."
                )
                self.target_tz2 = int(adjusted_tz2)
                self.symmetry_reductions["target_tz2_effective"] = int(adjusted_tz2)
        for site_index in range(int(self.lat.N_sites)):
            for coefficient, op_name in self.external_field_terms:
                self.add_onsite_term(coefficient, site_index, op_name, category=f"field_{op_name}")

        self.H_MPO = self.calc_H_MPO()

    def _add_yao_lee_bond(self, i: int, j: int, gamma: str, category: str) -> None:
        for coefficient, op_i, op_j in yao_lee_u1_two_site_terms_for_bond(
            str(gamma),
            alpha=self.alpha,
            beta=self.beta,
            coupling_j=self.coupling_j,
        ):
            if int(i) <= int(j):
                self.add_coupling_term(coefficient, int(i), int(j), op_i, op_j, category=category)
            else:
                self.add_coupling_term(coefficient, int(j), int(i), op_j, op_i, category=category)


def _geometry_lookup(geometry: GeometryData) -> dict[tuple[int, int, int], int]:
    lookup: dict[tuple[int, int, int], int] = {}
    for site, ((x_cell, y_cell), sublattice) in enumerate(
        zip(geometry.cell_indices, geometry.sublattice_indices)
    ):
        lookup[(int(x_cell), int(y_cell), int(sublattice))] = int(site)
    return lookup


def infinite_x_boundary_bonds(geometry: GeometryData) -> list[tuple[int, int, str]]:
    """Return bonds from the last x-cell to the next infinite unit cell.

    TeNPy represents couplings into the next infinite MPS unit cell with
    ``j >= N_sites``.  The finite cylinder geometry is open in x, so these
    extra bonds complete the iDMRG-x unit-cell Hamiltonian without changing
    the finite-DMRG path.
    """
    if int(geometry.number_of_sites) <= 0 or len(geometry.cell_indices) == 0:
        return []
    if bool(getattr(geometry, "circumference_x", False)):
        return []
    lookup = _geometry_lookup(geometry)
    x_values = sorted({int(cell[0]) for cell in geometry.cell_indices})
    y_values = sorted({int(cell[1]) for cell in geometry.cell_indices})
    if len(x_values) == 0 or len(y_values) == 0:
        return []
    x_last = int(max(x_values))
    n_sites = int(geometry.number_of_sites)
    has_honeycomb_sublattice = any(int(sub) == 1 for sub in geometry.sublattice_indices)
    boundary: list[tuple[int, int, str]] = []

    for y_cell in y_values:
        y = int(y_cell)
        if has_honeycomb_sublattice:
            left = lookup.get((x_last, y, 0))
            right = lookup.get((0, y, 1))
            if left is not None and right is not None:
                boundary.append((left, right + n_sites, "x"))
        else:
            left = lookup.get((x_last, y, 0))
            right_x = lookup.get((0, y, 0))
            if left is not None and right_x is not None:
                boundary.append((left, right_x + n_sites, "x"))
            y_minus = y - 1
            if y_minus < int(min(y_values)):
                y_minus = int(max(y_values))
            right_z = lookup.get((0, y_minus, 0))
            has_z_bonds = any(str(bond.gamma) == "z" for bond in geometry.bond_list)
            if has_z_bonds and left is not None and right_z is not None:
                boundary.append((left, right_z + n_sites, "z"))

    existing = {
        (int(bond.i), int(bond.j), str(bond.gamma))
        for bond in geometry.bond_list
    }
    return [
        (i, j, gamma)
        for i, j, gamma in boundary
        if (int(i), int(j), str(gamma)) not in existing
    ]


def sz_zero_product_state_labels(length: int, orbital_label: str = "Odown") -> List[str]:
    """Return an alternating product state with total ``2*Sz = 0``."""
    n_sites = int(length)
    if n_sites % 2 != 0:
        raise ValueError("Total Sz=0 product initialization requires an even number of sites.")
    orbital = "Oup" if str(orbital_label).lower() in ("up", "oup", "orbital_up") else "Odown"
    return [
        f"{'Sdown' if site % 2 == 0 else 'Sup'}_{orbital}"
        for site in range(n_sites)
    ]


def _spin_labels_for_target_sz2(length: int, target_sz2: int) -> List[str]:
    n_sites = int(length)
    if int(target_sz2) == 0 and n_sites % 2 == 0:
        return ["Sdown" if site % 2 == 0 else "Sup" for site in range(n_sites)]
    numerator = n_sites + int(target_sz2)
    if numerator % 2 != 0:
        raise ValueError(f"Total 2*Sz={int(target_sz2)} is unreachable for {n_sites} spin-1/2 sites.")
    n_up = numerator // 2
    if n_up < 0 or n_up > n_sites:
        raise ValueError(f"Total 2*Sz={int(target_sz2)} is unreachable for {n_sites} spin-1/2 sites.")
    return ["Sup" if site < n_up else "Sdown" for site in range(n_sites)]


def _spin_half_total_m2_is_reachable(length: int, target_m2: int) -> bool:
    n_sites = int(length)
    numerator = n_sites + int(target_m2)
    if numerator % 2 != 0:
        return False
    n_up = numerator // 2
    return 0 <= n_up <= n_sites


def _nearest_reachable_spin_half_total_m2(length: int, target_m2: int) -> int:
    n_sites = int(length)
    target = int(target_m2)
    if _spin_half_total_m2_is_reachable(n_sites, target):
        return target
    candidates = list(range(-n_sites, n_sites + 1, 2))
    if not candidates:
        raise ValueError("No reachable spin-1/2 total charge sectors exist for an empty unit cell.")
    return min(candidates, key=lambda value: (abs(value - target), abs(value), value))


def _orbital_labels_for_target_tz2(length: int, target_tz2: int) -> List[str]:
    n_sites = int(length)
    numerator = n_sites + int(target_tz2)
    if numerator % 2 != 0:
        raise ValueError(f"Total 2*Tz={int(target_tz2)} is unreachable for {n_sites} orbital-1/2 sites.")
    n_up = numerator // 2
    if n_up < 0 or n_up > n_sites:
        raise ValueError(f"Total 2*Tz={int(target_tz2)} is unreachable for {n_sites} orbital-1/2 sites.")
    labels: List[str] = []
    for site in range(n_sites):
        labels.append("Oup" if site < n_up else "Odown")
    return labels


def charge_target_product_state_labels(model: YaoLeeModel) -> List[str]:
    """Return a product state in the active TeNPy U(1) charge sector."""
    n_sites = int(model.lat.N_sites)
    if str(getattr(model, "symmetry_mode", "none")) == "u1_tz":
        orbital_labels = _orbital_labels_for_target_tz2(n_sites, int(getattr(model, "target_tz2", 0)))
        return [
            f"{'Sdown' if site % 2 == 0 else 'Sup'}_{orbital_labels[site]}"
            for site in range(n_sites)
        ]
    if str(getattr(model, "symmetry_mode", "none")) == "u1_sz":
        spin_labels = _spin_labels_for_target_sz2(n_sites, int(getattr(model, "target_sz2", 0)))
        return [f"{spin_labels[site]}_Odown" for site in range(n_sites)]
    return sz_zero_product_state_labels(n_sites)


def _tenpy_u1_target_sector_info(model: YaoLeeModel) -> dict[str, Any] | None:
    mode = str(getattr(model, "symmetry_mode", "none"))
    if mode == "u1_tz":
        target_tz2 = int(getattr(model, "target_tz2", 0))
        return {"mode": "u1_tz", "total_Tz_times_2": target_tz2, "target_charge": target_tz2}
    if mode == "u1_sz":
        target_sz2 = int(getattr(model, "target_sz2", 0))
        return {"mode": "u1_sz", "total_Sz_times_2": target_sz2, "target_charge": target_sz2}
    return None


def initialize_sz_zero_mps(model: YaoLeeModel, orbital_label: str = "Odown") -> MPS:
    """Initialize a TeNPy MPS in the active charge sector."""
    del orbital_label
    product_state = charge_target_product_state_labels(model)
    return MPS.from_product_state(
        model.lat.mps_sites(),
        product_state,
        bc=model.lat.bc_MPS,
        dtype=np.complex128,
        unit_cell_width=model.lat.mps_unit_cell_width,
    )


def _copy_initial_state_for_model(model: YaoLeeModel, initial_state: MPS | None) -> MPS:
    """Return an MPS compatible with ``model``, using an optimized state if supplied."""
    if initial_state is None:
        return initialize_sz_zero_mps(model)
    if int(getattr(initial_state, "L", -1)) != int(model.lat.N_sites):
        raise ValueError(
            "Adiabatic initial_state has incompatible unit-cell length: "
            f"{getattr(initial_state, 'L', None)} != {model.lat.N_sites}."
        )
    if str(getattr(initial_state, "bc", "")) != str(model.lat.bc_MPS):
        raise ValueError(
            "Adiabatic initial_state has incompatible MPS boundary condition: "
            f"{getattr(initial_state, 'bc', None)} != {model.lat.bc_MPS}."
        )
    return initial_state.copy()


def _center_site_indices(length: int) -> list[int]:
    """Return the central site index or central pair for finite-size diagnostics."""
    n_sites = int(length)
    if n_sites <= 0:
        return []
    if n_sites % 2 == 1:
        return [n_sites // 2]
    return [n_sites // 2 - 1, n_sites // 2]


def _center_bond_index(psi: MPS, *, infinite: bool) -> int | None:
    """Choose the central entanglement cut for a finite MPS or unit cell."""
    length = int(getattr(psi, "L", 0))
    if length <= 0:
        return None
    if infinite:
        return length // 2
    if length <= 1:
        return None
    return max(0, (length - 2) // 2)


def _safe_expectation_values(psi: MPS, operator_name: str) -> np.ndarray:
    values = np.asarray(psi.expectation_value(operator_name), dtype=np.complex128)
    return np.ravel(values)


def compute_plaquette_flux(
    mps: MPS,
    geometry: GeometryData,
    plaquette_center_idx: int | None = None,
) -> dict[str, Any]:
    """Evaluate the normalized six-orbital honeycomb plaquette flux ``W_p``.

    The local TeNPy orbital operators are ``tau_a = sigma_a / 2``.  The returned
    ``W_p`` value includes the normalization factor ``2**6`` so the conserved
    flux eigenvalues are near ``+/-1``.
    """
    plaquettes = honeycomb_plaquette_flux_operators(geometry)
    if len(plaquettes) == 0:
        raise ValueError("No honeycomb length-six plaquette was found in this geometry.")
    selected = select_honeycomb_plaquette_flux_operator(geometry, plaquette_center_idx)
    selected_index = int(selected["plaquette_index"])
    flux_map: dict[int, float] = {}
    details: dict[int, dict[str, Any]] = {}
    expectation_value_term = getattr(mps, "expectation_value_term", None)
    expectation_value_multi_sites = getattr(mps, "expectation_value_multi_sites", None)
    for plaquette in plaquettes:
        term = sorted(
            [
                (str(operator_name), int(site))
                for site, operator_name in zip(plaquette["sites"], plaquette["tenpy_operator_names"])
            ],
            key=lambda item: item[1],
        )
        raw_value: complex
        if callable(expectation_value_term):
            raw_value = complex(expectation_value_term(term))
        else:
            sorted_sites = sorted(int(site) for site in plaquette["sites"])
            if sorted_sites != list(range(sorted_sites[0], sorted_sites[0] + len(sorted_sites))):
                raise RuntimeError(
                    "TeNPy MPS does not expose expectation_value_term, and a selected "
                    "plaquette is not contiguous in MPS order."
                )
            operator_by_site = {
                int(site): str(operator_name)
                for site, operator_name in zip(plaquette["sites"], plaquette["tenpy_operator_names"])
            }
            if not callable(expectation_value_multi_sites):
                raise RuntimeError("TeNPy MPS does not expose a multi-site expectation evaluator.")
            raw_value = complex(
                expectation_value_multi_sites(
                    [operator_by_site[site] for site in sorted_sites],
                    sorted_sites[0],
                )
            )
        normalized_value = float(np.real(raw_value) * float(plaquette["normalization"]))
        plaquette_index = int(plaquette["plaquette_index"])
        flux_map[plaquette_index] = normalized_value
        details[plaquette_index] = {
            "plaquette_index": plaquette_index,
            "sites": [int(site) for site in plaquette["sites"]],
            "axes": [str(axis) for axis in plaquette["axes"]],
            "operators": [str(op) for op in plaquette["tenpy_operator_names"]],
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


def compute_all_plaquette_fluxes(
    mps: MPS,
    geometry: GeometryData,
) -> dict[int, float]:
    """Return normalized ``W_p`` on every valid elementary honeycomb plaquette."""
    flux_payload = compute_plaquette_flux(mps, geometry, plaquette_center_idx=None)
    flux_map = flux_payload.get("all_plaquette_fluxes", flux_payload.get("plaquette_flux_map", {}))
    if not isinstance(flux_map, dict):
        return {}
    return {int(index): float(value) for index, value in flux_map.items()}


def extract_phase_boundary_observables(
    psi: MPS,
    *,
    engine: str = "finite_dmrg",
    geometry: GeometryData | None = None,
) -> dict[str, Any]:
    """Extract observables useful for detecting phase-boundary changes.

    The local order parameters are sampled at the central finite sites, or at
    the central sites of the iDMRG unit cell when ``engine='idmrg'``.
    """
    engine_name = str(engine)
    infinite = str(getattr(psi, "bc", "")) == "infinite" or engine_name.lower() == "idmrg"
    length = int(getattr(psi, "L", 0))
    center_sites = _center_site_indices(length)
    center_bond = _center_bond_index(psi, infinite=infinite)

    entanglement_entropy = None
    entanglement_warning = None
    if center_bond is not None:
        try:
            entropy_values = np.asarray(psi.entanglement_entropy(n=1, bonds=[center_bond]), dtype=float)
            entanglement_entropy = float(np.ravel(entropy_values)[0])
        except Exception as exc:
            try:
                singular_values = np.asarray(psi.get_SL(center_bond), dtype=float)
                entanglement_entropy = float(_entropy_dict_from_singular_values(singular_values, (1,))["S1"])
                entanglement_warning = f"Used Schmidt values because entanglement_entropy failed: {exc}"
            except Exception as fallback_exc:
                entanglement_warning = (
                    "Failed to extract center-bond entanglement entropy: "
                    f"{exc}; fallback failed: {fallback_exc}"
                )

    local_order: dict[str, Any] = {
        "center_sites": [int(site) for site in center_sites],
    }
    for output_key, operator_name in (("Sz", "Sz"), ("tau_z", "tau_z")):
        try:
            values = _safe_expectation_values(psi, operator_name)
            sampled = [
                float(np.real(values[site]))
                for site in center_sites
                if 0 <= int(site) < int(values.size)
            ]
            local_order[f"{output_key}_center_values"] = sampled
            local_order[f"{output_key}_center_mean"] = (
                float(np.mean(sampled)) if len(sampled) > 0 else None
            )
        except Exception as exc:
            local_order[f"{output_key}_warning"] = str(exc)

    correlation_length = None
    correlation_length_warning = None
    if infinite:
        try:
            correlation_length = float(psi.correlation_length())
        except Exception as exc:
            correlation_length_warning = str(exc)

    plaquette_flux = None
    plaquette_flux_warning = None
    if geometry is not None:
        try:
            plaquette_flux = compute_plaquette_flux(psi, geometry, None)
        except Exception as exc:
            plaquette_flux_warning = str(exc)

    observables: dict[str, Any] = {
        "engine": "idmrg" if infinite else "finite_dmrg",
        "center_bond": None if center_bond is None else int(center_bond),
        "entanglement_entropy": entanglement_entropy,
        "S_E": entanglement_entropy,
        "local_order_parameters": local_order,
    }
    if plaquette_flux is not None:
        observables["plaquette_flux"] = plaquette_flux
        observables["all_plaquette_fluxes"] = plaquette_flux.get(
            "all_plaquette_fluxes",
            plaquette_flux.get("plaquette_flux_map", {}),
        )
        observables["W_p"] = plaquette_flux.get("W_p")
    if plaquette_flux_warning is not None:
        observables["plaquette_flux_warning"] = plaquette_flux_warning
    if entanglement_warning is not None:
        observables["entanglement_entropy_warning"] = entanglement_warning
    if infinite:
        observables["correlation_length_xi"] = correlation_length
        observables["xi"] = correlation_length
        if correlation_length_warning is not None:
            observables["correlation_length_warning"] = correlation_length_warning
    return observables


def tenpy_backend_template(
    geometry: GeometryData,
    alpha: float,
    beta: float,
    coupling_j: float = 1.0,
    dmrg_options: dict[str, Any] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    symmetry_reductions: dict[str, Any] | None = None,
) -> tuple[YaoLeeModel, MPS, dict[str, Any]]:
    """Small runnable TeNPy template returning ``(model, psi, dmrg_options)``."""
    model = YaoLeeModel(
        geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        external_field_terms=external_field_terms,
        symmetry_reductions=symmetry_reductions,
    )
    psi = initialize_sz_zero_mps(model)
    options = _stable_tenpy_dmrg_params(
        max_bond_dimension=128,
        requested_sweeps=TENPY_STABLE_MIN_SWEEPS,
        truncation_cutoff=TENPY_STABLE_TRUNC_CUT,
    )
    if dmrg_options:
        options.update(dmrg_options)
    return model, psi, options


def run_cylindrical_dmrg(
    geometry: GeometryData,
    alpha: float,
    beta: float,
    coupling_j: float,
    max_bond_dimension: int,
    max_sweeps: int,
    truncation_cutoff: float = 1.0e-10,
    svd_min: float | None = None,
    random_seed: int = 0,
    product_state_style: str = "alternating",
    initial_state: MPS | None = None,
    compute_phase_observables: bool = True,
    external_field_terms: list[tuple[float, str]] | None = None,
    symmetry_reductions: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> tuple[MPS, Any, dict[str, Any]]:
    """Compatibility hook for ``ylmodel_main.py --backend tenpy``."""
    del random_seed, product_state_style
    stage_start = _start_stage("TeNPy finite DMRG", show_progress)
    _configure_tenpy_progress_logging(show_progress)
    if show_progress:
        effective_options = _stable_tenpy_dmrg_params(
            max_bond_dimension=max_bond_dimension,
            requested_sweeps=max_sweeps,
            truncation_cutoff=truncation_cutoff,
            svd_min=svd_min,
        )
        print(
            "[tenpy-dmrg] setup: "
            f"N={int(geometry.number_of_sites)}, alpha={float(alpha):.8g}, beta={float(beta):.8g}, "
            f"chi_max={int(max_bond_dimension)}, sweeps={int(effective_options['max_sweeps'])}, "
            f"svd_min={float(effective_options['trunc_params']['svd_min']):.3g}, "
            f"mixer={effective_options['mixer']}"
        )
    else:
        effective_options = _stable_tenpy_dmrg_params(
            max_bond_dimension=max_bond_dimension,
            requested_sweeps=max_sweeps,
            truncation_cutoff=truncation_cutoff,
            svd_min=svd_min,
        )
    model = YaoLeeModel(
        geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        external_field_terms=external_field_terms,
        symmetry_reductions=symmetry_reductions,
    )
    psi = _copy_initial_state_for_model(model, initial_state)
    options = effective_options
    try:
        info = _run_dmrg_with_sweep_progress(
            psi,
            model,
            options,
            show_progress=show_progress,
            desc="tenpy dmrg sweeps",
            expected_sweeps=max(1, int(options["max_sweeps"])),
        )
    except Exception:
        _end_stage("TeNPy finite DMRG", stage_start, show_progress)
        raise
    phase_observables = None
    phase_observable_warning = None
    if compute_phase_observables:
        try:
            phase_observables = extract_phase_boundary_observables(psi, engine="finite_dmrg", geometry=geometry)
        except Exception as exc:
            phase_observable_warning = str(exc)
    canonicalization_warning = None
    norm_error_after_canonicalization = None
    try:
        psi.canonical_form()
        norm_error = psi.norm_test()
        norm_error_after_canonicalization = float(np.linalg.norm(norm_error))
    except Exception as exc:
        canonicalization_warning = str(exc)
    energy = float(info.get("E", info.get("energy", np.nan)))
    dmrg_info = {
        "E": energy,
        "converged": bool(info.get("shelve", False)) if "converged" not in info else bool(info["converged"]),
        "symmetry_mode": str(getattr(model, "symmetry_mode", "none")),
        "symmetry_enabled": bool(
            getattr(model, "tau_z_u1_conserved", False) or getattr(model, "spin_u1_conserved", False)
        ),
        "symmetry_backend_status": {
            "backend": "tenpy",
            "real_u1_tz": bool(getattr(model, "tau_z_u1_conserved", False)),
            "real_u1_sz": bool(getattr(model, "spin_u1_conserved", False)),
            "dense_fallback_used": bool(
                symmetry_reductions
                and (
                    bool(symmetry_reductions.get("use_tau_z_block", False))
                    or bool(symmetry_reductions.get("use_sz_block", False))
                )
                and str(getattr(model, "symmetry_mode", "none")) == "none"
            ),
            "z2_block": False,
        },
        "symmetry_validation_warnings": list(getattr(model, "symmetry_validation_warnings", [])),
        "u1_target_sector": _tenpy_u1_target_sector_info(model),
        "initial_state_style": (
            "adiabatic_previous_mps" if initial_state is not None else "alternating_sz_zero_product"
        ),
        "used_adiabatic_initial_state": bool(initial_state is not None),
        "mixer": str(options.get("mixer")),
        "mixer_params": dict(options.get("mixer_params", {})),
        "diag_method": "lanczos",
        "chi_list": {int(key): int(value) for key, value in options.get("chi_list", {}).items()},
        "trunc_params": dict(options.get("trunc_params", {})),
        "max_sweeps": int(options.get("max_sweeps", max_sweeps)),
        "norm_error_after_canonicalization": norm_error_after_canonicalization,
        "external_field_terms": [
            (float(coefficient), str(op_name))
            for coefficient, op_name in list(external_field_terms or [])
        ],
        "symmetry_reductions": dict(symmetry_reductions or {}),
    }
    if phase_observables is not None:
        dmrg_info["phase_observables"] = phase_observables
        dmrg_info["all_plaquette_fluxes"] = phase_observables.get("all_plaquette_fluxes", {})
    if phase_observable_warning is not None:
        dmrg_info["phase_observables_warning"] = phase_observable_warning
    if canonicalization_warning is not None:
        dmrg_info["canonicalization_warning"] = canonicalization_warning
    if "post_run_canonical_form_warning" in info:
        dmrg_info["post_run_canonical_form_warning"] = str(info["post_run_canonical_form_warning"])
    _end_stage("TeNPy finite DMRG", stage_start, show_progress)
    return psi, model.H_MPO, dmrg_info


def run_alpha_scan_with_adiabatic_state_passing(
    geometry: GeometryData,
    alpha_values: list[float],
    beta: float,
    coupling_j: float,
    max_bond_dimension: int,
    max_sweeps: int,
    truncation_cutoff: float = 1.0e-10,
    svd_min: float | None = None,
    initial_state: MPS | None = None,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    symmetry_reductions: dict[str, Any] | None = None,
    show_progress: bool = True,
    progress_bar: Any | None = None,
) -> tuple[list[dict[str, Any]], MPS | None]:
    """Scan alpha at fixed beta, passing each optimized MPS to the next point.

    This is the adiabatic continuation pattern:
    ``psi(alpha_i)`` is optimized, then used as ``initial_state`` for
    ``alpha_{i+1}``.  The Hamiltonian changes while the MPS basis and U(1)
    charge sector remain compatible because the geometry and ``YaoLeeSite`` are
    fixed throughout the row.
    """
    rows: list[dict[str, Any]] = []
    previous_psi = initial_state
    thresholds = classifier_thresholds or DEFAULT_PHASE_CLASSIFIER_THRESHOLDS
    for alpha_index, alpha in enumerate(alpha_values):
        used_adiabatic_state = previous_psi is not None
        if show_progress:
            print(
                "[phase-scan:dmrg] point started: "
                f"beta={float(beta):.8g}, alpha={float(alpha):.8g}, "
                f"adiabatic_initial={bool(used_adiabatic_state)}"
            )
        psi, _mpo, info = run_cylindrical_dmrg(
            geometry=geometry,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=float(coupling_j),
            max_bond_dimension=int(max_bond_dimension),
            max_sweeps=int(max_sweeps),
            truncation_cutoff=float(truncation_cutoff),
            svd_min=svd_min,
            initial_state=previous_psi,
            compute_phase_observables=True,
            external_field_terms=external_field_terms,
            symmetry_reductions=symmetry_reductions,
            show_progress=show_progress,
        )
        structure_rows: list[dict[str, Any]] = []
        bond_rows: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {}
        phase_label = "Weak/undetermined"
        try:
            correlations = collect_correlation_matrices_from_dmrg(psi, show_progress=show_progress)
            scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
            structure_rows = all_high_symmetry_structure_factors(
                scalar_correlations,
                geometry,
                show_progress=show_progress,
            )
            bond_rows = all_bond_energies(
                geometry,
                correlations,
                float(alpha),
                float(beta),
                float(coupling_j),
                show_progress=show_progress,
            )
            diagnostics = _phase_observable_diagnostics(
                structure_rows,
                bond_rows,
                geometry.number_of_sites,
                plaquette_flux=(info.get("phase_observables") or {}).get("plaquette_flux"),
            )
            phase_label = _classify_phase_from_diagnostics(
                diagnostics,
                float(alpha),
                float(beta),
                "tenpy_dmrg",
                thresholds,
            )
        except Exception as exc:
            diagnostics = {"warning": f"Failed to compute strict phase diagnostics: {exc}"}
        rows.append(
            {
                "status": "completed",
                "alpha_index": int(alpha_index),
                "alpha": float(alpha),
                "beta": float(beta),
                "energy": float(info.get("E", np.nan)),
                "energy_per_site": float(info.get("E", np.nan)) / float(max(1, geometry.number_of_sites)),
                "used_adiabatic_initial_state": bool(used_adiabatic_state),
                "observables": info.get("phase_observables", {}),
                "all_plaquette_fluxes": (info.get("phase_observables") or {}).get("all_plaquette_fluxes", {}),
                "dmrg_options": {
                    "symmetry_reductions": dict(symmetry_reductions or {}),
                    "symmetry_mode": str(info.get("symmetry_mode", "none")),
                    "symmetry_backend_status": dict(info.get("symmetry_backend_status", {})),
                    "trunc_params": dict(info.get("trunc_params", {})),
                    "max_sweeps": int(info.get("max_sweeps", max_sweeps)),
                },
                "phase_label": phase_label,
                "diagnostics": diagnostics,
                "structure_factors": structure_rows,
                "bond_energies": bond_rows,
            }
        )
        # Pass the optimized state from this alpha point to the next alpha point.
        previous_psi = psi
        if progress_bar is not None:
            progress_bar.update(1)
    return rows, previous_psi


def run_alpha_beta_dmrg_observable_scan(
    geometry: GeometryData,
    alpha_values: list[float],
    beta_values: list[float],
    coupling_j: float,
    max_bond_dimension: int,
    max_sweeps: int,
    truncation_cutoff: float = 1.0e-10,
    svd_min: float | None = None,
    carry_state_between_betas: bool = False,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    symmetry_reductions: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run a TeNPy DMRG observable scan over beta rows and alpha columns."""
    stage_start = _start_stage("TeNPy finite-DMRG phase scan", show_progress)
    all_rows: list[dict[str, Any]] = []
    beta_initial_state: MPS | None = None
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(alpha_values) * len(beta_values),
        desc="tenpy dmrg scan",
        unit="point",
        leave=False,
    )
    for beta_index, beta in enumerate(beta_values):
        rows, final_state = run_alpha_scan_with_adiabatic_state_passing(
            geometry=geometry,
            alpha_values=[float(value) for value in alpha_values],
            beta=float(beta),
            coupling_j=float(coupling_j),
            max_bond_dimension=int(max_bond_dimension),
            max_sweeps=int(max_sweeps),
            truncation_cutoff=float(truncation_cutoff),
            svd_min=svd_min,
            initial_state=beta_initial_state,
            classifier_thresholds=classifier_thresholds,
            external_field_terms=external_field_terms,
            symmetry_reductions=symmetry_reductions,
            show_progress=False,
            progress_bar=progress_bar,
        )
        for row in rows:
            row["beta_index"] = int(beta_index)
            all_rows.append(row)
        beta_initial_state = final_state if bool(carry_state_between_betas) else None
    if progress_bar is not None:
        progress_bar.close()
    _end_stage("TeNPy finite-DMRG phase scan", stage_start, show_progress)
    return {
        "status": "completed",
        "backend": "tenpy",
        "scan_type": "finite_dmrg_observable_scan",
        "adiabatic_state_passing": {
            "direction": "alpha",
            "carry_state_between_betas": bool(carry_state_between_betas),
        },
        "alpha_values": [float(value) for value in alpha_values],
        "beta_values": [float(value) for value in beta_values],
        "rows": all_rows,
    }


def run_alpha_scan_idmrg_with_adiabatic_state_passing(
    geometry: GeometryData,
    alpha_values: list[float],
    beta: float,
    coupling_j: float,
    max_bond_dimension: int,
    max_iterations: int,
    truncation_cutoff: float = 1.0e-10,
    svd_min: float | None = None,
    initial_state: MPS | None = None,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    symmetry_reductions: dict[str, Any] | None = None,
    show_progress: bool = True,
    progress_bar: Any | None = None,
) -> tuple[list[dict[str, Any]], MPS | None]:
    """Scan alpha with iDMRG, passing each optimized infinite MPS forward."""
    rows: list[dict[str, Any]] = []
    previous_psi = initial_state
    thresholds = classifier_thresholds or DEFAULT_PHASE_CLASSIFIER_THRESHOLDS
    for alpha_index, alpha in enumerate(alpha_values):
        used_adiabatic_state = previous_psi is not None
        point_stage = _start_stage(
            f"TeNPy iDMRG point beta={float(beta):.8g}, alpha={float(alpha):.8g}",
            show_progress,
        )
        _configure_tenpy_progress_logging(show_progress)
        model = YaoLeeModel(
            geometry,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=float(coupling_j),
            bc_MPS="infinite",
            infinite_x=True,
            external_field_terms=external_field_terms,
            symmetry_reductions=symmetry_reductions,
        )
        psi = _copy_initial_state_for_model(model, previous_psi)
        options = _stable_tenpy_dmrg_params(
            max_bond_dimension=max_bond_dimension,
            requested_sweeps=max_iterations,
            truncation_cutoff=truncation_cutoff,
            svd_min=svd_min,
        )
        try:
            info = _run_dmrg_with_sweep_progress(
                psi,
                model,
                options,
                show_progress=show_progress,
                desc="tenpy idmrg scan sweeps",
                expected_sweeps=max(1, int(options["max_sweeps"])),
            )
        except Exception:
            _end_stage("TeNPy iDMRG point", point_stage, show_progress)
            raise
        energy_density = float(info.get("E", info.get("energy", np.nan)))
        observables = extract_phase_boundary_observables(psi, engine="idmrg", geometry=geometry)
        structure_rows: list[dict[str, Any]] = []
        bond_rows: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {}
        phase_label = "Weak/undetermined"
        try:
            correlations = collect_correlation_matrices_from_dmrg(psi, show_progress=show_progress)
            scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
            structure_rows = all_high_symmetry_structure_factors(
                scalar_correlations,
                geometry,
                show_progress=show_progress,
            )
            bond_rows = all_bond_energies(
                geometry,
                correlations,
                float(alpha),
                float(beta),
                float(coupling_j),
                show_progress=show_progress,
            )
            diagnostics = _phase_observable_diagnostics(
                structure_rows,
                bond_rows,
                geometry.number_of_sites,
                plaquette_flux=observables.get("plaquette_flux"),
            )
            phase_label = _classify_phase_from_diagnostics(
                diagnostics,
                float(alpha),
                float(beta),
                "tenpy_idmrg",
                thresholds,
            )
        except Exception as exc:
            diagnostics = {"warning": f"Failed to compute strict phase diagnostics: {exc}"}
        rows.append(
            {
                "status": "completed",
                "alpha_index": int(alpha_index),
                "alpha": float(alpha),
                "beta": float(beta),
                "energy_per_site": energy_density,
                "energy_per_unit_cell": energy_density * float(geometry.number_of_sites),
                "used_adiabatic_initial_state": bool(used_adiabatic_state),
                "observables": observables,
                "all_plaquette_fluxes": observables.get("all_plaquette_fluxes", {}),
                "dmrg_options": {
                    "symmetry_reductions": dict(symmetry_reductions or {}),
                    "symmetry_mode": str(getattr(model, "symmetry_mode", "none")),
                    "symmetry_backend_status": {
                        "backend": "tenpy",
                        "real_u1_tz": bool(getattr(model, "tau_z_u1_conserved", False)),
                        "dense_fallback_used": bool(
                            symmetry_reductions
                            and bool(symmetry_reductions.get("use_tau_z_block", False))
                            and str(getattr(model, "symmetry_mode", "none")) == "none"
                        ),
                        "z2_block": False,
                    },
                    "mixer": str(options.get("mixer")),
                    "mixer_params": dict(options.get("mixer_params", {})),
                    "chi_list": {
                        int(key): int(value)
                        for key, value in options.get("chi_list", {}).items()
                    },
                    "trunc_params": dict(options.get("trunc_params", {})),
                    "max_sweeps": int(options.get("max_sweeps", max_iterations)),
                },
                "post_run_canonical_form_warning": info.get("post_run_canonical_form_warning"),
                "phase_label": phase_label,
                "diagnostics": diagnostics,
                "structure_factors": structure_rows,
                "bond_energies": bond_rows,
            }
        )
        previous_psi = psi
        _end_stage("TeNPy iDMRG point", point_stage, show_progress)
        if progress_bar is not None:
            progress_bar.update(1)
    return rows, previous_psi


def run_alpha_beta_idmrg_observable_scan(
    geometry: GeometryData,
    alpha_values: list[float],
    beta_values: list[float],
    coupling_j: float,
    max_bond_dimension: int,
    max_iterations: int,
    max_unit_cell_sites: int | None = None,
    truncation_cutoff: float = 1.0e-10,
    svd_min: float | None = None,
    carry_state_between_betas: bool = False,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    symmetry_reductions: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run a TeNPy iDMRG observable scan over beta rows and alpha columns."""
    if max_unit_cell_sites is not None and int(geometry.number_of_sites) > int(max_unit_cell_sites):
        return {
            "status": "skipped",
            "backend": "tenpy",
            "scan_type": "idmrg_observable_scan",
            "reason": (
                f"TeNPy iDMRG unit-cell safety cap is N <= {int(max_unit_cell_sites)}, "
                f"but geometry has N={int(geometry.number_of_sites)}."
            ),
            "rows": [],
            "completed_points": 0,
            "failed_points": 0,
            "skipped_points": int(len(alpha_values) * len(beta_values)),
        }
    stage_start = _start_stage("TeNPy iDMRG phase scan", show_progress)
    all_rows: list[dict[str, Any]] = []
    beta_initial_state: MPS | None = None
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(alpha_values) * len(beta_values),
        desc="tenpy idmrg scan",
        unit="point",
        leave=False,
    )
    for beta_index, beta in enumerate(beta_values):
        rows, final_state = run_alpha_scan_idmrg_with_adiabatic_state_passing(
            geometry=geometry,
            alpha_values=[float(value) for value in alpha_values],
            beta=float(beta),
            coupling_j=float(coupling_j),
            max_bond_dimension=int(max_bond_dimension),
            max_iterations=int(max_iterations),
            truncation_cutoff=float(truncation_cutoff),
            svd_min=svd_min,
            initial_state=beta_initial_state,
            classifier_thresholds=classifier_thresholds,
            external_field_terms=external_field_terms,
            symmetry_reductions=symmetry_reductions,
            show_progress=False,
            progress_bar=progress_bar,
        )
        for row in rows:
            row["beta_index"] = int(beta_index)
            all_rows.append(row)
        beta_initial_state = final_state if bool(carry_state_between_betas) else None
    if progress_bar is not None:
        progress_bar.close()
    _end_stage("TeNPy iDMRG phase scan", stage_start, show_progress)
    return {
        "status": "completed",
        "backend": "tenpy",
        "scan_type": "idmrg_observable_scan",
        "adiabatic_state_passing": {
            "direction": "alpha",
            "carry_state_between_betas": bool(carry_state_between_betas),
        },
        "alpha_values": [float(value) for value in alpha_values],
        "beta_values": [float(value) for value in beta_values],
        "rows": all_rows,
        "translation_symmetry": {
            "enabled": True,
            "implemented_as": "infinite repeated MPS unit cell along x",
        },
    }


def _entropy_dict_from_singular_values(values: np.ndarray, orders: tuple[int, ...]) -> dict[str, float]:
    probabilities = np.asarray(values, dtype=float) ** 2
    total = float(np.sum(probabilities))
    if total <= 0.0 or not np.isfinite(total):
        return {f"S{order_n}": float("nan") for order_n in orders}
    probabilities = probabilities / total
    probabilities = probabilities[probabilities > 0.0]
    output: dict[str, float] = {}
    for order_n in orders:
        if int(order_n) == 1:
            output[f"S{order_n}"] = float(-np.sum(probabilities * np.log(probabilities)))
        else:
            power_sum = float(np.sum(probabilities ** int(order_n)))
            output[f"S{order_n}"] = float(np.log(power_sum) / (1.0 - float(order_n)))
    return output


def _summarize_entropy_values(entropies: dict[str, list[float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, values in entropies.items():
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_max"] = float("nan")
            summary[f"{key}_min"] = float("nan")
        else:
            summary[f"{key}_mean"] = float(np.mean(finite))
            summary[f"{key}_max"] = float(np.max(finite))
            summary[f"{key}_min"] = float(np.min(finite))
    return summary


def tenpy_infinite_mps_entropy_profile(
    psi: MPS,
    finite_n_sites: int | None = None,
    orders: tuple[int, ...] = ENTROPY_ORDERS,
) -> dict[str, Any]:
    """Build an iDMRG entropy profile from TeNPy infinite-MPS Schmidt values."""
    if str(getattr(psi, "bc", "")) != "infinite":
        raise RuntimeError("TeNPy iDMRG entropy profile requires an infinite MPS.")
    unit_cell_size = int(getattr(psi, "L", 0))
    if unit_cell_size <= 0:
        raise RuntimeError("Invalid TeNPy infinite-MPS unit-cell size.")

    entropies: dict[str, list[float]] = {f"S{order_n}": [] for order_n in orders}
    for bond in range(unit_cell_size):
        singular_values = np.asarray(psi.get_SL(bond), dtype=float)
        entropy_values = _entropy_dict_from_singular_values(singular_values, orders)
        for key, value in entropy_values.items():
            entropies[key].append(float(value))

    if finite_n_sites is not None and int(finite_n_sites) > 1:
        finite_size = int(finite_n_sites)
        cuts = [float(cut) for cut in range(1, finite_size)]
        mapped = {key: [] for key in entropies}
        for cut in range(1, finite_size):
            bond_index = (cut - 1) % unit_cell_size
            for key, values in entropies.items():
                mapped[key].append(float(values[bond_index]))
        entropies_for_output = mapped
        output_cuts = cuts
        total_span = float(finite_size)
    else:
        entropies_for_output = entropies
        output_cuts = [float(bond + 1) for bond in range(unit_cell_size)]
        total_span = float(unit_cell_size)

    return {
        "method_label": "iDMRG-x",
        "cuts": output_cuts,
        "total_span": total_span,
        "entropies": entropies_for_output,
        "summary": _summarize_entropy_values(entropies_for_output),
        "context": {
            "backend": "tenpy",
            "bc": "infinite",
            "unit_cell_sites": unit_cell_size,
            "finite_n_sites_for_normalized_cuts": finite_n_sites,
        },
    }


def run_cylindrical_idmrg(
    geometry: GeometryData,
    alpha: float,
    beta: float,
    coupling_j: float,
    max_bond_dimension: int,
    max_iterations: int,
    truncation_cutoff: float = 1.0e-10,
    svd_min: float | None = None,
    random_seed: int = 0,
    product_state_style: str = "alternating",
    compute_entanglement: bool = True,
    compute_phase_observables: bool = True,
    initial_state: MPS | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    symmetry_reductions: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run TeNPy infinite-DMRG along x with one finite cylinder as the unit cell."""
    del random_seed, product_state_style
    stage_start = _start_stage("TeNPy iDMRG-x", show_progress)
    _configure_tenpy_progress_logging(show_progress)
    options = _stable_tenpy_dmrg_params(
        max_bond_dimension=max_bond_dimension,
        requested_sweeps=max_iterations,
        truncation_cutoff=truncation_cutoff,
        svd_min=svd_min,
    )
    if show_progress:
        print(
            "[tenpy-idmrg] setup: "
            f"N_unit_cell={int(geometry.number_of_sites)}, alpha={float(alpha):.8g}, beta={float(beta):.8g}, "
            f"chi_max={int(max_bond_dimension)}, iterations={int(options['max_sweeps'])}, "
            f"svd_min={float(options['trunc_params']['svd_min']):.3g}, "
            f"mixer={options['mixer']}"
        )
    model = YaoLeeModel(
        geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        bc_MPS="infinite",
        infinite_x=True,
        external_field_terms=external_field_terms,
        symmetry_reductions=symmetry_reductions,
    )
    psi = _copy_initial_state_for_model(model, initial_state)
    try:
        info = _run_dmrg_with_sweep_progress(
            psi,
            model,
            options,
            show_progress=show_progress,
            desc="tenpy idmrg sweeps",
            expected_sweeps=max(1, int(options["max_sweeps"])),
        )
    except Exception:
        _end_stage("TeNPy iDMRG-x", stage_start, show_progress)
        raise
    energy_density = float(info.get("E", info.get("energy", np.nan)))
    phase_observables = None
    phase_observable_warning = None
    if compute_phase_observables:
        try:
            phase_observables = extract_phase_boundary_observables(psi, engine="idmrg", geometry=geometry)
        except Exception as exc:
            phase_observable_warning = str(exc)
    entanglement_profile = None
    entanglement_warning = None
    entanglement_status = "skipped_disabled"
    if compute_entanglement:
        try:
            entanglement_profile = tenpy_infinite_mps_entropy_profile(
                psi,
                finite_n_sites=int(geometry.number_of_sites),
                orders=ENTROPY_ORDERS,
            )
            entanglement_status = "completed"
        except Exception as exc:
            entanglement_status = "failed"
            entanglement_warning = f"Failed to compute TeNPy iDMRG entanglement profile: {exc}"

    output: dict[str, Any] = {
        "status": "completed",
        "backend": "tenpy",
        "method_note": (
            "TeNPy iDMRG-x repeats the finite cylinder as the infinite MPS unit cell "
            "and adds missing x-boundary bonds into the next unit cell only when "
            "the finite geometry is open in x."
        ),
        "ground_state_energy_per_site": energy_density,
        "energy_per_original_site": energy_density,
        "energy_per_unit_cell": energy_density * float(geometry.number_of_sites),
        "unit_cell_sites": int(geometry.number_of_sites),
        "translation_symmetry": {
            "enabled": True,
            "implemented_as": "infinite repeated MPS unit cell along x",
        },
        "infinite_x_boundary_bonds": [
            {"i": int(i), "j": int(j), "gamma": str(gamma)}
            for i, j, gamma in infinite_x_boundary_bonds(geometry)
        ],
        "info": {
            "E": energy_density,
            "shelve": bool(info.get("shelve", False)),
            "symmetry_mode": str(getattr(model, "symmetry_mode", "none")),
            "symmetry_reductions": dict(symmetry_reductions or {}),
            "symmetry_backend_status": {
                "backend": "tenpy",
                "real_u1_tz": bool(getattr(model, "tau_z_u1_conserved", False)),
                "real_u1_sz": bool(getattr(model, "spin_u1_conserved", False)),
                "dense_fallback_used": bool(
                    symmetry_reductions
                    and (
                        bool(symmetry_reductions.get("use_tau_z_block", False))
                        or bool(symmetry_reductions.get("use_sz_block", False))
                    )
                    and str(getattr(model, "symmetry_mode", "none")) == "none"
                ),
                "z2_block": False,
            },
            "symmetry_validation_warnings": list(getattr(model, "symmetry_validation_warnings", [])),
            "u1_target_sector": _tenpy_u1_target_sector_info(model),
            "max_sweeps": int(options.get("max_sweeps", max_iterations)),
            "max_bond_dimension": int(max_bond_dimension),
            "mixer": str(options.get("mixer")),
            "mixer_params": dict(options.get("mixer_params", {})),
            "chi_list": {int(key): int(value) for key, value in options.get("chi_list", {}).items()},
            "trunc_params": dict(options.get("trunc_params", {})),
            "used_adiabatic_initial_state": bool(initial_state is not None),
            "post_run_canonical_form_warning": info.get("post_run_canonical_form_warning"),
        },
        "entanglement_status": entanglement_status,
    }
    if phase_observables is not None:
        output["phase_observables"] = phase_observables
        output["all_plaquette_fluxes"] = phase_observables.get("all_plaquette_fluxes", {})
    if phase_observable_warning is not None:
        output["phase_observables_warning"] = phase_observable_warning
    if entanglement_profile is not None:
        output["entanglement"] = entanglement_profile
    if entanglement_warning is not None:
        output["entanglement_warning"] = entanglement_warning
    _end_stage("TeNPy iDMRG-x", stage_start, show_progress)
    return output


def _mps_corr(psi: MPS, op_i: str, i: int, op_j: str, j: int) -> complex:
    value = psi.correlation_function(op_i, op_j, sites1=[int(i)], sites2=[int(j)])
    return complex(np.asarray(value).reshape(-1)[0])


def _mps_has_operator(psi: MPS, op_name: str) -> bool:
    try:
        psi.sites[0].get_op(str(op_name))
        return True
    except Exception:
        return False


def collect_correlation_matrices_from_dmrg(psi: MPS, show_progress: bool = False) -> dict[str, np.ndarray]:
    """Collect the correlation channels expected by the shared plot pipeline."""
    n_sites = int(psi.L)
    correlations: dict[str, np.ndarray] = {}
    for key in (
        "Sx_Sx", "Sy_Sy", "Sz_Sz",
        "Tx_Tx", "Ty_Ty", "Tz_Tz",
    ):
        correlations[key] = np.zeros((n_sites, n_sites), dtype=np.complex128)
    for orbital_axis in ("x", "y", "z"):
        for spin_axis in ("x", "y", "z"):
            correlations[f"S{spin_axis}T{orbital_axis}_S{spin_axis}T{orbital_axis}"] = np.zeros(
                (n_sites, n_sites),
                dtype=np.complex128,
            )
    correlations["STx_STx"] = correlations["SxTx_SxTx"]
    correlations["STy_STy"] = correlations["SyTy_SyTy"]
    correlations["STz_STz"] = correlations["SzTz_SzTz"]

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=max(0, n_sites * (n_sites - 1)),
        desc="TeNPy correlations",
        unit="pair",
        leave=False,
    )
    has_tx_ty = _mps_has_operator(psi, "Tx") and _mps_has_operator(psi, "Ty")
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            sp_sm = _mps_corr(psi, "Sp", i, "Sm", j)
            sm_sp = _mps_corr(psi, "Sm", i, "Sp", j)
            transverse_spin = 0.25 * (sp_sm + sm_sp)
            correlations["Sx_Sx"][i, j] = transverse_spin
            correlations["Sy_Sy"][i, j] = transverse_spin
            correlations["Sz_Sz"][i, j] = _mps_corr(psi, "Sz", i, "Sz", j)
            if has_tx_ty:
                axis_ops = (("x", "Tx"), ("y", "Ty"), ("z", "Tz"))
                for axis, op in axis_ops:
                    correlations[f"T{axis}_T{axis}"][i, j] = _mps_corr(psi, op, i, op, j)
                    sp_t = f"SpT{axis}"
                    sm_t = f"SmT{axis}"
                    sz_t = f"SzT{axis}"
                    mixed_transverse = 0.25 * (
                        _mps_corr(psi, sp_t, i, sm_t, j)
                        + _mps_corr(psi, sm_t, i, sp_t, j)
                    )
                    correlations[f"SxT{axis}_SxT{axis}"][i, j] = mixed_transverse
                    correlations[f"SyT{axis}_SyT{axis}"][i, j] = mixed_transverse
                    correlations[f"SzT{axis}_SzT{axis}"][i, j] = _mps_corr(psi, sz_t, i, sz_t, j)
            else:
                orbital_transverse = 0.25 * (
                    _mps_corr(psi, "Tp", i, "Tm", j)
                    + _mps_corr(psi, "Tm", i, "Tp", j)
                )
                correlations["Tx_Tx"][i, j] = orbital_transverse
                correlations["Ty_Ty"][i, j] = orbital_transverse
                correlations["Tz_Tz"][i, j] = _mps_corr(psi, "Tz", i, "Tz", j)
                for orbital_axis in ("x", "y"):
                    for spin_axis in ("x", "y", "z"):
                        mixed = 0.25 * (
                            _mps_corr(psi, f"S{spin_axis}Tp", i, f"S{spin_axis}Tm", j)
                            + _mps_corr(psi, f"S{spin_axis}Tm", i, f"S{spin_axis}Tp", j)
                        )
                        correlations[f"S{spin_axis}T{orbital_axis}_S{spin_axis}T{orbital_axis}"][i, j] = mixed
                for spin_axis in ("x", "y", "z"):
                    correlations[f"S{spin_axis}Tz_S{spin_axis}Tz"][i, j] = _mps_corr(
                        psi,
                        f"S{spin_axis}Tz",
                        i,
                        f"S{spin_axis}Tz",
                        j,
                    )
            if progress_bar is not None:
                progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return correlations


def build_spin_orbital_scalar_correlations(correlations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scalar = _ed_scalar_correlations(correlations)
    return {
        **scalar,
        "spin_scalar": scalar["S"],
        "orbital_scalar": scalar["T"],
        "mixed_scalar": scalar["ST"],
    }


def all_bond_energies(
    geometry: GeometryData,
    correlations: dict[str, np.ndarray],
    alpha: float,
    beta: float,
    coupling_j: float,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    return _ed_all_bond_energies(
        geometry,
        correlations,
        DEFAULT_MODEL_SPEC,
        alpha,
        beta,
        coupling_j,
        show_progress=show_progress,
        progress_desc="TeNPy bond energies",
    )


def all_high_symmetry_structure_factors(
    scalar_correlations: dict[str, np.ndarray],
    geometry: GeometryData,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    shared = {
        "S": scalar_correlations.get("S", scalar_correlations.get("spin_scalar")),
        "T": scalar_correlations.get("T", scalar_correlations.get("orbital_scalar")),
        "ST": scalar_correlations.get("ST", scalar_correlations.get("mixed_scalar")),
    }
    return _models_all_high_symmetry_structure_factors(
        shared,
        geometry,
        lattice="honeycomb",
        show_progress=show_progress,
        progress_desc="TeNPy structure factors",
    )
