#!/usr/bin/env python3
"""TeNPy U(1)-symmetric site/model template for the spin-orbital Yao-Lee model.

The physical local basis is fixed to

    0: Sdown_Odown
    1: Sdown_Oup
    2: Sup_Odown
    3: Sup_Oup

The conserved TeNPy charge is ``2*Sz`` with local charges ``[-1, -1, +1, +1]``.
Orbital operators are neutral and may freely mix ``Odown``/``Oup``.
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
        all_high_symmetry_structure_factors as _models_all_high_symmetry_structure_factors,
        build_honeycomb_cylinder_geometry,
        build_model_spec,
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
        all_high_symmetry_structure_factors as _models_all_high_symmetry_structure_factors,
        build_honeycomb_cylinder_geometry,
        build_model_spec,
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
        leave=False,
    )
    return _TENPY_SWEEP_PROGRESS_BAR


def _finish_tenpy_sweep_progress(progress_bar: Any | None) -> None:
    """Close the active TeNPy sweep progress bar."""
    global _TENPY_SWEEP_PROGRESS_BAR, _TENPY_SWEEP_PROGRESS_LAST
    if progress_bar is not None:
        progress_bar.close()
    if progress_bar is _TENPY_SWEEP_PROGRESS_BAR:
        _TENPY_SWEEP_PROGRESS_BAR = None
        _TENPY_SWEEP_PROGRESS_LAST = 0


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
        return dmrg.run(psi, model, options)
    finally:
        _finish_tenpy_sweep_progress(progress_bar)


class YaoLeeSite(Site):
    """One d=4 spin-1/2 tensor orbital-1/2 site conserving total spin Sz."""

    state_labels = ["Sdown_Odown", "Sdown_Oup", "Sup_Odown", "Sup_Oup"]

    def __init__(self, conserve: str | None = "Sz", sort_charge: bool = False) -> None:
        conserve_text = "Sz" if conserve in ("Sz", "sz", "U1", "u1", True) else "None"
        if conserve_text == "Sz":
            chinfo = npc.ChargeInfo([1], ["2*Sz"])
            leg = npc.LegCharge.from_qflat(chinfo, [-1, -1, 1, 1])
        else:
            leg = npc.LegCharge.from_trivial(4)

        spin_down_up = {
            "Sp": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
            "Sm": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
            "Sz": np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
            "Sx": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128),
            "Sy": np.array([[0.0, 0.5j], [-0.5j, 0.0]], dtype=np.complex128),
        }
        orbital_down_up = {
            "tau_p": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
            "tau_m": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
            "tau_z": np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
            "tau_x": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128),
            "tau_y": np.array([[0.0, 0.5j], [-0.5j, 0.0]], dtype=np.complex128),
        }
        spin_id = np.eye(2, dtype=np.complex128)
        orbital_id = np.eye(2, dtype=np.complex128)

        ops: dict[str, np.ndarray] = {
            "Sp": np.kron(spin_down_up["Sp"], orbital_id),
            "Sm": np.kron(spin_down_up["Sm"], orbital_id),
            "Sz": np.kron(spin_down_up["Sz"], orbital_id),
            "tau_p": np.kron(spin_id, orbital_down_up["tau_p"]),
            "tau_m": np.kron(spin_id, orbital_down_up["tau_m"]),
            "tau_z": np.kron(spin_id, orbital_down_up["tau_z"]),
            "tau_x": np.kron(spin_id, orbital_down_up["tau_x"]),
            "tau_y": np.kron(spin_id, orbital_down_up["tau_y"]),
        }
        if conserve_text != "Sz":
            ops["Sx"] = np.kron(spin_down_up["Sx"], orbital_id)
            ops["Sy"] = np.kron(spin_down_up["Sy"], orbital_id)
        alias_pairs = {
            "Tp": "tau_p",
            "Tm": "tau_m",
            "Tz": "tau_z",
            "Tx": "tau_x",
            "Ty": "tau_y",
        }
        for alias, source in alias_pairs.items():
            ops[alias] = ops[source]

        for spin_name in ("Sp", "Sm", "Sz"):
            for orbital_alias in ("Tx", "Ty", "Tz", "Tp", "Tm"):
                ops[f"{spin_name}{orbital_alias}"] = ops[spin_name] @ ops[orbital_alias]
            for orbital_name in ("tau_x", "tau_y", "tau_z", "tau_p", "tau_m"):
                ops[f"{spin_name}_{orbital_name}"] = ops[spin_name] @ ops[orbital_name]

        self.conserve = conserve_text
        super().__init__(leg, self.state_labels, sort_charge=sort_charge, **ops)
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, dtype=int)

    def __repr__(self) -> str:
        return f"YaoLeeSite(conserve='{self.conserve}')"


class YaoLeeModel(CouplingModel):
    """Minimal TeNPy CouplingModel, using dense sites when transverse fields require them."""

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
    ) -> None:
        field_terms = [
            (float(coefficient), str(op_name))
            for coefficient, op_name in list(external_field_terms or [])
            if abs(float(coefficient)) > 1e-14
        ]
        field_breaks_spin_u1 = any(op_name in ("Sx", "Sy") for _coefficient, op_name in field_terms)
        site = YaoLeeSite(conserve=None if field_breaks_spin_u1 else "Sz", sort_charge=sort_charge)
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
        self.spin_u1_conserved = not bool(field_breaks_spin_u1)
        for coefficient, op_name in self.external_field_terms:
            self.add_onsite_term(coefficient, 0, op_name, category=f"field_{op_name}")

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


def initialize_sz_zero_mps(model: YaoLeeModel, orbital_label: str = "Odown") -> MPS:
    """Initialize a TeNPy MPS exactly in the total ``Sz=0`` charge sector."""
    product_state = sz_zero_product_state_labels(model.lat.N_sites, orbital_label=orbital_label)
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
) -> tuple[YaoLeeModel, MPS, dict[str, Any]]:
    """Small runnable TeNPy template returning ``(model, psi, dmrg_options)``."""
    model = YaoLeeModel(
        geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        external_field_terms=external_field_terms,
    )
    psi = initialize_sz_zero_mps(model)
    options = {
        "mixer": None,
        "diag_method": "lanczos",
        "trunc_params": {"chi_max": 128, "svd_min": 1.0e-10},
        "max_trunc_err": None,
        "norm_tol": None,
        "max_sweeps": 20,
    }
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
    random_seed: int = 0,
    product_state_style: str = "alternating",
    initial_state: MPS | None = None,
    compute_phase_observables: bool = True,
    external_field_terms: list[tuple[float, str]] | None = None,
    show_progress: bool = True,
) -> tuple[MPS, Any, dict[str, Any]]:
    """Compatibility hook for ``ylmodel_main.py --backend tenpy``."""
    del random_seed, product_state_style
    stage_start = _start_stage("TeNPy finite DMRG", show_progress)
    _configure_tenpy_progress_logging(show_progress)
    if show_progress:
        print(
            "[tenpy-dmrg] setup: "
            f"N={int(geometry.number_of_sites)}, alpha={float(alpha):.8g}, beta={float(beta):.8g}, "
            f"chi_max={int(max_bond_dimension)}, sweeps={int(max_sweeps)}"
        )
    model = YaoLeeModel(
        geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        external_field_terms=external_field_terms,
    )
    psi = _copy_initial_state_for_model(model, initial_state)
    options = {
        # The TeNPy density-matrix mixer can make rho_L/rho_R ill-conditioned
        # for this charge-constrained product initialization.  Two-site DMRG
        # can grow the bond dimension without a mixer, so keep this off unless
        # a future caller deliberately opts into a custom TeNPy workflow.
        "mixer": None,
        "diag_method": "lanczos",
        "lanczos_params": {"N_min": 2, "N_max": 40},
        "max_trunc_err": None,
        "norm_tol": None,
        "max_sweeps": int(max_sweeps),
        "N_sweeps_check": 1,
        "trunc_params": {
            "chi_max": int(max_bond_dimension),
            "svd_min": float(truncation_cutoff),
        },
    }
    try:
        info = _run_dmrg_with_sweep_progress(
            psi,
            model,
            options,
            show_progress=show_progress,
            desc="tenpy dmrg sweeps",
            expected_sweeps=max(1, 2 * int(max_sweeps)),
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
        "symmetry_mode": "u1_sz" if bool(getattr(model, "spin_u1_conserved", True)) else "none",
        "symmetry_enabled": bool(getattr(model, "spin_u1_conserved", True)),
        "u1_target_sector": (
            {"mode": "u1_sz", "total_Sz_times_2": 0, "target_charge": 0}
            if bool(getattr(model, "spin_u1_conserved", True))
            else None
        ),
        "initial_state_style": (
            "adiabatic_previous_mps" if initial_state is not None else "alternating_sz_zero_product"
        ),
        "used_adiabatic_initial_state": bool(initial_state is not None),
        "mixer": None,
        "diag_method": "lanczos",
        "norm_error_after_canonicalization": norm_error_after_canonicalization,
        "external_field_terms": [
            (float(coefficient), str(op_name))
            for coefficient, op_name in list(external_field_terms or [])
        ],
    }
    if phase_observables is not None:
        dmrg_info["phase_observables"] = phase_observables
        dmrg_info["all_plaquette_fluxes"] = phase_observables.get("all_plaquette_fluxes", {})
    if phase_observable_warning is not None:
        dmrg_info["phase_observables_warning"] = phase_observable_warning
    if canonicalization_warning is not None:
        dmrg_info["canonicalization_warning"] = canonicalization_warning
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
    initial_state: MPS | None = None,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
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
            initial_state=previous_psi,
            compute_phase_observables=True,
            external_field_terms=external_field_terms,
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
    carry_state_between_betas: bool = False,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
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
            initial_state=beta_initial_state,
            classifier_thresholds=classifier_thresholds,
            external_field_terms=external_field_terms,
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
    initial_state: MPS | None = None,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
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
        )
        psi = _copy_initial_state_for_model(model, previous_psi)
        options = {
            "mixer": None,
            "diag_method": "lanczos",
            "lanczos_params": {"N_min": 2, "N_max": 40},
            "max_trunc_err": None,
            "norm_tol": None,
            "max_sweeps": int(max_iterations),
            "N_sweeps_check": 1,
            "trunc_params": {
                "chi_max": int(max_bond_dimension),
                "svd_min": float(truncation_cutoff),
            },
        }
        try:
            info = _run_dmrg_with_sweep_progress(
                psi,
                model,
                options,
                show_progress=show_progress,
                desc="tenpy idmrg scan sweeps",
                expected_sweeps=max(1, 2 * int(max_iterations)),
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
    truncation_cutoff: float = 1.0e-10,
    carry_state_between_betas: bool = False,
    classifier_thresholds: dict[str, float] | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run a TeNPy iDMRG observable scan over beta rows and alpha columns."""
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
            initial_state=beta_initial_state,
            classifier_thresholds=classifier_thresholds,
            external_field_terms=external_field_terms,
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
    random_seed: int = 0,
    product_state_style: str = "alternating",
    compute_entanglement: bool = True,
    compute_phase_observables: bool = True,
    initial_state: MPS | None = None,
    external_field_terms: list[tuple[float, str]] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run TeNPy infinite-DMRG along x with one finite cylinder as the unit cell."""
    del random_seed, product_state_style
    stage_start = _start_stage("TeNPy iDMRG-x", show_progress)
    _configure_tenpy_progress_logging(show_progress)
    if show_progress:
        print(
            "[tenpy-idmrg] setup: "
            f"N_unit_cell={int(geometry.number_of_sites)}, alpha={float(alpha):.8g}, beta={float(beta):.8g}, "
            f"chi_max={int(max_bond_dimension)}, iterations={int(max_iterations)}"
        )
    model = YaoLeeModel(
        geometry,
        alpha=alpha,
        beta=beta,
        coupling_j=coupling_j,
        bc_MPS="infinite",
        infinite_x=True,
        external_field_terms=external_field_terms,
    )
    psi = _copy_initial_state_for_model(model, initial_state)
    options = {
        "mixer": None,
        "diag_method": "lanczos",
        "lanczos_params": {"N_min": 2, "N_max": 40},
        "max_trunc_err": None,
        "norm_tol": None,
        "max_sweeps": int(max_iterations),
        "N_sweeps_check": 1,
        "trunc_params": {
            "chi_max": int(max_bond_dimension),
            "svd_min": float(truncation_cutoff),
        },
    }
    try:
        info = _run_dmrg_with_sweep_progress(
            psi,
            model,
            options,
            show_progress=show_progress,
            desc="tenpy idmrg sweeps",
            expected_sweeps=max(1, 2 * int(max_iterations)),
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
        "infinite_x_boundary_bonds": [
            {"i": int(i), "j": int(j), "gamma": str(gamma)}
            for i, j, gamma in infinite_x_boundary_bonds(geometry)
        ],
        "info": {
            "E": energy_density,
            "shelve": bool(info.get("shelve", False)),
            "symmetry_mode": "u1_sz",
            "max_sweeps": int(max_iterations),
            "max_bond_dimension": int(max_bond_dimension),
            "used_adiabatic_initial_state": bool(initial_state is not None),
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
            for axis, op in (("x", "Tx"), ("y", "Ty"), ("z", "Tz")):
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
