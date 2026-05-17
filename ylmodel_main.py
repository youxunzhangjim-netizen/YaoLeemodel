#!/usr/bin/env python3
"""
CLI orchestration for Yao-Lee model benchmarking with Tenax/TeNPy DMRG and optional ED.

The canonical work is split across sibling modules:
- models.py: model specs, local operators, geometry, and shared physics helpers.
- ed_backend.py: full ED plus bitwise total-Sz-conserved sparse ED.
- tenax_backend.py: Tenax MPO/DMRG/iDMRG execution.
- tenpy_backend.py: TeNPy YaoLeeSite/YaoLeeModel execution.
- peps_backend.py: optional quimb.tensor PEPS/iPEPS execution.
- analysis.py: phase scans, entropy, diagnostics, and summary helpers.
- plot_outputs.py: plotting helpers.

This file keeps settings, CLI, consistency checks, and run orchestration.
main() binds the split modules before running so fixes stay shared by owner.
"""

from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Tuple

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"llvmlite\.binding\.ffi",
)

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Configuration (edit this top block for normal runs)
# ----------------------------------------------------------------------
# ACTIVE_RESOURCE_PROFILE: local_laptop | shared_workstation
LOCAL_LAPTOP_SETTINGS = {
    "geometry": {
        "length_x": 2,
        "length_y": 2,
        "circumference_x": 1,
        "circumference_y": 1,
        "lattice_type": "honeycomb",
    },
    "finite_dmrg": {
        "max_sites": 18,
        "max_bond_dimension": 64,
        "max_sweeps": 20,
        "svd_min": 1.0e-10,
    },
    "finite_peps": {
        "max_sites": 18,
        "max_bond_dimension": 4,
        "bond_dimension_cap": 6,
        "max_sweeps": 40,
        "sweep_cap": 160,
        "ctm_chi": 64,
        "ctm_chi_cap": 96,
        "tau": 0.1,
        "entropy_max_dense_dim": 262144,
    },
    "ed": {
        "run": True,
        "max_sites": 16,
        "max_hilbert_dim": 5000000,
        "max_eigenstates": 6,
        "solver": "sparse",
        "sparse_tol": 0.0,
        "sparse_maxiter": 0,
    },
    "finite_temperature_ed": {
        "run": True,
        "max_sites": 16,
        "max_hilbert_dim": 5000000,
        "full_spectrum_max_dim": 512,
        "max_eigenstates": 12,
        "temperature_min": 0.05,
        "temperature_max": 2.0,
        "temperature_points": 32,
        "temperature_scale": "log",
    },
    "idmrg": {
        "run": True,
        "max_bond_dimension": 64,
        "max_iterations": 60,
        "max_local_dim": 16,
        "bulk_kind": "auto",
        "svd_min": 1.0e-10,
    },
    "ipeps": {
        "max_unit_cell_sites": 18,
        "max_bond_dimension": 4,
        "bond_dimension_cap": 6,
        "max_iterations": 80,
        "iteration_cap": 160,
        "ctm_chi": 64,
        "ctm_chi_cap": 96,
        "tau": 0.1,
    },
}

SHARED_WORKSTATION_SETTINGS = {
    "geometry": {
        "length_x": 3,
        "length_y": 3,
        "circumference_x": 1,
        "circumference_y": 1,
        "lattice_type": "honeycomb",
    },
    "finite_dmrg": {
        "max_sites": 32,
        "max_bond_dimension": 128,
        "max_sweeps": 60,
        "svd_min": 1.0e-10,
    },
    "finite_peps": {
        "max_sites": 32,
        "max_bond_dimension": 6,
        "bond_dimension_cap": 8,
        "max_sweeps": 120,
        "sweep_cap": 240,
        "ctm_chi": 96,
        "ctm_chi_cap": 160,
        "tau": 0.1,
        "entropy_max_dense_dim": 1048576,
    },
    "ed": {
        "run": True,
        "max_sites": 32,
        "max_hilbert_dim": 60000000,
        "max_eigenstates": 6,
        "solver": "sparse",
        "sparse_tol": 0.0,
        "sparse_maxiter": 0,
    },
    "finite_temperature_ed": {
        "run": True,
        "max_sites": 32,
        "max_hilbert_dim": 60000000,
        "full_spectrum_max_dim": 1024,
        "max_eigenstates": 24,
        "temperature_min": 0.03,
        "temperature_max": 2.5,
        "temperature_points": 40,
        "temperature_scale": "log",
    },
    "idmrg": {
        "run": True,
        "max_bond_dimension": 96,
        "max_iterations": 60,
        "max_local_dim": 16,
        "bulk_kind": "auto",
        "svd_min": 1.0e-10,
    },
    "ipeps": {
        "max_unit_cell_sites": 32,
        "max_bond_dimension": 6,
        "bond_dimension_cap": 8,
        "max_iterations": 120,
        "iteration_cap": 240,
        "ctm_chi": 96,
        "ctm_chi_cap": 160,
        "tau": 0.1,
    },
}

RESOURCE_PROFILES = {
    "local_laptop": LOCAL_LAPTOP_SETTINGS,
    "shared_workstation": SHARED_WORKSTATION_SETTINGS,
    }


ACTIVE_RESOURCE_PROFILE = "local_laptop"  # local_laptop | shared_workstation

BACKEND = "tenpy"  # auto | tenax | tenpy | quimb
METHOD = "dmrg"  # auto | dmrg | idmrg | peps | ipeps

MODEL_FAMILY = "yao_lee"  # yao_lee | ising_like | heisenberg | xy | xxz | xyz
SPIN_REP = "1/2"  # 1/2 | 3/2
ORBITAL_REP = "1/2"  # 0 | 1/2
ISING_AXIS = "z"  # x | y | z
ALPHA = 1.0
BETA = 0.5
COUPLING_J = 1.0
JX = 1.0
JY = 1.0
JZ = 1.0

EXTERNAL_FIELD_TREATMENT = "off"  # off | perturbation | hamiltonian
EXTERNAL_FIELD_AXIS = "111"  # 111 | 001 | custom
EXTERNAL_FIELD_STRENGTH = 1.0
FIELD_HX = 0.0
FIELD_HY = 0.0
FIELD_HZ = 1.0
MU_B = 1.0
FIELD_SIGN = -1.0
FIELD_SIGMA_FACTOR = 1.0

SYMMETRY_REDUCTIONS = ("tz", "z2")  # auto | none | sz | tz | z2
U1_TARGET_TOTAL_SZ2 = 0
U1_TARGET_TOTAL_TZ2 = 0
Z2_TARGET_PARITY = 0  # 0 | 1
STRICT_SYMMETRY_SELECTION_RULES = True
SYMMETRY_PRECHECK = True
STRICT_SYMMETRY_PRECHECK = True
SYMMETRY_ALLOW_DENSE_FALLBACK = True

TRUNCATION_CUTOFF = 1e-8
SEED = 42
INITIAL_STATE_STYLE = "random"  # alternating | random

ED_BACKEND = "quspin"  # standard | quspin
ED_SYMMETRY_ENGINE = "quspin"  # auto | standard_projector | quspin/quspin_native | quspin_experimental_c3
ED_QUSPIN_EXPERIMENTAL_FUSED_TRANSLATION = False
ED_C3_MODE = "off"  # auto | off | on
ED_C3_Q_BLOCKS = "all"  # all | 0 | 1 | 2
ED_Z2_MODE = "auto"  # auto | off | on
ED_Z2_KIND = "auto"  # auto | spin_flip | spin_pi_z
SZ_CONSERVED_ED_EIGENSTATES = 3
CHECK_GROUND_STATE_DEGENERACY = True
ED_GROUND_MANIFOLD_ABS_TOL = 1e-12
ED_GROUND_MANIFOLD_REL_TOL = 1e-12

DMRG_EXCITED_OVERLAP_TOL = 1e-6
DMRG_EXCITED_ENERGY_TOL = 1e-7
DMRG_EXCITED_VARIANCE_TOL = 1e-7
DMRG_EXCITED_MAX_ATTEMPTS = 10

USE_TRANSLATION_X_BLOCK = 1
USE_TRANSLATION_Y_BLOCK = 1
MOMENTUM_X_BLOCK = 0
MOMENTUM_Y_BLOCK = 0
USE_REFLECTION_BLOCK = 0
REFLECTION_BLOCK = 0
QUSPIN_CHECK_SYMMETRIES = False
QUSPIN_CHECK_HERMITICITY = True
QUSPIN_CHECK_PARTICLE_CONSERVATION = False

PHASE_SCAN_ONLY = 1
PHASE_DIAGRAM_ENABLED = 1
RUN_PHASE_SCAN = bool(PHASE_DIAGRAM_ENABLED) or bool(PHASE_SCAN_ONLY)

PHASE_SCAN_MODE = "quantum"  # quantum | classical | both
PHASE_SCAN_METHODS = "ed"  # ed | dmrg | idmrg | peps | ipeps | all
PHASE_SCAN_CHANNELS = "normal"  # auto | none | normal | external | both
EXTERNAL_SCAN_MODE = "e_b"  # none | e_b | alpha_b_classical | alpha_b_quantum | alpha_b_both | alpha_b_all
PHASE_SCAN_ALPHA_MIN = -0.25
PHASE_SCAN_ALPHA_MAX = 2.25
PHASE_SCAN_ALPHA_POINTS = 17
PHASE_SCAN_BETA_MIN = -0.25
PHASE_SCAN_BETA_MAX = 2.25
PHASE_SCAN_BETA_POINTS = 13
PHASE_SCAN_CLASSICAL_RESTARTS = 6
PHASE_SCAN_CLASSICAL_SWEEPS = 320
PHASE_SCAN_CLASSICAL_INITIAL_TEMPERATURE = 0.08
PHASE_SCAN_CLASSICAL_FINAL_TEMPERATURE = 0.001
PHASE_SCAN_CLASSICAL_INITIAL_STEP = 0.8
PHASE_SCAN_CLASSICAL_FINAL_STEP = 0.05
PHASE_SCAN_RANDOM_SEED = SEED + 1000
PHASE_SCAN_QUANTUM_WEAK_ORDER_THRESHOLD = 0.035
PHASE_SCAN_CLASSICAL_WEAK_ORDER_THRESHOLD = 0.075
PHASE_SCAN_QUANTUM_NEMATICITY_THRESHOLD = 0.10
PHASE_SCAN_CLASSICAL_NEMATICITY_THRESHOLD = 0.08
PHASE_SCAN_PLAQUETTE_FLUX_TARGET = 1.0
PHASE_SCAN_PLAQUETTE_FLUX_TOLERANCE = 0.15

EXTERNAL_SCAN_FIELD_MIN = -0.5
EXTERNAL_SCAN_FIELD_MAX = 2.0
EXTERNAL_SCAN_FIELD_POINTS = PHASE_SCAN_BETA_POINTS
EXTERNAL_SCAN_ED_BANDS = 5

OUTPUT_FOLDER = "outputs"
OVERWRITE_EXISTING_PLOTS = True
CONTINUE_AFTER_PLOT_ERROR = True
STRICT_PLOT_ERRORS = not CONTINUE_AFTER_PLOT_ERROR
SHOW_PROGRESS = True

PROFILE_ENABLED = False
PROFILE_TIMING = True
PROFILE_MEMORY = True
PROFILE_CPROFILE = False
PROFILE_LINE_HOOKS = False
PROFILE_SCAN_POINTS = True
PROFILE_OUTPUT_JSON = True
PROFILE_OUTPUT_FOLDER = "outputs/profiling"


def _resolve_output_folder(output_folder: str | None) -> str:
    """Resolve output folders to the canonical folder beside this script.

    Normal runs should write to ``<repo>/DMRG/outputs``.  If a user launches
    from inside ``DMRG`` and passes ``--output-folder DMRG/outputs``, collapse
    the accidental duplicate path ``DMRG/DMRG/outputs`` back to the canonical
    sibling output folder.
    """
    raw_folder = OUTPUT_FOLDER if output_folder is None else str(output_folder)
    expanded_folder = os.path.expanduser(raw_folder)
    if os.path.isabs(expanded_folder):
        candidate = os.path.abspath(expanded_folder)
    else:
        candidate = os.path.abspath(os.path.join(SCRIPT_DIR, expanded_folder))
    script_name = os.path.basename(SCRIPT_DIR)
    duplicate_root = os.path.abspath(os.path.join(SCRIPT_DIR, script_name))
    if candidate == duplicate_root or candidate.startswith(duplicate_root + os.sep):
        candidate = os.path.abspath(
            os.path.join(os.path.dirname(SCRIPT_DIR), os.path.relpath(candidate, SCRIPT_DIR))
        )
    return candidate

CALCULATE_CORRELATIONS = True
CALCULATE_BOND_ENERGIES = True
CALCULATE_STRUCTURE_FACTORS = True
CALCULATE_ENTANGLEMENT = True
CALCULATE_UNIFORM_OBSERVABLES = True
CALCULATE_REAL_SPACE_PATTERNS = True
REFERENCE_SITE_IDX = None

PLOT_GEOMETRY = 0
PLOT_BOND_ENERGIES = True
PLOT_STRUCTURE_FACTORS = True
PLOT_CORRELATION_HEATMAPS = True
PLOT_REAL_SPACE_PATTERNS = True
PLOT_ENTANGLEMENT = True
PLOT_ENERGY_COMPARISON = True
PLOT_LOW_ENERGY_SPECTRUM = True
PLOT_FINITE_TEMPERATURE = True
PLOT_PHASE_SCAN = RUN_PHASE_SCAN

# ----------------------------------------------------------------------
# Derived/profile-linked defaults and available choices
# ----------------------------------------------------------------------

ACTIVE_RESOURCE_SETTINGS = RESOURCE_PROFILES[ACTIVE_RESOURCE_PROFILE]

LENGTH_X = int(ACTIVE_RESOURCE_SETTINGS["geometry"]["length_x"])
LENGTH_Y = int(ACTIVE_RESOURCE_SETTINGS["geometry"]["length_y"])
CIRCUMFERENCE_X = bool(ACTIVE_RESOURCE_SETTINGS["geometry"]["circumference_x"])
CIRCUMFERENCE_Y = bool(ACTIVE_RESOURCE_SETTINGS["geometry"]["circumference_y"])
LATTICE_TYPE = str(ACTIVE_RESOURCE_SETTINGS["geometry"]["lattice_type"])  # honeycomb | square | triangular

MAX_DMRG_SITES = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_sites"])
MAX_BOND_DIMENSION = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_bond_dimension"])
MAX_SWEEPS = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_sweeps"])
DMRG_SVD_MIN = float(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"].get("svd_min", 1.0e-10))

MAX_PEPS_SITES = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["max_sites"])
PEPS_MAX_BOND_DIMENSION = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["max_bond_dimension"])
PEPS_BOND_DIMENSION_CAP = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["bond_dimension_cap"])
PEPS_MAX_SWEEPS = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["max_sweeps"])
PEPS_SWEEP_CAP = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["sweep_cap"])
PEPS_CTM_CHI = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["ctm_chi"])
PEPS_CTM_CHI_CAP = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["ctm_chi_cap"])
PEPS_TAU = float(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["tau"])
PEPS_ENTANGLEMENT_MAX_DENSE_DIM = int(ACTIVE_RESOURCE_SETTINGS["finite_peps"]["entropy_max_dense_dim"])
PEPS_SYMMETRY_MODE = "auto"  # auto | none | u1_tz | u1_tz_z2
PEPS_STRICT_SYMMETRY = True
PEPS_ALLOW_DENSE_FALLBACK = True

RUN_ED = bool(ACTIVE_RESOURCE_SETTINGS["ed"]["run"])
MAX_ED_SITES = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_sites"])
MAX_ED_HILBERT_DIM = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_hilbert_dim"])
ED_MAX_EIGENSTATES = int(ACTIVE_RESOURCE_SETTINGS["ed"].get("max_eigenstates", 20))
ED_SOLVER = str(ACTIVE_RESOURCE_SETTINGS["ed"].get("solver", "sparse"))
ED_SPARSE_TOL = float(ACTIVE_RESOURCE_SETTINGS["ed"].get("sparse_tol", 0.0))
ED_SPARSE_MAXITER = int(ACTIVE_RESOURCE_SETTINGS["ed"].get("sparse_maxiter", 0))

RUN_FINITE_TEMPERATURE_ED = bool(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["run"])
MAX_THERMAL_ED_SITES = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["max_sites"])
MAX_THERMAL_ED_HILBERT_DIM = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["max_hilbert_dim"])
THERMAL_FULL_SPECTRUM_MAX_DIM = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["full_spectrum_max_dim"])
THERMAL_MAX_EIGENSTATES = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["max_eigenstates"])
TEMPERATURE_MIN = float(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_min"])
TEMPERATURE_MAX = float(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_max"])
TEMPERATURE_POINTS = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_points"])
TEMPERATURE_SCALE = str(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_scale"])

RUN_IDMRG = bool(ACTIVE_RESOURCE_SETTINGS["idmrg"]["run"])
IDMRG_MAX_BOND_DIMENSION = int(ACTIVE_RESOURCE_SETTINGS["idmrg"]["max_bond_dimension"])
IDMRG_MAX_ITERATIONS = int(ACTIVE_RESOURCE_SETTINGS["idmrg"]["max_iterations"])
IDMRG_MAX_LOCAL_DIM = int(ACTIVE_RESOURCE_SETTINGS["idmrg"]["max_local_dim"])
IDMRG_BULK_KIND = str(ACTIVE_RESOURCE_SETTINGS["idmrg"]["bulk_kind"])  # auto | pair | single
IDMRG_SVD_MIN = float(ACTIVE_RESOURCE_SETTINGS["idmrg"].get("svd_min", DMRG_SVD_MIN))
IDMRG_USE_TRANSLATION_SYMMETRY = True

MAX_IPEPS_UNIT_CELL_SITES = int(ACTIVE_RESOURCE_SETTINGS["ipeps"]["max_unit_cell_sites"])
IPEPS_MAX_BOND_DIMENSION = int(ACTIVE_RESOURCE_SETTINGS["ipeps"]["max_bond_dimension"])
IPEPS_BOND_DIMENSION_CAP = int(ACTIVE_RESOURCE_SETTINGS["ipeps"]["bond_dimension_cap"])
IPEPS_MAX_ITERATIONS = int(ACTIVE_RESOURCE_SETTINGS["ipeps"]["max_iterations"])
IPEPS_ITERATION_CAP = int(ACTIVE_RESOURCE_SETTINGS["ipeps"]["iteration_cap"])
IPEPS_CTM_CHI = int(ACTIVE_RESOURCE_SETTINGS["ipeps"]["ctm_chi"])
IPEPS_CTM_CHI_CAP = int(ACTIVE_RESOURCE_SETTINGS["ipeps"]["ctm_chi_cap"])
IPEPS_TAU = float(ACTIVE_RESOURCE_SETTINGS["ipeps"]["tau"])
IPEPS_SYMMETRY_MODE = "auto"  # auto | none | u1_tz | u1_tz_z2
IPEPS_STRICT_SYMMETRY = True
IPEPS_ALLOW_DENSE_FALLBACK = True
IPEPS_UNIT_CELL_KIND = "auto"  # auto | minimal | two_sublattice | stripy | zigzag | plaquette
IPEPS_USE_TRANSLATION_SYMMETRY = True
IPEPS_CONTRACTION_METHOD = "ctmrg"  # auto | ctmrg | crtg | boundary
IPEPS_UNIT_CELL_CANDIDATES = (
    "minimal",
    "two_sublattice",
    "stripy",
    "zigzag",
    "plaquette",
)

PHASE_SCAN_ED_MAX_SITES = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_sites"])
PHASE_SCAN_ED_MAX_HILBERT_DIM = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_hilbert_dim"])

# Available choices for argparse and validation.
LATTICE_OPTIONS = ("honeycomb", "square", "triangular")
SPIN_ONLY_MODEL_FAMILIES = ("heisenberg", "xy", "xxz", "xyz")
MODEL_FAMILY_OPTIONS = ("yao_lee", "ising_like") + SPIN_ONLY_MODEL_FAMILIES
SPIN_REP_OPTIONS = ("1/2", "3/2")
ORBITAL_REP_OPTIONS = ("0", "1/2")
AXIS_OPTIONS = ("x", "y", "z")
INITIAL_STATE_OPTIONS = ("alternating", "random")
SYMMETRY_MODE_OPTIONS = ("none", "auto", "u1", "u1_sz", "u1_tz", "z2", "u1_tz_z2", "tz_z2")
SYMMETRY_REDUCTION_OPTIONS = ("auto", "none", "sz", "tz", "z2", "u1", "u1_sz", "u1_tz", "u1_tz_z2", "tz_z2")
PEPS_SYMMETRY_MODE_OPTIONS = ("auto", "none", "u1_tz", "u1_tz_z2")
IPEPS_SYMMETRY_MODE_OPTIONS = PEPS_SYMMETRY_MODE_OPTIONS
IPEPS_UNIT_CELL_KIND_OPTIONS = ("auto",) + IPEPS_UNIT_CELL_CANDIDATES
IPEPS_CONTRACTION_METHOD_OPTIONS = ("auto", "ctmrg", "crtg", "boundary")
U1_CHARGE_TZ_STRIDE = 4096
Z2_PARITY_OPTIONS = (0, 1)
IDMRG_BULK_KIND_OPTIONS = ("auto", "pair", "single")
BACKEND_OPTIONS = ("auto", "tenax", "tenpy", "quimb")
ED_BACKEND_OPTIONS = ("standard", "ed", "quspin")
ED_SOLVER_OPTIONS = ("auto", "sparse", "dense")
ED_SYMMETRY_ENGINE_OPTIONS = (
    "auto",
    "standard_projector",
    "quspin_native",
    "quspin",  # readable alias for quspin_native
    "quspin_experimental_c3",
    "projector",  # legacy alias for standard_projector
)
ED_C3_MODE_OPTIONS = ("auto", "off", "on")
ED_C3_Q_BLOCK_OPTIONS = ("all", "0", "1", "2")
ED_Z2_MODE_OPTIONS = ("auto", "off", "on")
ED_Z2_KIND_OPTIONS = ("auto", "spin_flip", "spin_pi_z")
EXTERNAL_FIELD_TREATMENT_OPTIONS = ("off", "perturbation", "hamiltonian")
EXTERNAL_FIELD_AXIS_OPTIONS = ("custom", "111", "001")
PHASE_SCAN_QUANTUM_METHOD_OPTIONS = ("ed", "dmrg", "idmrg", "peps", "ipeps")
PHASE_SCAN_METHOD_OPTIONS = PHASE_SCAN_QUANTUM_METHOD_OPTIONS + ("all",)
PHASE_SCAN_CHANNEL_OPTIONS = ("auto", "none", "normal", "external", "both")
EXTERNAL_SCAN_MODE_OPTIONS = (
    "none",
    "e_b",
    "alpha_b_classical",
    "alpha_b_quantum",
    "alpha_b_both",
    "alpha_b_all",
)
CALCULATION_METHOD_OPTIONS = (
    "auto",
    "dmrg",
    "idmrg",
    "peps",
    "ipeps",
    "finite_peps",
    "infinite_peps",
    "quimb_peps",
    "quimb_ipeps",
)
PHASE_SCAN_MODE_OPTIONS = (
    "quantum",
    "classical",
    "both",
    "all",
    # Legacy aliases accepted for old command lines.
    "methods",
    "quantum_ed",
    "ed",
    "exact",
    "exact_diagonalization",
    "classical_product",
    "dmrg",
    "finite_dmrg",
    "idmrg",
    "infinite_dmrg",
    "peps",
    "finite_peps",
    "ipeps",
    "infinite_peps",
    "tenax_dmrg",
    "tenax_idmrg",
    "tenpy_dmrg",
    "tenpy_idmrg",
    "quimb_peps",
    "quimb_ipeps",
)
REFLECTION_BLOCK_OPTIONS = (-1, 0, 1)
ENTROPY_ORDERS = (1, 2, 3, 4)

# Runtime implementation symbols are imported from sibling modules by
# _bind_split_module_implementations(); optional PEPS code is loaded lazily
# only when selected. Keep implementation work in: models.py, ed_backend.py,
# tenax_backend.py, tenpy_backend.py, peps_backend.py, analysis.py, and
# plot_outputs.py.
# ----------------------------------------------------------------------
# CLI + main
# ----------------------------------------------------------------------

def _parse_boundary_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in ("true", "t", "yes", "y", "on", "pbc", "periodic"):
        return True
    if text in ("false", "f", "no", "n", "off", "obc", "open"):
        return False
    raise argparse.ArgumentTypeError("Boundary flags expect true/false, pbc/obc, on/off.")


def _normalize_geometry_cli_args(args: argparse.Namespace) -> None:
    length_y = int(getattr(args, "length_y", LENGTH_Y))
    if length_y <= 0:
        raise ValueError("length_y must be positive.")
    circumference_x = CIRCUMFERENCE_X
    circumference_y = CIRCUMFERENCE_Y

    circumference_x_value = getattr(args, "circumference_x", None)
    if circumference_x_value is not None:
        circumference_x = bool(circumference_x_value)

    circumference_y_value = getattr(args, "circumference_y", None)
    if circumference_y_value is not None:
        circumference_y = bool(circumference_y_value)

    args.length_y = int(length_y)
    args.circumference_x = bool(circumference_x)
    args.circumference_y = bool(circumference_y)


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tenax/TeNPy/ED/QuSpin benchmarking for the Yao-Lee model.")
    parser.add_argument(
        "--spin-rep",
        "--spin_rep",
        dest="spin_rep",
        type=str,
        choices=list(SPIN_REP_OPTIONS),
        default=SPIN_REP,
        help="Spin representation: 1/2 or 3/2.",
    )
    parser.add_argument(
        "--orbital-rep",
        "--orbital_rep",
        dest="orbital_rep",
        type=str,
        choices=list(ORBITAL_REP_OPTIONS) + ["1"],
        default=ORBITAL_REP,
        help="Orbital representation: 0 (trivial) or 1/2. (Legacy alias: 1 -> 0)",
    )
    parser.add_argument(
        "--model-family",
        "--model_family",
        dest="model_family",
        type=str,
        choices=list(MODEL_FAMILY_OPTIONS),
        default=MODEL_FAMILY,
        help=(
            "Hamiltonian family. yao_lee uses the paper Eq. 7 spin-orbital bond formula."
        ),
    )
    parser.add_argument(
        "--ising-axis",
        "--ising_axis",
        dest="ising_axis",
        type=str,
        choices=list(AXIS_OPTIONS),
        default=ISING_AXIS,
        help="Spin/orbital axis used by ising_like couplings (and orbital_rep=0 fallback).",
    )
    parser.add_argument(
        "--symmetry-reductions",
        "--symmetry_reductions",
        dest="symmetry_reductions",
        default=None,
        help=(
            "Additive shared symmetry reductions for ED/QuSpin/DMRG. "
            "Use comma-separated values from: auto, none, sz, tz, z2. "
            "For no-field Yao-Lee QuSpin ED, use tz,z2 for orbital Tz plus spin-flip Z2. "
            "Aliases u1/u1_sz/u1_tz/u1_tz_z2 are accepted."
        ),
    )
    parser.add_argument(
        "--symmetry-mode",
        "--symmetry_mode",
        dest="symmetry_mode",
        type=str,
        choices=list(SYMMETRY_MODE_OPTIONS),
        default=None,
        help=(
            "Legacy symmetry shortcut. Prefer --symmetry-reductions to combine sz/tz/z2."
        ),
    )
    parser.add_argument(
        "--u1-target-sz2",
        "--u1_target_sz2",
        dest="u1_target_sz2",
        type=int,
        default=U1_TARGET_TOTAL_SZ2,
        help="Target total 2*Sz sector when the sz reduction is active.",
    )
    parser.add_argument(
        "--u1-target-tz2",
        "--u1_target_tz2",
        dest="u1_target_tz2",
        type=int,
        default=U1_TARGET_TOTAL_TZ2,
        help="Target total 2*Tz sector when the tz reduction is active.",
    )
    parser.add_argument(
        "--z2-target-parity",
        "--z2_target_parity",
        dest="z2_target_parity",
        type=int,
        choices=list(Z2_PARITY_OPTIONS),
        default=Z2_TARGET_PARITY,
        help="Target global parity sector when the z2 reduction is active (0=even, 1=odd).",
    )
    parser.add_argument(
        "--strict-symmetry-selection-rules",
        "--strict_symmetry_selection_rules",
        dest="strict_symmetry_selection_rules",
        action=argparse.BooleanOptionalAction,
        default=STRICT_SYMMETRY_SELECTION_RULES,
        help="If enabled, reject MPO terms that violate the active U1/Z2 selection rule.",
    )
    parser.add_argument(
        "--symmetry-precheck",
        "--symmetry_precheck",
        dest="symmetry_precheck",
        action=argparse.BooleanOptionalAction,
        default=SYMMETRY_PRECHECK,
        help="Analyze whether the generated Hamiltonian really conserves U1/Z2 before DMRG.",
    )
    parser.add_argument(
        "--strict-symmetry-precheck",
        "--strict_symmetry_precheck",
        dest="strict_symmetry_precheck",
        action=argparse.BooleanOptionalAction,
        default=STRICT_SYMMETRY_PRECHECK,
        help="Fail before DMRG when the requested symmetry is not conserved, unreachable, or unsupported.",
    )
    parser.add_argument(
        "--symmetry-allow-dense-fallback",
        "--symmetry_allow_dense_fallback",
        dest="symmetry_allow_dense_fallback",
        action=argparse.BooleanOptionalAction,
        default=SYMMETRY_ALLOW_DENSE_FALLBACK,
        help="Allow a requested symmetry run to continue in dense symmetry_mode=none if Tenax cannot impose it.",
    )
    parser.add_argument(
        "--lattice",
        type=str,
        choices=list(LATTICE_OPTIONS),
        default=LATTICE_TYPE,
        help="Lattice type used to build the Hamiltonian bonds.",
    )
    parser.add_argument(
        "--length-x",
        "--length_x",
        dest="length_x",
        type=int,
        default=LENGTH_X,
        help="Number of unit cells along x.",
    )
    parser.add_argument(
        "--length-y",
        "--length_y",
        dest="length_y",
        type=int,
        default=LENGTH_Y,
        help="Number of unit cells along y.",
    )
    parser.add_argument(
        "--circumference-x",
        "--circumference_x",
        dest="circumference_x",
        nargs="?",
        const=True,
        default=None,
        type=_parse_boundary_bool,
        help="Use periodic boundary conditions along x. Use --no-circumference-x for open x.",
    )
    parser.add_argument(
        "--no-circumference-x",
        "--no_circumference_x",
        dest="circumference_x",
        action="store_false",
        help="Use open boundary conditions along x.",
    )
    parser.add_argument(
        "--circumference-y",
        "--circumference_y",
        dest="circumference_y",
        nargs="?",
        const=True,
        default=None,
        type=_parse_boundary_bool,
        help="Use periodic boundary conditions along y. Use --no-circumference-y for open y.",
    )
    parser.add_argument(
        "--no-circumference-y",
        "--no_circumference_y",
        dest="circumference_y",
        action="store_false",
        help="Use open boundary conditions along y.",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Model alpha parameter.")
    parser.add_argument("--beta", type=float, default=BETA, help="Model beta parameter.")
    parser.add_argument(
        "--coupling-j",
        "--coupling_j",
        dest="coupling_j",
        type=float,
        default=COUPLING_J,
        help="Overall exchange scale J. Use a non-zero value for a non-empty Hamiltonian.",
    )
    parser.add_argument("--jx", type=float, default=JX, help="Sx coupling multiplier for XY/XXZ/XYZ models.")
    parser.add_argument("--jy", type=float, default=JY, help="Sy coupling multiplier for XY/XXZ/XYZ models.")
    parser.add_argument("--jz", type=float, default=JZ, help="Sz coupling multiplier for XXZ/XYZ models.")
    parser.add_argument(
        "--external-field-treatment",
        "--external_field_treatment",
        dest="external_field_treatment",
        type=str,
        choices=list(EXTERNAL_FIELD_TREATMENT_OPTIONS),
        default=EXTERNAL_FIELD_TREATMENT,
        help=(
            "External spin field handling: off, perturbation (annotate only), "
            "or hamiltonian (add one-site Zeeman terms)."
        ),
    )
    parser.add_argument(
        "--external-field-axis",
        "--external_field_axis",
        dest="external_field_axis",
        type=str,
        choices=list(EXTERNAL_FIELD_AXIS_OPTIONS),
        default=EXTERNAL_FIELD_AXIS,
        help=(
            "External field direction source: 111 uses H/sqrt(3)*(1,1,1); "
            "001 uses H*(0,0,1); custom uses hx/hy/hz."
        ),
    )
    parser.add_argument(
        "--external-field-strength",
        "--external_field_strength",
        dest="external_field_strength",
        type=float,
        default=EXTERNAL_FIELD_STRENGTH,
        help="Field magnitude H used when external_field_axis is 111 or 001.",
    )
    parser.add_argument("--field-hx", "--field_hx", dest="field_hx", type=float, default=FIELD_HX)
    parser.add_argument("--field-hy", "--field_hy", dest="field_hy", type=float, default=FIELD_HY)
    parser.add_argument("--field-hz", "--field_hz", dest="field_hz", type=float, default=FIELD_HZ)
    parser.add_argument("--mu-b", "--mu_b", dest="mu_b", type=float, default=MU_B)
    parser.add_argument("--field-sign", "--field_sign", dest="field_sign", type=float, default=FIELD_SIGN)
    parser.add_argument(
        "--field-sigma-factor",
        "--field_sigma_factor",
        dest="field_sigma_factor",
        type=float,
        default=FIELD_SIGMA_FACTOR,
        help=(
            "Multiplier on spin operators in the Zeeman term. The default 1 uses physical "
            "spin S; use 2 only for an explicit Pauli sigma convention."
        ),
    )
    parser.add_argument(
        "--max-bond-dimension",
        "--max_bond_dimension",
        "--dmrg-final-max-bond-dimension",
        "--dmrg_final_max_bond_dimension",
        dest="max_bond_dimension",
        type=int,
        default=MAX_BOND_DIMENSION,
        help="Finite-DMRG final maximum bond dimension after warmup.",
    )
    parser.add_argument(
        "--max-dmrg-sites",
        "--max_dmrg_sites",
        dest="max_dmrg_sites",
        type=int,
        default=MAX_DMRG_SITES,
        help=(
            "Shared-workstation finite-DMRG site safety cap. Increase deliberately "
            "for aragorn/beehive runs."
        ),
    )
    parser.add_argument(
        "--max-sweeps",
        "--max_sweeps",
        dest="max_sweeps",
        type=int,
        default=MAX_SWEEPS,
        help="Finite-DMRG maximum sweep count.",
    )
    parser.add_argument(
        "--dmrg-svd-min",
        "--dmrg_svd_min",
        dest="dmrg_svd_min",
        type=float,
        default=DMRG_SVD_MIN,
        help=(
            "Finite-DMRG SVD singular-value truncation threshold passed as TeNPy "
            "trunc_params['svd_min']; set 0 to disable this floor."
        ),
    )
    parser.add_argument(
        "--max-peps-sites",
        "--max_peps_sites",
        dest="max_peps_sites",
        type=int,
        default=MAX_PEPS_SITES,
        help="Finite-PEPS site safety cap, independent of finite-DMRG max sites.",
    )
    parser.add_argument(
        "--peps-max-bond-dimension",
        "--peps_max_bond_dimension",
        dest="peps_max_bond_dimension",
        type=int,
        default=PEPS_MAX_BOND_DIMENSION,
        help="Finite-PEPS virtual bond dimension D, independent of DMRG chi.",
    )
    parser.add_argument(
        "--peps-bond-dimension-cap",
        "--peps_bond_dimension_cap",
        dest="peps_bond_dimension_cap",
        type=int,
        default=PEPS_BOND_DIMENSION_CAP,
        help="Profile safety cap for finite-PEPS bond dimension; raise deliberately for larger devices.",
    )
    parser.add_argument(
        "--peps-max-sweeps",
        "--peps_max_sweeps",
        dest="peps_max_sweeps",
        type=int,
        default=PEPS_MAX_SWEEPS,
        help="Finite-PEPS Simple Update steps/sweeps, independent of DMRG sweeps.",
    )
    parser.add_argument(
        "--peps-sweep-cap",
        "--peps_sweep_cap",
        dest="peps_sweep_cap",
        type=int,
        default=PEPS_SWEEP_CAP,
        help="Profile safety cap for finite-PEPS Simple Update steps.",
    )
    parser.add_argument(
        "--peps-ctm-chi",
        "--peps_ctm_chi",
        dest="peps_ctm_chi",
        type=int,
        default=PEPS_CTM_CHI,
        help="Finite-PEPS boundary/CTMRG contraction chi.",
    )
    parser.add_argument(
        "--peps-ctm-chi-cap",
        "--peps_ctm_chi_cap",
        dest="peps_ctm_chi_cap",
        type=int,
        default=PEPS_CTM_CHI_CAP,
        help="Profile safety cap for finite-PEPS CTMRG chi.",
    )
    parser.add_argument(
        "--peps-tau",
        "--peps_tau",
        dest="peps_tau",
        type=float,
        default=PEPS_TAU,
        help="Finite-PEPS Simple Update imaginary-time step.",
    )
    parser.add_argument(
        "--peps-entanglement-max-dense-dim",
        "--peps_entanglement_max_dense_dim",
        dest="peps_entanglement_max_dense_dim",
        type=int,
        default=PEPS_ENTANGLEMENT_MAX_DENSE_DIM,
        help="Dense Hilbert-dimension cap for optional finite-PEPS entanglement post-processing.",
    )
    parser.add_argument(
        "--peps-symmetry-mode",
        "--peps_symmetry_mode",
        dest="peps_symmetry_mode",
        type=str,
        choices=list(PEPS_SYMMETRY_MODE_OPTIONS),
        default=PEPS_SYMMETRY_MODE,
        help="Finite-PEPS tensor symmetry request: auto, none, u1_tz, or u1_tz_z2.",
    )
    parser.add_argument(
        "--peps-strict-symmetry",
        "--peps_strict_symmetry",
        dest="peps_strict_symmetry",
        action=argparse.BooleanOptionalAction,
        default=PEPS_STRICT_SYMMETRY,
        help="Raise when a requested finite-PEPS spin-sector Z2 tensor symmetry is unsupported.",
    )
    parser.add_argument(
        "--peps-allow-dense-fallback",
        "--peps_allow_dense_fallback",
        dest="peps_allow_dense_fallback",
        action=argparse.BooleanOptionalAction,
        default=PEPS_ALLOW_DENSE_FALLBACK,
        help="Allow finite PEPS to run dense when requested tensor symmetries are unsupported.",
    )
    parser.add_argument(
        "--truncation-cutoff",
        "--truncation_cutoff",
        dest="truncation_cutoff",
        type=float,
        default=TRUNCATION_CUTOFF,
        help="Tensor truncation cutoff for TeNPy and quimb PEPS/iPEPS; Tenax backend may ignore it.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--initial-state",
        "--initial_state",
        dest="initial_state",
        type=str,
        choices=list(INITIAL_STATE_OPTIONS),
        default=INITIAL_STATE_STYLE,
        help=(
            "Initial MPS style. In Tenax U1 runs, alternating builds an exact total-Sz=0 "
            "product state; random builds a random symmetric MPS in the requested sector."
        ),
    )
    parser.add_argument(
        "--run-ed",
        "--run_ed",
        dest="run_ed",
        action=argparse.BooleanOptionalAction,
        default=RUN_ED,
        help="Run exact diagonalization comparison when the Hilbert-space size is within the configured limits.",
    )
    parser.add_argument(
        "--max-ed-sites",
        "--max_ed_sites",
        dest="max_ed_sites",
        type=int,
        default=MAX_ED_SITES,
        help="Site cap for the single-point ED comparison.",
    )
    parser.add_argument(
        "--max-ed-hilbert-dim",
        "--max_ed_hilbert_dim",
        dest="max_ed_hilbert_dim",
        type=int,
        default=MAX_ED_HILBERT_DIM,
        help="Hilbert-space dimension cap for the single-point ED comparison.",
    )
    parser.add_argument(
        "--ed-max-eigenstates",
        "--ed_max_eigenstates",
        dest="ed_max_eigenstates",
        type=int,
        default=ED_MAX_EIGENSTATES,
        help="Number of lowest ED eigenstates to request for the low-energy comparison.",
    )
    parser.add_argument(
        "--ed-backend",
        "--ed_backend",
        dest="ed_backend",
        type=str,
        choices=list(ED_BACKEND_OPTIONS),
        default=ED_BACKEND,
        help=(
            "Exact-diagonalization backend used for ED comparison/phase scans. "
            "standard uses ed_backend.py; ed is an alias for standard; quspin uses quspin_backend.py."
        ),
    )
    parser.add_argument(
        "--ed-solver",
        "--ed_solver",
        dest="ed_solver",
        type=str,
        choices=list(ED_SOLVER_OPTIONS),
        default=ED_SOLVER,
        help="ED eigensolver: sparse requests ARPACK eigsh, dense computes the full dense spectrum, auto keeps the legacy fallback.",
    )
    parser.add_argument(
        "--ed-symmetry-engine",
        "--ed_symmetry_engine",
        dest="ed_symmetry_engine",
        type=str,
        choices=list(ED_SYMMETRY_ENGINE_OPTIONS),
        default=ED_SYMMETRY_ENGINE,
        help=(
            "ED symmetry engine: auto chooses the fastest physically valid route; "
            "standard_projector uses in-repo Tz/spin_pi_z/fused-translation/combined-C3 projectors; "
            "quspin or quspin_native uses only QuSpin-representable symmetries; "
            "quspin_experimental_c3 checks QuSpin custom-basis API support but rejects pure C3 maps for Yao-Lee."
        ),
    )
    parser.add_argument(
        "--ed-quspin-experimental-fused-translation",
        "--ed_quspin_experimental_fused_translation",
        dest="ed_quspin_experimental_fused_translation",
        action=argparse.BooleanOptionalAction,
        default=ED_QUSPIN_EXPERIMENTAL_FUSED_TRANSLATION,
        help=(
            "Experimental opt-in for a future QuSpin packed user_basis route implementing fused "
            "spin-orbital translations with Tz. The current code probes API support but still "
            "routes Tz+translation to standard_projector unless that path is implemented and covered by tests."
        ),
    )
    parser.add_argument(
        "--ed-c3-mode",
        "--ed_c3_mode",
        dest="ed_c3_mode",
        type=str,
        choices=list(ED_C3_MODE_OPTIONS),
        default=ED_C3_MODE,
        help="ED combined spin-lattice C3 projector request: auto, off, or on.",
    )
    parser.add_argument(
        "--ed-c3-q-blocks",
        "--ed_c3_q_blocks",
        dest="ed_c3_q_blocks",
        type=str,
        choices=list(ED_C3_Q_BLOCK_OPTIONS),
        default=ED_C3_Q_BLOCKS,
        help="Combined C3 charge sectors for ED projector planning: all, 0, 1, or 2.",
    )
    parser.add_argument(
        "--ed-z2-mode",
        "--ed_z2_mode",
        dest="ed_z2_mode",
        type=str,
        choices=list(ED_Z2_MODE_OPTIONS),
        default=ED_Z2_MODE,
        help="ED spin-sector Z2 projector request: auto, off, or on.",
    )
    parser.add_argument(
        "--ed-z2-kind",
        "--ed_z2_kind",
        dest="ed_z2_kind",
        type=str,
        choices=list(ED_Z2_KIND_OPTIONS),
        default=ED_Z2_KIND,
        help="ED Z2 kind: auto, spin_flip, or spin_pi_z.",
    )
    parser.add_argument(
        "--use-sz-conserved",
        "--use_sz_conserved",
        dest="use_sz_conserved",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ed-sparse-tol",
        "--ed_sparse_tol",
        dest="ed_sparse_tol",
        type=float,
        default=ED_SPARSE_TOL,
        help="Optional eigsh tolerance for sparse ED; 0 keeps SciPy's machine-precision default.",
    )
    parser.add_argument(
        "--ed-sparse-maxiter",
        "--ed_sparse_maxiter",
        dest="ed_sparse_maxiter",
        type=int,
        default=ED_SPARSE_MAXITER,
        help="Optional eigsh max iterations for sparse ED; 0 lets SciPy choose.",
    )
    parser.add_argument(
        "--use-translation-x-block",
        "--use_translation_x_block",
        "--quspin-use-translation-x-block",
        "--quspin_use_translation_x_block",
        dest="use_translation_x_block",
        action=argparse.BooleanOptionalAction,
        default=USE_TRANSLATION_X_BLOCK,
        help="Request the x/T1 unit-cell momentum block for methods/backends that support it.",
    )
    parser.add_argument(
        "--use-translation-y-block",
        "--use_translation_y_block",
        "--quspin-use-translation-y-block",
        "--quspin_use_translation_y_block",
        dest="use_translation_y_block",
        action=argparse.BooleanOptionalAction,
        default=USE_TRANSLATION_Y_BLOCK,
        help="Request the y/T2 unit-cell momentum block for methods/backends that support it.",
    )
    parser.add_argument(
        "--momentum-x-block",
        "--momentum_x_block",
        "--quspin-momentum-x-block",
        "--quspin_momentum_x_block",
        dest="momentum_x_block",
        type=int,
        default=MOMENTUM_X_BLOCK,
        help="Momentum sector index for the x/T1 translation block.",
    )
    parser.add_argument(
        "--momentum-y-block",
        "--momentum_y_block",
        "--quspin-momentum-y-block",
        "--quspin_momentum_y_block",
        dest="momentum_y_block",
        type=int,
        default=MOMENTUM_Y_BLOCK,
        help="Momentum sector index for the y/T2 translation block.",
    )
    parser.add_argument(
        "--use-reflection-block",
        "--use_reflection_block",
        "--quspin-use-reflection-block",
        "--quspin_use_reflection_block",
        dest="use_reflection_block",
        action=argparse.BooleanOptionalAction,
        default=USE_REFLECTION_BLOCK,
        help="Request a spatial reflection block for methods/backends that support it.",
    )
    parser.add_argument(
        "--reflection-block",
        "--reflection_block",
        "--quspin-reflection-block",
        "--quspin_reflection_block",
        dest="reflection_block",
        type=int,
        choices=list(REFLECTION_BLOCK_OPTIONS),
        default=REFLECTION_BLOCK,
        help="Reflection sector: 0 disables, +1 even, -1 odd.",
    )
    quspin_group = parser.add_argument_group("QuSpin ED Settings")
    quspin_group.add_argument(
        "--quspin-check-symmetries",
        "--quspin_check_symmetries",
        dest="quspin_check_symmetries",
        action=argparse.BooleanOptionalAction,
        default=QUSPIN_CHECK_SYMMETRIES,
        help="Ask QuSpin to check operator symmetry consistency when constructing the Hamiltonian.",
    )
    quspin_group.add_argument(
        "--quspin-check-hermiticity",
        "--quspin_check_hermiticity",
        dest="quspin_check_hermiticity",
        action=argparse.BooleanOptionalAction,
        default=QUSPIN_CHECK_HERMITICITY,
        help="Ask QuSpin to check Hamiltonian Hermiticity.",
    )
    quspin_group.add_argument(
        "--quspin-check-particle-conservation",
        "--quspin_check_particle_conservation",
        dest="quspin_check_particle_conservation",
        action=argparse.BooleanOptionalAction,
        default=QUSPIN_CHECK_PARTICLE_CONSERVATION,
        help=(
            "Ask QuSpin to run its particle-conservation check. This is usually false for "
            "the spin-orbital Yao-Lee basis unless the QuSpin backend maps the model to particles."
        ),
    )
    parser.add_argument(
        "--run-finite-temperature",
        "--run_finite_temperature",
        dest="run_finite_temperature",
        action=argparse.BooleanOptionalAction,
        default=RUN_FINITE_TEMPERATURE_ED,
        help="Run finite-temperature ED observables/correlations versus T within the thermal ED safety caps.",
    )
    parser.add_argument(
        "--temperature-min",
        "--temperature_min",
        dest="temperature_min",
        type=float,
        default=TEMPERATURE_MIN,
        help="Minimum positive temperature for finite-temperature ED plots.",
    )
    parser.add_argument(
        "--temperature-max",
        "--temperature_max",
        dest="temperature_max",
        type=float,
        default=TEMPERATURE_MAX,
        help="Maximum temperature for finite-temperature ED plots.",
    )
    parser.add_argument(
        "--temperature-points",
        "--temperature_points",
        dest="temperature_points",
        type=int,
        default=TEMPERATURE_POINTS,
        help="Number of T samples for finite-temperature ED plots.",
    )
    parser.add_argument(
        "--temperature-scale",
        "--temperature_scale",
        dest="temperature_scale",
        type=str,
        choices=["linear", "log"],
        default=TEMPERATURE_SCALE,
        help="Temperature grid spacing for finite-temperature ED plots.",
    )
    parser.add_argument(
        "--thermal-max-sites",
        "--thermal_max_sites",
        dest="thermal_max_sites",
        type=int,
        default=MAX_THERMAL_ED_SITES,
        help="Site cap for finite-temperature ED.",
    )
    parser.add_argument(
        "--thermal-max-hilbert-dim",
        "--thermal_max_hilbert_dim",
        dest="thermal_max_hilbert_dim",
        type=int,
        default=MAX_THERMAL_ED_HILBERT_DIM,
        help="Hilbert-space dimension cap for finite-temperature ED.",
    )
    parser.add_argument(
        "--thermal-full-spectrum-max-dim",
        "--thermal_full_spectrum_max_dim",
        dest="thermal_full_spectrum_max_dim",
        type=int,
        default=THERMAL_FULL_SPECTRUM_MAX_DIM,
        help="Use exact full-spectrum thermal ED up to this Hilbert-space dimension.",
    )
    parser.add_argument(
        "--thermal-max-eigenstates",
        "--thermal_max_eigenstates",
        dest="thermal_max_eigenstates",
        type=int,
        default=THERMAL_MAX_EIGENSTATES,
        help="Number of low-energy eigenstates used when full-spectrum thermal ED is too large.",
    )
    parser.add_argument(
        "--check-ground-state-degeneracy",
        "--check_ground_state_degeneracy",
        dest="check_ground_state_degeneracy",
        action=argparse.BooleanOptionalAction,
        default=CHECK_GROUND_STATE_DEGENERACY,
        help=(
            "Resolve ED ground-state degeneracy for the low-energy spectrum comparison. "
            "If disabled, the comparison uses the raw second ED eigenvalue and labels degeneracy as unchecked."
        ),
    )
    parser.add_argument(
        "--ed-ground-manifold-abs-tol",
        "--ed_ground_manifold_abs_tol",
        dest="ed_ground_manifold_abs_tol",
        type=float,
        default=ED_GROUND_MANIFOLD_ABS_TOL,
        help="Absolute tolerance for grouping ED levels into the ground-state manifold.",
    )
    parser.add_argument(
        "--ed-ground-manifold-rel-tol",
        "--ed_ground_manifold_rel_tol",
        dest="ed_ground_manifold_rel_tol",
        type=float,
        default=ED_GROUND_MANIFOLD_REL_TOL,
        help=(
            "Relative tolerance for grouping ED levels into the ground-state manifold; "
            "prevents tiny numerical splittings from becoming the plotted ED gap."
        ),
    )
    parser.add_argument(
        "--dmrg-excited-overlap-tol",
        "--dmrg_excited_overlap_tol",
        dest="dmrg_excited_overlap_tol",
        type=float,
        default=DMRG_EXCITED_OVERLAP_TOL,
        help="Full-MPS overlap tolerance for accepting a finite-DMRG penalty excited state.",
    )
    parser.add_argument(
        "--dmrg-excited-energy-tol",
        "--dmrg_excited_energy_tol",
        dest="dmrg_excited_energy_tol",
        type=float,
        default=DMRG_EXCITED_ENERGY_TOL,
        help="Minimum energy separation above the DMRG ground state for accepting an excited state.",
    )
    parser.add_argument(
        "--dmrg-excited-variance-tol",
        "--dmrg_excited_variance_tol",
        dest="dmrg_excited_variance_tol",
        type=float,
        default=DMRG_EXCITED_VARIANCE_TOL,
        help="Original-Hamiltonian variance tolerance for accepting a DMRG excited state.",
    )
    parser.add_argument(
        "--dmrg-excited-max-attempts",
        "--dmrg_excited_max_attempts",
        dest="dmrg_excited_max_attempts",
        type=int,
        default=DMRG_EXCITED_MAX_ATTEMPTS,
        help="Maximum penalty-DMRG attempts per penalty weight.",
    )
    parser.add_argument(
        "--run-idmrg",
        "--run_idmrg",
        dest="run_idmrg",
        action=argparse.BooleanOptionalAction,
        default=RUN_IDMRG,
        help="Run an iDMRG-x ground-state workflow and compare with finite DMRG/ED.",
    )
    parser.add_argument(
        "--idmrg-max-iterations",
        "--idmrg_max_iterations",
        dest="idmrg_max_iterations",
        type=int,
        default=IDMRG_MAX_ITERATIONS,
        help="Maximum iDMRG iterations.",
    )
    parser.add_argument(
        "--idmrg-max-bond-dimension",
        "--idmrg_max_bond_dimension",
        "--idmrg-final-max-bond-dimension",
        "--idmrg_final_max_bond_dimension",
        dest="idmrg_max_bond_dimension",
        type=int,
        default=IDMRG_MAX_BOND_DIMENSION,
        help="iDMRG final maximum bond dimension after warmup, independent of finite-DMRG chi.",
    )
    parser.add_argument(
        "--idmrg-svd-min",
        "--idmrg_svd_min",
        dest="idmrg_svd_min",
        type=float,
        default=IDMRG_SVD_MIN,
        help=(
            "iDMRG SVD singular-value truncation threshold passed as TeNPy "
            "trunc_params['svd_min']; set 0 to disable this floor."
        ),
    )
    parser.add_argument(
        "--idmrg-max-local-dim",
        "--idmrg_max_local_dim",
        dest="idmrg_max_local_dim",
        type=int,
        default=IDMRG_MAX_LOCAL_DIM,
        help="Safety limit for effective local dimension used by iDMRG bulk mapping.",
    )
    parser.add_argument(
        "--idmrg-bulk-kind",
        "--idmrg_bulk_kind",
        dest="idmrg_bulk_kind",
        type=str,
        choices=list(IDMRG_BULK_KIND_OPTIONS),
        default=IDMRG_BULK_KIND,
        help="How to extract iDMRG bulk MPO from finite MPO.",
    )
    parser.add_argument(
        "--idmrg-use-translation-symmetry",
        "--idmrg_use_translation_symmetry",
        dest="idmrg_use_translation_symmetry",
        action=argparse.BooleanOptionalAction,
        default=IDMRG_USE_TRANSLATION_SYMMETRY,
        help="Use the infinite translated MPS unit cell in iDMRG; disable to skip iDMRG translation benchmarks.",
    )
    parser.add_argument(
        "--max-ipeps-unit-cell-sites",
        "--max_ipeps_unit_cell_sites",
        dest="max_ipeps_unit_cell_sites",
        type=int,
        default=MAX_IPEPS_UNIT_CELL_SITES,
        help="iPEPS unit-cell site safety cap, independent of finite-DMRG and iDMRG caps.",
    )
    parser.add_argument(
        "--ipeps-max-bond-dimension",
        "--ipeps_max_bond_dimension",
        dest="ipeps_max_bond_dimension",
        type=int,
        default=IPEPS_MAX_BOND_DIMENSION,
        help="iPEPS virtual bond dimension D, independent of iDMRG chi.",
    )
    parser.add_argument(
        "--ipeps-bond-dimension-cap",
        "--ipeps_bond_dimension_cap",
        dest="ipeps_bond_dimension_cap",
        type=int,
        default=IPEPS_BOND_DIMENSION_CAP,
        help="Profile safety cap for iPEPS virtual bond dimension.",
    )
    parser.add_argument(
        "--ipeps-max-iterations",
        "--ipeps_max_iterations",
        dest="ipeps_max_iterations",
        type=int,
        default=IPEPS_MAX_ITERATIONS,
        help="iPEPS Simple Update iterations, independent of iDMRG iterations.",
    )
    parser.add_argument(
        "--ipeps-iteration-cap",
        "--ipeps_iteration_cap",
        dest="ipeps_iteration_cap",
        type=int,
        default=IPEPS_ITERATION_CAP,
        help="Profile safety cap for iPEPS Simple Update iterations.",
    )
    parser.add_argument(
        "--ipeps-ctm-chi",
        "--ipeps_ctm_chi",
        dest="ipeps_ctm_chi",
        type=int,
        default=IPEPS_CTM_CHI,
        help="iPEPS CTMRG/boundary contraction chi.",
    )
    parser.add_argument(
        "--ipeps-ctm-chi-cap",
        "--ipeps_ctm_chi_cap",
        dest="ipeps_ctm_chi_cap",
        type=int,
        default=IPEPS_CTM_CHI_CAP,
        help="Profile safety cap for iPEPS CTMRG chi.",
    )
    parser.add_argument(
        "--ipeps-tau",
        "--ipeps_tau",
        dest="ipeps_tau",
        type=float,
        default=IPEPS_TAU,
        help="iPEPS Simple Update imaginary-time step.",
    )
    parser.add_argument(
        "--ipeps-symmetry-mode",
        "--ipeps_symmetry_mode",
        dest="ipeps_symmetry_mode",
        type=str,
        choices=list(IPEPS_SYMMETRY_MODE_OPTIONS),
        default=IPEPS_SYMMETRY_MODE,
        help="iPEPS tensor symmetry request: auto, none, u1_tz, or u1_tz_z2.",
    )
    parser.add_argument(
        "--ipeps-strict-symmetry",
        "--ipeps_strict_symmetry",
        dest="ipeps_strict_symmetry",
        action=argparse.BooleanOptionalAction,
        default=IPEPS_STRICT_SYMMETRY,
        help="Raise when a requested iPEPS spin-sector Z2 tensor symmetry is unsupported.",
    )
    parser.add_argument(
        "--ipeps-allow-dense-fallback",
        "--ipeps_allow_dense_fallback",
        dest="ipeps_allow_dense_fallback",
        action=argparse.BooleanOptionalAction,
        default=IPEPS_ALLOW_DENSE_FALLBACK,
        help="Allow iPEPS to run dense when requested tensor symmetries are unsupported.",
    )
    parser.add_argument(
        "--ipeps-unit-cell-kind",
        "--ipeps_unit_cell_kind",
        dest="ipeps_unit_cell_kind",
        type=str,
        choices=list(IPEPS_UNIT_CELL_KIND_OPTIONS),
        default=IPEPS_UNIT_CELL_KIND,
        help="iPEPS variational unit-cell ansatz label, separate from internal tensor symmetry.",
    )
    parser.add_argument(
        "--ipeps-use-translation-symmetry",
        "--ipeps_use_translation_symmetry",
        dest="ipeps_use_translation_symmetry",
        action=argparse.BooleanOptionalAction,
        default=IPEPS_USE_TRANSLATION_SYMMETRY,
        help="Use a repeated translated iPEPS unit cell; disable to skip iPEPS and use finite PEPS instead.",
    )
    parser.add_argument(
        "--ipeps-contraction-method",
        "--ipeps_contraction_method",
        "--ipeps-ctm-method",
        "--ipeps_ctm_method",
        "--ipeps-crtg-method",
        "--ipeps_crtg_method",
        dest="ipeps_contraction_method",
        type=str,
        choices=list(IPEPS_CONTRACTION_METHOD_OPTIONS),
        default=IPEPS_CONTRACTION_METHOD,
        help="iPEPS environment contraction option: ctmrg/crtg or boundary.",
    )
    parser.add_argument(
        "--phase-diagram",
        "--phase_diagram",
        dest="phase_diagram",
        action=argparse.BooleanOptionalAction,
        default=PHASE_DIAGRAM_ENABLED,
        help="Combined switch: when enabled, always run, plot, and save the selected phase scans.",
    )
    parser.add_argument(
        "--run-phase-scan",
        "--run_phase_scan",
        dest="run_phase_scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy calculation-only switch. If omitted, --phase-diagram controls it.",
    )
    parser.add_argument(
        "--phase-scan-only",
        "--phase_scan_only",
        dest="phase_scan_only",
        action=argparse.BooleanOptionalAction,
        default=PHASE_SCAN_ONLY,
        help=(
            "Run only the alpha-beta phase scan and skip the single-point workflow; "
            "this automatically enables phase-diagram plotting/saving."
        ),
    )
    parser.add_argument(
        "--phase-scan-methods",
        "--phase_scan_methods",
        "--phase-scan-quantum-methods",
        "--phase_scan_quantum_methods",
        dest="phase_scan_methods",
        type=str,
        default=None,
        help=(
            "Comma-separated quantum phase-scan methods: ed, dmrg, idmrg, peps, ipeps, or all. "
            "PEPS/iPEPS are explicit quimb methods and are not inferred from --backend."
        ),
    )
    parser.add_argument(
        "--phase-scan-mode",
        "--phase_scan_mode",
        dest="phase_scan_mode",
        type=str,
        choices=("quantum", "classical", "both"),
        default=PHASE_SCAN_MODE,
        help="High-level phase scan content: quantum, classical, or both.",
    )
    parser.add_argument(
        "--phase-scan-channels",
        "--phase_scan_channels",
        dest="phase_scan_channels",
        type=str,
        choices=list(PHASE_SCAN_CHANNEL_OPTIONS),
        default=PHASE_SCAN_CHANNELS,
        help=(
            "Choose which phase-scan channel(s) to execute: auto, none, normal, external, or both. "
            "normal is the alpha-beta scan; external is controlled by --external-scan-mode."
        ),
    )
    parser.add_argument("--phase-scan-alpha-min", "--phase_scan_alpha_min", dest="phase_scan_alpha_min", type=float, default=PHASE_SCAN_ALPHA_MIN)
    parser.add_argument("--phase-scan-alpha-max", "--phase_scan_alpha_max", dest="phase_scan_alpha_max", type=float, default=PHASE_SCAN_ALPHA_MAX)
    parser.add_argument("--phase-scan-alpha-points", "--phase_scan_alpha_points", dest="phase_scan_alpha_points", type=int, default=PHASE_SCAN_ALPHA_POINTS)
    parser.add_argument("--phase-scan-beta-min", "--phase_scan_beta_min", dest="phase_scan_beta_min", type=float, default=PHASE_SCAN_BETA_MIN)
    parser.add_argument("--phase-scan-beta-max", "--phase_scan_beta_max", dest="phase_scan_beta_max", type=float, default=PHASE_SCAN_BETA_MAX)
    parser.add_argument("--phase-scan-beta-points", "--phase_scan_beta_points", dest="phase_scan_beta_points", type=int, default=PHASE_SCAN_BETA_POINTS)
    parser.add_argument(
        "--external-scan-mode",
        "--external_scan_mode",
        dest="external_scan_mode",
        type=str,
        choices=list(EXTERNAL_SCAN_MODE_OPTIONS),
        default=EXTERNAL_SCAN_MODE,
        help=(
            "When phase scanning and an external field are active, choose the field scan: "
            f"{', '.join(EXTERNAL_SCAN_MODE_OPTIONS)}. "
            "e_b overlays DMRG ground energy with ED low-energy bands versus |H|; "
            "alpha_b_* scans alpha versus |H| using classical/quantum/both/all levels."
        ),
    )
    parser.add_argument(
        "--external-scan-field-min",
        "--external_scan_field_min",
        dest="external_scan_field_min",
        type=float,
        default=EXTERNAL_SCAN_FIELD_MIN,
        help="Minimum external-field strength B for external phase scans.",
    )
    parser.add_argument(
        "--external-scan-field-max",
        "--external_scan_field_max",
        dest="external_scan_field_max",
        type=float,
        default=EXTERNAL_SCAN_FIELD_MAX,
        help="Maximum external-field strength B for external phase scans.",
    )
    parser.add_argument(
        "--external-scan-field-points",
        "--external_scan_field_points",
        dest="external_scan_field_points",
        type=int,
        default=EXTERNAL_SCAN_FIELD_POINTS,
        help="Number of external-field B samples for external phase scans.",
    )
    parser.add_argument(
        "--external-scan-ed-bands",
        "--external_scan_ed_bands",
        dest="external_scan_ed_bands",
        type=int,
        default=EXTERNAL_SCAN_ED_BANDS,
        help="Number of lowest ED bands to overlay in external_scan_mode=e_b.",
    )
    parser.add_argument(
        "--phase-scan-ed-max-sites",
        "--phase_scan_ed_max_sites",
        dest="phase_scan_ed_max_sites",
        type=int,
        default=None,
        help="Site cap for quantum ED phase scans. Omit to reuse --max-ed-sites.",
    )
    parser.add_argument(
        "--phase-scan-ed-max-hilbert-dim",
        "--phase_scan_ed_max_hilbert_dim",
        dest="phase_scan_ed_max_hilbert_dim",
        type=int,
        default=None,
        help="Hilbert-space dimension cap for quantum ED phase scans. Omit to reuse --max-ed-hilbert-dim.",
    )
    parser.add_argument("--phase-scan-classical-restarts", "--phase_scan_classical_restarts", dest="phase_scan_classical_restarts", type=int, default=PHASE_SCAN_CLASSICAL_RESTARTS)
    parser.add_argument("--phase-scan-classical-sweeps", "--phase_scan_classical_sweeps", dest="phase_scan_classical_sweeps", type=int, default=PHASE_SCAN_CLASSICAL_SWEEPS)
    parser.add_argument("--phase-scan-classical-initial-temperature", "--phase_scan_classical_initial_temperature", dest="phase_scan_classical_initial_temperature", type=float, default=PHASE_SCAN_CLASSICAL_INITIAL_TEMPERATURE)
    parser.add_argument("--phase-scan-classical-final-temperature", "--phase_scan_classical_final_temperature", dest="phase_scan_classical_final_temperature", type=float, default=PHASE_SCAN_CLASSICAL_FINAL_TEMPERATURE)
    parser.add_argument("--phase-scan-classical-initial-step", "--phase_scan_classical_initial_step", dest="phase_scan_classical_initial_step", type=float, default=PHASE_SCAN_CLASSICAL_INITIAL_STEP)
    parser.add_argument("--phase-scan-classical-final-step", "--phase_scan_classical_final_step", dest="phase_scan_classical_final_step", type=float, default=PHASE_SCAN_CLASSICAL_FINAL_STEP)
    parser.add_argument("--phase-scan-random-seed", "--phase_scan_random_seed", dest="phase_scan_random_seed", type=int, default=PHASE_SCAN_RANDOM_SEED)
    parser.add_argument("--phase-scan-quantum-weak-order-threshold", "--phase_scan_quantum_weak_order_threshold", dest="phase_scan_quantum_weak_order_threshold", type=float, default=PHASE_SCAN_QUANTUM_WEAK_ORDER_THRESHOLD)
    parser.add_argument("--phase-scan-classical-weak-order-threshold", "--phase_scan_classical_weak_order_threshold", dest="phase_scan_classical_weak_order_threshold", type=float, default=PHASE_SCAN_CLASSICAL_WEAK_ORDER_THRESHOLD)
    parser.add_argument("--phase-scan-quantum-nematicity-threshold", "--phase_scan_quantum_nematicity_threshold", dest="phase_scan_quantum_nematicity_threshold", type=float, default=PHASE_SCAN_QUANTUM_NEMATICITY_THRESHOLD)
    parser.add_argument("--phase-scan-classical-nematicity-threshold", "--phase_scan_classical_nematicity_threshold", dest="phase_scan_classical_nematicity_threshold", type=float, default=PHASE_SCAN_CLASSICAL_NEMATICITY_THRESHOLD)
    parser.add_argument(
        "--phase-scan-plaquette-flux-target",
        "--phase_scan_plaquette_flux_target",
        dest="phase_scan_plaquette_flux_target",
        type=float,
        default=PHASE_SCAN_PLAQUETTE_FLUX_TARGET,
        help="Normalized W_p target used to identify the Spin-Orbital Liquid phase.",
    )
    parser.add_argument(
        "--phase-scan-plaquette-flux-tolerance",
        "--phase_scan_plaquette_flux_tolerance",
        dest="phase_scan_plaquette_flux_tolerance",
        type=float,
        default=PHASE_SCAN_PLAQUETTE_FLUX_TOLERANCE,
        help="Tolerance for |W_p| being near the conserved plaquette-flux value.",
    )
    parser.add_argument("--output-folder", "--output_folder", dest="output_folder", type=str, default=OUTPUT_FOLDER)
    profile_group = parser.add_argument_group("Profiling")
    profile_group.add_argument(
        "--profile-enabled",
        "--profile_enabled",
        dest="profile_enabled",
        action=argparse.BooleanOptionalAction,
        default=PROFILE_ENABLED,
        help="Enable lightweight standard-library profiling output.",
    )
    profile_group.add_argument(
        "--profile-timing",
        "--profile_timing",
        dest="profile_timing",
        action=argparse.BooleanOptionalAction,
        default=PROFILE_TIMING,
        help="Record wall-clock stage timing when profiling is enabled.",
    )
    profile_group.add_argument(
        "--profile-memory",
        "--profile_memory",
        dest="profile_memory",
        action=argparse.BooleanOptionalAction,
        default=PROFILE_MEMORY,
        help="Record tracemalloc/resource memory metadata when profiling is enabled.",
    )
    profile_group.add_argument(
        "--profile-cprofile",
        "--profile_cprofile",
        dest="profile_cprofile",
        action=argparse.BooleanOptionalAction,
        default=PROFILE_CPROFILE,
        help="Collect optional cProfile cumulative stats when profiling is enabled.",
    )
    profile_group.add_argument(
        "--profile-line-hooks",
        "--profile_line_hooks",
        dest="profile_line_hooks",
        action=argparse.BooleanOptionalAction,
        default=PROFILE_LINE_HOOKS,
        help="Use an existing builtins @profile hook if the process was launched under line_profiler.",
    )
    profile_group.add_argument(
        "--profile-scan-points",
        "--profile_scan_points",
        dest="profile_scan_points",
        action=argparse.BooleanOptionalAction,
        default=PROFILE_SCAN_POINTS,
        help="Record per phase-scan point wall time when profiling is enabled.",
    )
    profile_group.add_argument(
        "--profile-output-json",
        "--profile_output_json",
        dest="profile_output_json",
        action=argparse.BooleanOptionalAction,
        default=PROFILE_OUTPUT_JSON,
        help="Write outputs/profiling/profile_summary.json when profiling is enabled.",
    )
    profile_group.add_argument(
        "--profile-output-folder",
        "--profile_output_folder",
        dest="profile_output_folder",
        type=str,
        default=PROFILE_OUTPUT_FOLDER,
        help="Folder for profile_summary.json when profiling output is enabled.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(BACKEND_OPTIONS),
        default=BACKEND,
        help=(
            "Select the DMRG/tensor-network backend. auto uses the local TeNPy "
            "Yao-Lee path first when compatible, then falls back to Tenax. quimb uses quimb.tensor "
            "PEPS for finite calculations and iPEPS for infinite calculations. "
            "Use --ed-backend for ED."
        ),
    )
    parser.add_argument(
        "--method",
        "--calculation-method",
        "--calculation_method",
        dest="method",
        type=str,
        choices=list(CALCULATION_METHOD_OPTIONS),
        default=METHOD,
        help=(
            "Primary calculation method. With --backend quimb, choose peps/quimb_peps "
            "for finite PEPS or ipeps/quimb_ipeps for infinite iPEPS. auto preserves "
            "the default finite method for the selected backend."
        ),
    )
    parser.add_argument(
        "--overwrite-plots",
        action="store_true",
        default=OVERWRITE_EXISTING_PLOTS,
        help="Regenerate PNG outputs even if files already exist.",
    )
    parser.add_argument(
        "--calculate-correlations",
        "--calculate_correlations",
        dest="calculate_correlations",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_CORRELATIONS,
        help="Compute two-point correlation matrices for DMRG/ED post-processing.",
    )
    parser.add_argument(
        "--calculate-bond-energies",
        "--calculate_bond_energies",
        dest="calculate_bond_energies",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_BOND_ENERGIES,
        help="Compute bond-energy rows from correlation matrices.",
    )
    parser.add_argument(
        "--calculate-structure-factors",
        "--calculate_structure_factors",
        dest="calculate_structure_factors",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_STRUCTURE_FACTORS,
        help="Compute high-symmetry structure-factor rows from scalar correlations.",
    )
    parser.add_argument(
        "--calculate-entanglement",
        "--calculate_entanglement",
        dest="calculate_entanglement",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_ENTANGLEMENT,
        help="Compute entanglement entropy profiles when the backend/state representation supports it.",
    )
    parser.add_argument(
        "--calculate-uniform-observables",
        "--calculate_uniform_observables",
        dest="calculate_uniform_observables",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_UNIFORM_OBSERVABLES,
        help="Compute simple uniform one-point observables such as spin_z_per_site when supported.",
    )
    parser.add_argument(
        "--calculate-real-space-patterns",
        "--calculate_real_space_patterns",
        dest="calculate_real_space_patterns",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_REAL_SPACE_PATTERNS,
        help="Extract compact spin/orbital correlation rows for real-space order-pattern diagrams.",
    )
    parser.add_argument(
        "--reference-site-idx",
        "--reference_site_idx",
        dest="reference_site_idx",
        type=int,
        default=REFERENCE_SITE_IDX,
        help="Reference site for real-space pattern plots. Omit to choose the geometric center site.",
    )
    parser.add_argument(
        "--plot-geometry",
        "--plot_geometry",
        dest="plot_geometry",
        action=argparse.BooleanOptionalAction,
        default=PLOT_GEOMETRY,
        help="Save the lattice geometry diagram.",
    )
    parser.add_argument(
        "--plot-bond-energies",
        "--plot_bond_energies",
        dest="plot_bond_energies",
        action=argparse.BooleanOptionalAction,
        default=PLOT_BOND_ENERGIES,
        help="Save DMRG/ED bond-energy diagrams when bond energies were computed.",
    )
    parser.add_argument(
        "--plot-structure-factors",
        "--plot_structure_factors",
        dest="plot_structure_factors",
        action=argparse.BooleanOptionalAction,
        default=PLOT_STRUCTURE_FACTORS,
        help="Save structure-factor plots when structure factors were computed.",
    )
    parser.add_argument(
        "--plot-correlation-heatmaps",
        "--plot_correlation_heatmaps",
        dest="plot_correlation_heatmaps",
        action=argparse.BooleanOptionalAction,
        default=PLOT_CORRELATION_HEATMAPS,
        help="Save scalar-correlation heatmaps when scalar correlations were computed.",
    )
    parser.add_argument(
        "--plot-real-space-patterns",
        "--plot_real_space_patterns",
        dest="plot_real_space_patterns",
        action=argparse.BooleanOptionalAction,
        default=PLOT_REAL_SPACE_PATTERNS,
        help="Save compact real-space pattern diagrams where supported.",
    )
    parser.add_argument(
        "--plot-entanglement",
        "--plot_entanglement",
        dest="plot_entanglement",
        action=argparse.BooleanOptionalAction,
        default=PLOT_ENTANGLEMENT,
        help="Save entanglement entropy comparison plots when entropy profiles were computed.",
    )
    parser.add_argument(
        "--plot-energy-comparison",
        "--plot_energy_comparison",
        dest="plot_energy_comparison",
        action=argparse.BooleanOptionalAction,
        default=PLOT_ENERGY_COMPARISON,
        help="Save DMRG/ED/iDMRG energy comparison plots.",
    )
    parser.add_argument(
        "--plot-low-energy-spectrum",
        "--plot_low_energy_spectrum",
        dest="plot_low_energy_spectrum",
        action=argparse.BooleanOptionalAction,
        default=PLOT_LOW_ENERGY_SPECTRUM,
        help="Save the low-energy spectrum comparison plot.",
    )
    parser.add_argument(
        "--plot-finite-temperature",
        "--plot_finite_temperature",
        dest="plot_finite_temperature",
        action=argparse.BooleanOptionalAction,
        default=PLOT_FINITE_TEMPERATURE,
        help="Save finite-temperature ED plots when finite-temperature ED was run.",
    )
    parser.add_argument(
        "--plot-phase-scan",
        "--plot_phase_scan",
        dest="plot_phase_scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy plot-only switch. If omitted, --phase-diagram controls it.",
    )
    parser.add_argument(
        "--strict-plot-errors",
        action="store_true",
        default=STRICT_PLOT_ERRORS,
        help="Stop immediately when one plot save fails.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help="Show tqdm progress bars and stage timing logs.",
    )
    args = parser.parse_args()
    _normalize_geometry_cli_args(args)
    return args


def to_json_compatible(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_json_compatible(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_compatible(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
        return float(obj)
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    return obj


def write_json(filepath: str, data_dict: Dict[str, Any]) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(to_json_compatible(data_dict), file, indent=2, sort_keys=True)


def _save_summary_checkpoint(output_folder: str, summary: Dict[str, Any]) -> None:
    outputs = summary.get("outputs", {})
    summary_filename = "run_summary.json"
    if isinstance(outputs, dict):
        summary_filename = str(outputs.get("run_summary_json", summary_filename))

    run_summary_path = os.path.join(output_folder, summary_filename)
    write_json(run_summary_path, summary)


def _load_previous_summary(output_folder: str) -> Dict[str, Any] | None:
    candidate_paths = [os.path.join(output_folder, "run_summary.json")]
    try:
        if os.path.isdir(output_folder):
            labeled_summaries = [
                os.path.join(output_folder, filename)
                for filename in os.listdir(output_folder)
                if filename.endswith("_run_summary.json")
            ]
            labeled_summaries.sort(key=lambda path: os.path.getmtime(path), reverse=True)
            candidate_paths.extend(labeled_summaries)
    except Exception:
        pass

    for run_summary_path in candidate_paths:
        if not os.path.exists(run_summary_path):
            continue
        try:
            with open(run_summary_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _extract_consistency_signature(parameters: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "lattice",
        "length_x",
        "length_y",
        "circumference_x",
        "circumference_y",
        "alpha",
        "beta",
        "coupling_j",
        "jx",
        "jy",
        "jz",
        "external_field_treatment",
        "external_field_axis",
        "external_field_strength",
        "field_hx",
        "field_hy",
        "field_hz",
        "mu_b",
        "field_sign",
        "field_sigma_factor",
        "spin_rep",
        "orbital_rep",
        "model_family",
        "ising_axis",
        "symmetry_reductions",
        "symmetry_mode",
        "u1_target_sz2",
        "u1_target_tz2",
        "z2_target_parity",
        "strict_symmetry_selection_rules",
        "symmetry_precheck",
        "strict_symmetry_precheck",
        "symmetry_allow_dense_fallback",
        "backend",
        "method",
        "max_dmrg_sites",
        "max_bond_dimension",
        "max_sweeps",
        "max_peps_sites",
        "peps_max_bond_dimension",
        "peps_bond_dimension_cap",
        "peps_max_sweeps",
        "peps_sweep_cap",
        "peps_ctm_chi",
        "peps_ctm_chi_cap",
        "peps_tau",
        "peps_entanglement_max_dense_dim",
        "peps_symmetry_mode",
        "peps_strict_symmetry",
        "peps_allow_dense_fallback",
        "run_idmrg",
        "idmrg_max_bond_dimension",
        "idmrg_max_iterations",
        "idmrg_max_local_dim",
        "idmrg_bulk_kind",
        "idmrg_use_translation_symmetry",
        "max_ipeps_unit_cell_sites",
        "ipeps_max_bond_dimension",
        "ipeps_bond_dimension_cap",
        "ipeps_max_iterations",
        "ipeps_iteration_cap",
        "ipeps_ctm_chi",
        "ipeps_ctm_chi_cap",
        "ipeps_tau",
        "ipeps_symmetry_mode",
        "ipeps_strict_symmetry",
        "ipeps_allow_dense_fallback",
        "ipeps_unit_cell_kind",
        "ipeps_use_translation_symmetry",
        "ipeps_contraction_method",
        "seed",
        "run_ed",
        "max_ed_sites",
        "max_ed_hilbert_dim",
        "ed_max_eigenstates",
        "ed_backend",
        "ed_solver",
        "ed_symmetry_engine",
        "ed_quspin_experimental_fused_translation",
        "ed_c3_mode",
        "ed_c3_q_blocks",
        "ed_z2_mode",
        "ed_z2_kind",
        "ed_sparse_tol",
        "ed_sparse_maxiter",
        "use_translation_x_block",
        "use_translation_y_block",
        "momentum_x_block",
        "momentum_y_block",
        "use_reflection_block",
        "reflection_block",
        "quspin_check_symmetries",
        "quspin_check_hermiticity",
        "quspin_check_particle_conservation",
        "run_finite_temperature",
        "temperature_min",
        "temperature_max",
        "temperature_points",
        "temperature_scale",
        "thermal_max_sites",
        "thermal_max_hilbert_dim",
        "thermal_full_spectrum_max_dim",
        "thermal_max_eigenstates",
        "check_ground_state_degeneracy",
        "ed_ground_manifold_abs_tol",
        "ed_ground_manifold_rel_tol",
        "dmrg_excited_overlap_tol",
        "dmrg_excited_energy_tol",
        "dmrg_excited_variance_tol",
        "dmrg_excited_max_attempts",
        "phase_diagram",
        "run_phase_scan",
        "phase_scan_only",
        "phase_scan_methods",
        "phase_scan_mode",
        "phase_scan_alpha_min",
        "phase_scan_alpha_max",
        "phase_scan_alpha_points",
        "phase_scan_beta_min",
        "phase_scan_beta_max",
        "phase_scan_beta_points",
        "external_scan_mode",
        "external_scan_field_min",
        "external_scan_field_max",
        "external_scan_field_points",
        "external_scan_ed_bands",
        "phase_scan_ed_max_sites",
        "phase_scan_ed_max_hilbert_dim",
        "phase_scan_classical_restarts",
        "phase_scan_classical_sweeps",
        "phase_scan_classical_initial_temperature",
        "phase_scan_classical_final_temperature",
        "phase_scan_classical_initial_step",
        "phase_scan_classical_final_step",
        "phase_scan_random_seed",
        "phase_scan_quantum_weak_order_threshold",
        "phase_scan_classical_weak_order_threshold",
        "phase_scan_quantum_nematicity_threshold",
        "phase_scan_classical_nematicity_threshold",
        "phase_scan_plaquette_flux_target",
        "phase_scan_plaquette_flux_tolerance",
    ]
    return {key: parameters.get(key) for key in keys}


def _finite_float_from_mapping(mapping: Any, *keys: str) -> float | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key not in mapping:
            continue
        try:
            value = float(mapping[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _require_positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}.")
    return parsed


def _require_positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value; got {value!r}.")
    return parsed


def _require_nonnegative_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite value; got {value!r}.")
    return parsed


def _validate_solver_resource_args(args: argparse.Namespace) -> None:
    """Enforce profile-linked PEPS/iPEPS safety caps after CLI parsing."""
    positive_int_names = (
        "max_dmrg_sites",
        "max_bond_dimension",
        "max_sweeps",
        "max_peps_sites",
        "peps_max_bond_dimension",
        "peps_bond_dimension_cap",
        "peps_max_sweeps",
        "peps_sweep_cap",
        "peps_ctm_chi",
        "peps_ctm_chi_cap",
        "peps_entanglement_max_dense_dim",
        "idmrg_max_bond_dimension",
        "idmrg_max_iterations",
        "idmrg_max_local_dim",
        "max_ipeps_unit_cell_sites",
        "ipeps_max_bond_dimension",
        "ipeps_bond_dimension_cap",
        "ipeps_max_iterations",
        "ipeps_iteration_cap",
        "ipeps_ctm_chi",
        "ipeps_ctm_chi_cap",
        "external_scan_field_points",
        "external_scan_ed_bands",
    )
    for name in positive_int_names:
        setattr(args, name, _require_positive_int(getattr(args, name), name))
    args.truncation_cutoff = _require_nonnegative_float(args.truncation_cutoff, "truncation_cutoff")
    args.dmrg_svd_min = _require_nonnegative_float(args.dmrg_svd_min, "dmrg_svd_min")
    args.idmrg_svd_min = _require_nonnegative_float(args.idmrg_svd_min, "idmrg_svd_min")
    args.peps_tau = _require_positive_float(args.peps_tau, "peps_tau")
    args.ipeps_tau = _require_positive_float(args.ipeps_tau, "ipeps_tau")
    args.external_scan_field_min = float(args.external_scan_field_min)
    args.external_scan_field_max = float(args.external_scan_field_max)
    if not np.isfinite(args.external_scan_field_min) or not np.isfinite(args.external_scan_field_max):
        raise ValueError("external_scan_field_min/max must be finite.")

    cap_checks = (
        ("peps_max_bond_dimension", "peps_bond_dimension_cap", "finite PEPS bond dimension"),
        ("peps_max_sweeps", "peps_sweep_cap", "finite PEPS Simple Update sweeps"),
        ("peps_ctm_chi", "peps_ctm_chi_cap", "finite PEPS CTMRG chi"),
        ("ipeps_max_bond_dimension", "ipeps_bond_dimension_cap", "iPEPS bond dimension"),
        ("ipeps_max_iterations", "ipeps_iteration_cap", "iPEPS Simple Update iterations"),
        ("ipeps_ctm_chi", "ipeps_ctm_chi_cap", "iPEPS CTMRG chi"),
    )
    for value_name, cap_name, label in cap_checks:
        value = int(getattr(args, value_name))
        cap = int(getattr(args, cap_name))
        if value > cap:
            raise ValueError(
                f"{label} requested {value}, above profile cap {cap}. "
                f"Increase --{cap_name.replace('_', '-')} deliberately for a larger device."
            )


def _idmrg_energy_is_suspicious(idmrg_energy_per_site: float | None, dmrg_energy_per_site: float) -> bool:
    if idmrg_energy_per_site is None:
        return True
    if not np.isfinite(float(idmrg_energy_per_site)):
        return True
    reference_scale = max(1.0, abs(float(dmrg_energy_per_site)))
    return abs(float(idmrg_energy_per_site) - float(dmrg_energy_per_site)) > 5.0 * reference_scale


def _split_phase_scan_csv(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        item.strip()
        for item in str(value).replace("+", ",").replace(";", ",").split(",")
        if item.strip()
    ]


def _normalize_phase_scan_mode(value: Any) -> str:
    """Normalize the high-level phase-scan content choice."""
    key = str(value if value is not None else PHASE_SCAN_MODE).strip().lower()
    alias_map = {
        "quantum": "quantum",
        "methods": "quantum",
        "quantum_methods": "quantum",
        "quantum_ed": "quantum",
        "ed": "quantum",
        "exact": "quantum",
        "exact_diagonalization": "quantum",
        "tenax_dmrg": "quantum",
        "tenpy_dmrg": "quantum",
        "dmrg": "quantum",
        "finite_dmrg": "quantum",
        "tenax_idmrg": "quantum",
        "tenpy_idmrg": "quantum",
        "idmrg": "quantum",
        "infinite_dmrg": "quantum",
        "peps": "quantum",
        "finite_peps": "quantum",
        "quimb_peps": "quantum",
        "ipeps": "quantum",
        "infinite_peps": "quantum",
        "quimb_ipeps": "quantum",
        "classical": "classical",
        "classical_product": "classical",
        "both": "both",
        "all": "both",
    }
    if key not in alias_map:
        raise ValueError("Unsupported phase-scan mode '{0}'. Choose from: quantum, classical, both.".format(value))
    return alias_map[key]


def _default_quantum_phase_scan_methods_from_legacy_mode(legacy_mode: str | None) -> List[str]:
    key = str(legacy_mode or PHASE_SCAN_MODE).strip().lower()
    if key in ("quantum_ed", "ed", "exact", "exact_diagonalization"):
        return ["ed"]
    if key in ("tenax_dmrg", "tenpy_dmrg", "dmrg", "finite_dmrg"):
        return ["dmrg"]
    if key in ("tenax_idmrg", "tenpy_idmrg", "idmrg", "infinite_dmrg"):
        return ["idmrg"]
    if key in ("peps", "finite_peps", "quimb_peps"):
        return ["peps"]
    if key in ("ipeps", "infinite_peps", "quimb_ipeps"):
        return ["ipeps"]
    if key == "all":
        return list(PHASE_SCAN_QUANTUM_METHOD_OPTIONS)
    return _split_phase_scan_csv(PHASE_SCAN_METHODS)


def _normalize_phase_scan_quantum_methods(
    methods_value: Any,
    legacy_mode: str | None = None,
) -> List[str]:
    """Normalize quantum phase-scan solver choices to ed/dmrg/idmrg/peps/ipeps."""
    alias_map = {
        "quantum_ed": "ed",
        "ed": "ed",
        "exact": "ed",
        "exact_diagonalization": "ed",
        "tenax_dmrg": "dmrg",
        "tenpy_dmrg": "dmrg",
        "dmrg": "dmrg",
        "finite_dmrg": "dmrg",
        "tenax_idmrg": "idmrg",
        "tenpy_idmrg": "idmrg",
        "idmrg": "idmrg",
        "infinite_dmrg": "idmrg",
        "peps": "peps",
        "finite_peps": "peps",
        "quimb_peps": "peps",
        "ipeps": "ipeps",
        "infinite_peps": "ipeps",
        "quimb_ipeps": "ipeps",
    }
    grouped_aliases = {
        "all": list(PHASE_SCAN_QUANTUM_METHOD_OPTIONS),
        "quantum": list(PHASE_SCAN_QUANTUM_METHOD_OPTIONS),
        "methods": _split_phase_scan_csv(PHASE_SCAN_METHODS),
    }
    if methods_value is None or str(methods_value).strip() == "":
        raw_items = _default_quantum_phase_scan_methods_from_legacy_mode(legacy_mode)
    else:
        raw_items = _split_phase_scan_csv(methods_value)

    normalized: List[str] = []
    ignored_classical = False
    for raw_item in raw_items:
        key = str(raw_item).strip().lower()
        if key in ("classical", "classical_product"):
            ignored_classical = True
            continue
        expanded = grouped_aliases.get(key, [key])
        for item in expanded:
            alias_key = str(item).strip().lower()
            if alias_key not in alias_map:
                raise ValueError(
                    f"Unsupported quantum phase-scan method '{raw_item}'. "
                    f"Choose from: {', '.join(PHASE_SCAN_METHOD_OPTIONS)}."
                )
            method = alias_map[alias_key]
            if method not in normalized:
                normalized.append(method)
    if len(normalized) == 0 and not ignored_classical:
        raise ValueError("At least one quantum phase-scan method must be selected.")
    return normalized


def _selected_phase_scan_methods(scan_mode: str, quantum_methods: List[str]) -> List[str]:
    """Return the internal concrete scan outputs requested by the top-level controls."""
    mode = _normalize_phase_scan_mode(scan_mode)
    methods: List[str] = []
    if mode in ("quantum", "both"):
        methods.extend(str(method) for method in quantum_methods)
    if mode in ("classical", "both"):
        methods.append("classical")
    if len(methods) == 0:
        raise ValueError("The phase-scan selection is empty.")
    return methods


def _normalize_phase_scan_methods(
    methods_value: Any,
    legacy_mode: str | None = None,
) -> List[str]:
    """Backward-compatible wrapper returning concrete phase-scan methods."""
    scan_mode = _normalize_phase_scan_mode(legacy_mode)
    quantum_methods = _normalize_phase_scan_quantum_methods(methods_value, legacy_mode)
    return _selected_phase_scan_methods(scan_mode, quantum_methods)


def _phase_scan_legacy_mode_from_methods(methods: List[str]) -> str | None:
    method_set = set(methods)
    if method_set == {"ed"}:
        return "quantum_ed"
    if method_set == {"classical"}:
        return "classical_product"
    if method_set == {"ed", "classical"}:
        return "both"
    return None


def _normalize_ed_backend(value: str | None) -> str:
    text = str(value if value is not None else ED_BACKEND).strip().lower()
    if text in ("standard", "ed", "builtin", "built_in", "scipy"):
        return "standard"
    if text == "quspin":
        return "quspin"
    raise ValueError(f"Unsupported ED backend '{value}'. Choose from: standard, quspin.")


def _normalize_backend(value: str | None) -> str:
    text = str(value if value is not None else BACKEND).strip().lower()
    aliases = {
        "auto": "auto",
        "tenax": "tenax",
        "jax": "tenax",
        "tenpy": "tenpy",
        "quimb": "quimb",
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError(f"Unsupported backend '{value}'. Choose from: {', '.join(BACKEND_OPTIONS)}.")
    return normalized


def _normalize_calculation_method(value: str | None, backend: str | None = None) -> str:
    text = str(value if value is not None else METHOD).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "default": "auto",
        "dmrg": "dmrg",
        "finite_dmrg": "dmrg",
        "idmrg": "idmrg",
        "infinite_dmrg": "idmrg",
        "peps": "peps",
        "finite_peps": "peps",
        "quimb_peps": "peps",
        "ipeps": "ipeps",
        "infinite_peps": "ipeps",
        "quimb_ipeps": "ipeps",
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError(
            f"Unsupported calculation method '{value}'. Choose from: {', '.join(CALCULATION_METHOD_OPTIONS)}."
        )
    backend_key = str(backend or BACKEND).strip().lower()
    if normalized == "auto":
        return "peps" if backend_key == "quimb" else "dmrg"
    if backend_key == "quimb" and normalized not in ("peps", "ipeps"):
        raise ValueError("--backend quimb requires --method peps/quimb_peps or --method ipeps/quimb_ipeps.")
    return normalized


def _normalize_ipeps_contraction_method(value: str | None) -> str:
    text = str(value if value is not None else IPEPS_CONTRACTION_METHOD).strip().lower().replace("-", "_")
    aliases = {
        "auto": "ctmrg",
        "ctm": "ctmrg",
        "ctmrg": "ctmrg",
        "crtg": "ctmrg",
        "boundary": "boundary",
        "boundary_mps": "boundary",
    }
    normalized = aliases.get(text, text)
    if normalized not in ("ctmrg", "boundary"):
        raise ValueError(
            f"Unsupported iPEPS contraction method '{value}'. "
            f"Choose from: {', '.join(IPEPS_CONTRACTION_METHOD_OPTIONS)}."
        )
    return normalized


def _normalize_symmetry_reductions(value: Any, legacy_mode: str | None = None) -> tuple[str, ...]:
    default_items = (
        [str(legacy_mode)]
        if legacy_mode is not None and str(legacy_mode).strip() != ""
        else [str(item) for item in SYMMETRY_REDUCTIONS]
    )
    if value is None:
        raw_items: List[str] = list(default_items)
    elif isinstance(value, (list, tuple, set)):
        raw_items = []
        for item in value:
            raw_items.extend(
                part.strip()
                for part in str(item).replace("+", ",").replace(";", ",").split(",")
                if part.strip()
            )
    else:
        raw_items = [
            item.strip()
            for item in str(value).replace("+", ",").replace(";", ",").split(",")
            if item.strip()
        ]
    if len(raw_items) == 0:
        raw_items = list(default_items)

    aliases = {
        "0": "none",
        "false": "none",
        "off": "none",
        "dense": "none",
        "full": "none",
        "u1": "u1",
        "u1_tz_z2": ("tz", "z2"),
        "u1-tz-z2": ("tz", "z2"),
        "u1tz_z2": ("tz", "z2"),
        "tz_z2": ("tz", "z2"),
        "tz-z2": ("tz", "z2"),
        "tzz2": ("tz", "z2"),
        "u1_sz": "sz",
        "u1-sz": "sz",
        "u1sz": "sz",
        "spin": "sz",
        "spin_z": "sz",
        "s_z": "sz",
        "sz": "sz",
        "u1_tz": "tz",
        "u1-tz": "tz",
        "u1tz": "tz",
        "tau": "tz",
        "tau_z": "tz",
        "t_z": "tz",
        "tz": "tz",
        "parity": "z2",
        "z2": "z2",
        "auto": "auto",
        "none": "none",
    }
    normalized: List[str] = []
    for raw in raw_items:
        key = str(raw).strip().lower()
        mapped = aliases.get(key)
        if mapped is None:
            raise ValueError(
                f"Unsupported symmetry reduction '{raw}'. Choose from: {', '.join(SYMMETRY_REDUCTION_OPTIONS)}."
            )
        if isinstance(mapped, (tuple, list)):
            for item in mapped:
                if item not in normalized:
                    normalized.append(str(item))
        elif mapped == "u1":
            for item in ("sz", "tz"):
                if item not in normalized:
                    normalized.append(item)
        elif mapped in ("auto", "none"):
            return (mapped,)
        elif mapped not in normalized:
            normalized.append(mapped)
    if len(normalized) == 0:
        return ("none",)
    order = {"sz": 0, "tz": 1, "z2": 2}
    return tuple(sorted(normalized, key=lambda item: order[item]))


def _legacy_symmetry_mode_from_reductions(reductions: tuple[str, ...]) -> str:
    reduction_set = set(reductions)
    if "auto" in reduction_set:
        return "auto"
    if len(reduction_set) == 0 or "none" in reduction_set:
        return "none"
    if {"sz", "tz"}.issubset(reduction_set):
        return "u1"
    if "sz" in reduction_set:
        return "u1_sz"
    if "tz" in reduction_set:
        return "u1_tz"
    if "z2" in reduction_set:
        return "z2"
    return "none"


def _u1_report_supports_sector(report: Dict[str, Any] | None, mode: str) -> bool:
    if not isinstance(report, dict):
        return False
    mode_report = report.get(mode, {})
    target_sector = mode_report.get("target_sector", {}) if isinstance(mode_report, dict) else {}
    return bool(isinstance(mode_report, dict) and mode_report.get("conserved", False)) and bool(
        isinstance(target_sector, dict) and target_sector.get("reachable", False)
    )


def _z2_report_supports_sector(report: Dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    z2_report = report.get("z2", {})
    target_sector = z2_report.get("target_sector", {}) if isinstance(z2_report, dict) else {}
    return bool(isinstance(z2_report, dict) and z2_report.get("conserved_global_parity", False)) and bool(
        isinstance(target_sector, dict) and target_sector.get("reachable", False)
    )


def _symmetry_reduction_settings_from_report(
    args: argparse.Namespace,
    report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    reductions = tuple(getattr(args, "symmetry_reductions", ("none",)))
    model_selection = getattr(args, "model_symmetry_selection", None)
    validation_report = None if (isinstance(report, dict) and report.get("status") == "disabled") else report
    if "auto" in reductions:
        use_sz_block = _u1_report_supports_sector(validation_report, "u1") or _u1_report_supports_sector(validation_report, "u1_sz")
        use_tau_z_block = _u1_report_supports_sector(validation_report, "u1") or _u1_report_supports_sector(validation_report, "u1_tz")
        use_z2_block = _z2_report_supports_sector(validation_report)
    else:
        use_sz_block = "sz" in reductions and (
            validation_report is None
            or _u1_report_supports_sector(validation_report, "u1")
            or _u1_report_supports_sector(validation_report, "u1_sz")
        )
        use_tau_z_block = "tz" in reductions and (
            validation_report is None
            or _u1_report_supports_sector(validation_report, "u1")
            or _u1_report_supports_sector(validation_report, "u1_tz")
        )
        use_z2_block = "z2" in reductions and (
            validation_report is None or _z2_report_supports_sector(validation_report)
        )
    z2_generator = None
    accepted_reductions = [] if reductions == ("none",) else list(reductions)
    dropped_reductions: List[str] = []
    backend_support_status: Dict[str, Any] = {}
    if isinstance(model_selection, dict):
        z2_generator = model_selection.get("z2_generator")
        accepted_reductions = list(model_selection.get("accepted_reductions", model_selection.get("effective_reductions", accepted_reductions)))
        dropped_reductions = list(model_selection.get("dropped_reductions", []))
        backend_support_status = dict(model_selection.get("backend_support_status", {}))
        use_sz_block = bool(model_selection.get("use_sz_block", use_sz_block)) and bool(use_sz_block)
        use_tau_z_block = bool(model_selection.get("use_tau_z_block", use_tau_z_block)) and bool(use_tau_z_block)
        use_z2_block = bool(model_selection.get("use_z2_block", use_z2_block)) and bool(use_z2_block)
    if z2_generator is None:
        use_z2_block = False
    return {
        "source": "shared_symmetry_reductions",
        "requested_reductions": list(reductions),
        "accepted_reductions": list(accepted_reductions),
        "dropped_reductions": list(dropped_reductions),
        "model_aware_selection": model_selection,
        "backend_support_status": backend_support_status,
        "legacy_tenax_mode": _legacy_symmetry_mode_from_reductions(reductions),
        "use_sz_block": bool(use_sz_block),
        "use_tau_z_block": bool(use_tau_z_block),
        "use_z2_block": bool(use_z2_block),
        "z2_generator": z2_generator,
        "target_sz2": int(getattr(args, "u1_target_sz2", U1_TARGET_TOTAL_SZ2)),
        "target_tz2": int(getattr(args, "u1_target_tz2", U1_TARGET_TOTAL_TZ2)),
        "z2_target_parity": int(getattr(args, "z2_target_parity", Z2_TARGET_PARITY)) % 2,
        "allow_dense_fallback": bool(getattr(args, "symmetry_allow_dense_fallback", SYMMETRY_ALLOW_DENSE_FALLBACK)),
        "use_translation_x_block": bool(getattr(args, "use_translation_x_block", USE_TRANSLATION_X_BLOCK)),
        "use_translation_y_block": bool(getattr(args, "use_translation_y_block", USE_TRANSLATION_Y_BLOCK)),
        "momentum_x_block": int(getattr(args, "momentum_x_block", MOMENTUM_X_BLOCK)),
        "momentum_y_block": int(getattr(args, "momentum_y_block", MOMENTUM_Y_BLOCK)),
        "use_reflection_block": bool(getattr(args, "use_reflection_block", USE_REFLECTION_BLOCK)),
        "reflection_block": int(getattr(args, "reflection_block", REFLECTION_BLOCK)),
    }


def _sector_dimension_for_spin_half(n_sites: int, target_m2: int) -> int:
    n = int(n_sites)
    numerator = n + int(target_m2)
    if numerator % 2 != 0:
        return 0
    nup = numerator // 2
    if nup < 0 or nup > n:
        return 0
    return int(math.comb(n, nup))


def _spin_orbital_symmetry_reduced_dimension(
    n_sites: int,
    use_sz_block: bool,
    target_sz2: int,
    use_tau_z_block: bool,
    target_tz2: int,
) -> int:
    n = int(n_sites)
    spin_dim = _sector_dimension_for_spin_half(n, target_sz2) if bool(use_sz_block) else (1 << n)
    orbital_dim = _sector_dimension_for_spin_half(n, target_tz2) if bool(use_tau_z_block) else (1 << n)
    return int(spin_dim * orbital_dim)


def _ed_symmetry_mode_enabled(mode: str, *, auto_default: bool) -> bool:
    text = str(mode).strip().lower()
    if text == "on":
        return True
    if text == "off":
        return False
    return bool(auto_default)


def _normalize_ed_symmetry_engine(engine: str) -> str:
    text = str(engine).strip().lower()
    if text == "projector":
        return "standard_projector"
    if text in {"quspin", "quspin_native"}:
        return "quspin_native"
    if text in {"auto", "standard_projector", "quspin_native", "quspin_experimental_c3"}:
        return text
    raise ValueError(
        "ED symmetry engine must be one of auto, standard_projector, "
        "quspin/quspin_native, or quspin_experimental_c3."
    )


def _geometry_allows_translation(geometry_obj: Any, axis: str) -> Tuple[bool, str]:
    axis_key = str(axis).strip().lower()
    if axis_key == "x":
        if not bool(getattr(geometry_obj, "circumference_x", False)):
            return False, "x translation requires periodic boundary conditions along x."
        return True, "uniform Hamiltonian preserves x unit-cell translation on this periodic geometry."
    if axis_key == "y":
        if not bool(getattr(geometry_obj, "circumference_y", False)):
            return False, "y translation requires periodic boundary conditions along y."
        return True, "uniform Hamiltonian preserves y unit-cell translation on this periodic geometry."
    return False, f"unknown translation axis {axis!r}."


def _geometry_can_host_combined_c3(args: argparse.Namespace, geometry_obj: Any) -> Tuple[bool, str]:
    if str(getattr(args, "lattice", "")).strip().lower() != "honeycomb":
        return False, "combined spin-lattice C3 is defined here only for honeycomb geometry."
    if int(getattr(args, "length_x", 0)) != int(getattr(args, "length_y", -1)):
        return False, "combined C3 projector requires an Lx=Ly honeycomb torus in this ED planner."
    if int(getattr(args, "length_x", 0)) < 2:
        return False, "combined C3 projector requires at least a 2x2 honeycomb torus with resolved bond directions."
    if not (bool(getattr(geometry_obj, "circumference_x", False)) and bool(getattr(geometry_obj, "circumference_y", False))):
        return False, "combined C3 projector requires periodic boundaries in both directions."
    return True, "honeycomb Lx=Ly torus can host the combined spin-lattice C3 check."


def _resolve_ed_symmetry_plan(
    *,
    args: argparse.Namespace,
    model_spec_obj: Any,
    geometry_obj: Any,
    resolved_field_vector: Tuple[float, float, float],
    hamiltonian_field_terms: List[Tuple[float, str]],
    shared_symmetry_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve ED symmetries from physics first, then record backend support.

    This intentionally does not change DMRG/iDMRG/PEPS choices.  The result is an
    ED-only plan stored on ``args.ed_symmetry_plan`` and copied into summaries.
    """
    field_report = classify_external_field(
        getattr(args, "external_field_treatment", "off"),
        resolved_field_vector,
    )
    field_class = str(field_report.get("field_class", "none"))
    actual_field_class = "none" if field_class == "perturbation_only" else field_class
    is_yao_lee_spin_orbital = bool(
        str(getattr(model_spec_obj, "model_family", "")).strip().lower() == "yao_lee"
        and str(getattr(model_spec_obj, "orbital_rep", "")).strip() == "1/2"
    )
    requested_engine = _normalize_ed_symmetry_engine(getattr(args, "ed_symmetry_engine", ED_SYMMETRY_ENGINE))
    engine_prefers_quspin_native = requested_engine in ("quspin_native", "quspin_experimental_c3")
    quspin_experimental_c3_report: Dict[str, Any] | None = None
    quspin_experimental_fused_translation_report: Dict[str, Any] | None = None
    if requested_engine == "quspin_experimental_c3":
        try:
            import quspin_backend as quspin_c3_backend

            quspin_experimental_c3_report = quspin_c3_backend.quspin_combined_c3_api_support_report(
                model_family=str(getattr(model_spec_obj, "model_family", "")),
                phase_scan_requested=bool(getattr(args, "run_phase_scan", RUN_PHASE_SCAN)),
            )
        except Exception as exc:
            quspin_experimental_c3_report = {
                "status": "checked",
                "experimental": True,
                "combined_c3_implemented": False,
                "phase_scan_allowed": False,
                "api_error": str(exc),
                "reason": "Could not validate QuSpin experimental C3 API support.",
            }
    requested_backend = str(getattr(args, "ed_backend", ED_BACKEND)).strip().lower()
    if requested_backend == "ed":
        requested_backend = "standard"
    target_tz2 = int(getattr(args, "u1_target_tz2", U1_TARGET_TOTAL_TZ2))
    target_sz2 = int(getattr(args, "u1_target_sz2", U1_TARGET_TOTAL_SZ2))
    z2_target_parity = int(getattr(args, "z2_target_parity", Z2_TARGET_PARITY)) % 2
    n_sites = int(getattr(geometry_obj, "number_of_sites", 0))

    accepted: List[str] = []
    dropped: List[Dict[str, str]] = []
    reasons: Dict[str, str] = {}
    warnings_list: List[str] = []

    def accept(name: str, reason: str) -> None:
        if name not in accepted:
            accepted.append(name)
        reasons[name] = reason

    def drop(name: str, reason: str) -> None:
        if name not in {str(item.get("name")) for item in dropped}:
            dropped.append({"name": name, "reason": reason})
        reasons[name] = reason

    use_sz_block = False
    if is_yao_lee_spin_orbital:
        if bool(shared_symmetry_settings.get("use_sz_block", False)) or "sz" in set(getattr(args, "symmetry_reductions", ())):
            drop("sz", "Total Sz is not conserved for the spin-orbital Yao-Lee Hamiltonian; ED never uses an Sz block.")
    else:
        use_sz_block = bool(shared_symmetry_settings.get("use_sz_block", False))
        if use_sz_block:
            accept("sz", "non-Yao-Lee model requested the shared total-Sz ED block.")

    use_tau_z_block = False
    if is_yao_lee_spin_orbital:
        tz_dim = _sector_dimension_for_spin_half(n_sites, target_tz2)
        if tz_dim > 0:
            use_tau_z_block = True
            accept("tz", "Total Tz is conserved for spin-orbital Yao-Lee, including uniform spin fields.")
        else:
            drop("tz", f"Requested total 2*Tz={target_tz2} is unreachable for N={n_sites} orbital-1/2 sites.")
    elif bool(shared_symmetry_settings.get("use_tau_z_block", False)):
        use_tau_z_block = True
        accept("tz", "shared symmetry settings requested a reachable total-Tz block.")

    translation_plan: Dict[str, Any] = {
        "requested_x": bool(getattr(args, "use_translation_x_block", USE_TRANSLATION_X_BLOCK)),
        "requested_y": bool(getattr(args, "use_translation_y_block", USE_TRANSLATION_Y_BLOCK)),
        "use_x": False,
        "use_y": False,
        "momentum_x": int(getattr(args, "momentum_x_block", MOMENTUM_X_BLOCK)),
        "momentum_y": int(getattr(args, "momentum_y_block", MOMENTUM_Y_BLOCK)),
        "reasons": {},
    }
    if translation_plan["requested_x"]:
        ok, reason = _geometry_allows_translation(geometry_obj, "x")
        translation_plan["use_x"] = bool(ok)
        translation_plan["reasons"]["x"] = reason
        if ok:
            accept("translation_x", reason)
        else:
            drop("translation_x", reason)
    if translation_plan["requested_y"]:
        ok, reason = _geometry_allows_translation(geometry_obj, "y")
        translation_plan["use_y"] = bool(ok)
        translation_plan["reasons"]["y"] = reason
        if ok:
            accept("translation_y", reason)
        else:
            drop("translation_y", reason)

    z2_mode = str(getattr(args, "ed_z2_mode", ED_Z2_MODE)).strip().lower()
    z2_kind_request = str(getattr(args, "ed_z2_kind", ED_Z2_KIND)).strip().lower()
    z2_auto_default = bool(is_yao_lee_spin_orbital and actual_field_class in ("none", "hz"))
    z2_requested = _ed_symmetry_mode_enabled(z2_mode, auto_default=z2_auto_default)
    use_z2_block = False
    z2_kind: str | None = None
    if z2_requested:
        if not is_yao_lee_spin_orbital:
            drop("z2", "ED Z2 projector planning is currently specialized to spin-orbital Yao-Lee.")
        else:
            if z2_kind_request == "auto":
                if engine_prefers_quspin_native:
                    z2_kind = "spin_flip" if actual_field_class == "none" else None
                else:
                    z2_kind = "spin_pi_z" if actual_field_class in ("none", "hz") else None
            else:
                z2_kind = z2_kind_request
            if actual_field_class == "h111":
                drop("z2", "Pure spin-sector Z2 is not conserved for normalized H[111]; ED disables Z2.")
                z2_kind = None
            elif actual_field_class not in ("none", "hz"):
                drop("z2", f"Pure spin-sector Z2 is not conserved for field_class={actual_field_class}.")
                z2_kind = None
            elif z2_kind == "spin_flip" and actual_field_class == "hz":
                drop("z2", "spin_flip Z2 is broken by pure Hz; use spin_pi_z for the surviving parity.")
                z2_kind = None
            elif z2_kind == "spin_pi_z":
                use_z2_block = True
                accept("z2:spin_pi_z", "Rz(pi) spin parity is conserved for zero field and pure Hz.")
            elif z2_kind == "spin_flip" and actual_field_class == "none":
                use_z2_block = True
                accept("z2:spin_flip", "spin_flip Z2 is conserved for the zero-field Yao-Lee Hamiltonian.")
            else:
                drop("z2", f"Unsupported ED Z2 kind {z2_kind_request!r} for field_class={actual_field_class}.")
                z2_kind = None
    else:
        if is_yao_lee_spin_orbital and actual_field_class == "h111":
            drop("z2", "Pure spin-sector Z2 is not conserved for normalized H[111]; ED disables Z2.")
        elif is_yao_lee_spin_orbital and actual_field_class not in ("none", "hz"):
            drop("z2", f"Pure spin-sector Z2 is not conserved for field_class={actual_field_class}.")
        else:
            drop("z2", "ED Z2 mode is off or auto did not select a conserved spin-sector parity.")

    c3_mode = str(getattr(args, "ed_c3_mode", ED_C3_MODE)).strip().lower()
    c3_auto_default = bool(is_yao_lee_spin_orbital and actual_field_class in ("none", "h111"))
    c3_requested = _ed_symmetry_mode_enabled(c3_mode, auto_default=c3_auto_default)
    use_c3_block = False
    c3_reason = None
    c3_gamma_momentum_reason = (
        "combined C3 is implemented only in the Gamma momentum sector because translations "
        "and C3 do not commute at generic momentum."
    )
    c3_strict = bool(getattr(args, "strict_symmetry_selection_rules", STRICT_SYMMETRY_SELECTION_RULES))
    if c3_requested:
        if not is_yao_lee_spin_orbital:
            c3_reason = "combined spin-lattice C3 is currently defined only for spin-orbital Yao-Lee."
            drop("combined_c3", c3_reason)
        elif actual_field_class == "hz":
            c3_reason = "pure Hz breaks combined spin-lattice C3; ED disables C3."
            drop("combined_c3", c3_reason)
        elif actual_field_class not in ("none", "h111"):
            c3_reason = f"field_class={actual_field_class} does not preserve combined spin-lattice C3."
            drop("combined_c3", c3_reason)
        else:
            geometry_ok, geometry_reason = _geometry_can_host_combined_c3(args, geometry_obj)
            if geometry_ok:
                if int(translation_plan["momentum_x"]) != 0 or int(translation_plan["momentum_y"]) != 0:
                    c3_reason = c3_gamma_momentum_reason
                    if c3_strict:
                        raise ValueError(c3_reason)
                    drop("combined_c3", c3_reason)
                else:
                    for axis in ("x", "y"):
                        ok, reason = _geometry_allows_translation(geometry_obj, axis)
                        if ok:
                            translation_plan[f"use_{axis}"] = True
                            translation_plan["reasons"][axis] = (
                                "enabled automatically because combined C3 is applied only in the "
                                "C3-invariant k=(0,0) translation sector."
                            )
                            accept(f"translation_{axis}", translation_plan["reasons"][axis])
                        else:
                            drop(f"translation_{axis}", reason)
                    translation_plan["momentum_x"] = 0
                    translation_plan["momentum_y"] = 0
                    use_c3_block = bool(translation_plan["use_x"] and translation_plan["use_y"])
                    c3_reason = (
                        "combined spin-lattice C3 is allowed by the field class; "
                        "standard projector ED verifies [H,C3] and applies q-sector projectors."
                    )
                    if use_c3_block:
                        accept("combined_c3", c3_reason)
                    else:
                        drop("combined_c3", "combined C3 requires both x and y translation sectors at k=(0,0).")
            else:
                c3_reason = geometry_reason
                drop("combined_c3", geometry_reason)
    else:
        if is_yao_lee_spin_orbital and actual_field_class == "hz":
            c3_reason = "pure Hz breaks combined spin-lattice C3; ED disables C3."
        elif is_yao_lee_spin_orbital and actual_field_class not in ("none", "h111"):
            c3_reason = f"field_class={actual_field_class} does not preserve combined spin-lattice C3."
        else:
            c3_reason = "ED C3 mode is off or auto did not select C3 for this field class."
        drop("combined_c3", c3_reason)

    def remove_accepted(prefix: str) -> None:
        accepted[:] = [item for item in accepted if not str(item).startswith(prefix)]

    if use_c3_block and use_z2_block:
        remove_accepted("z2:")
        use_z2_block = False
        dropped_z2_kind = z2_kind
        z2_kind = None
        reason = (
            "ED does not currently implement the full group projector for simultaneous "
            "spin-sector Z2 and true combined spin-lattice C3 labels.  The planner keeps "
            "the C3 + Gamma-translation route and drops Z2; use C3 off for the Z2+translation route."
        )
        drop("z2", reason)
        warnings_list.append(
            f"Dropped Z2 ({dropped_z2_kind}) because combined C3 and Z2 are not treated as "
            "independent commuting labels in the current ED projector implementation."
        )

    if (
        requested_engine == "auto"
        and actual_field_class == "none"
        and use_z2_block
        and z2_kind == "spin_pi_z"
        and not translation_plan["use_x"]
        and not translation_plan["use_y"]
        and not use_c3_block
    ):
        remove_accepted("z2:")
        z2_kind = "spin_flip"
        accept(
            "z2:spin_flip",
            "auto selected QuSpin-native zero-field spin_flip because no projector-only ED symmetries are active.",
        )

    fused_translation_tz_reason = (
        "QuSpin tensor_basis translation is not used with Tz because Yao-Lee translation must act "
        "on fused spin-orbital physical sites."
    )
    experimental_fused_translation_requested = bool(
        getattr(args, "ed_quspin_experimental_fused_translation", ED_QUSPIN_EXPERIMENTAL_FUSED_TRANSLATION)
    )
    quspin_requested_with_fused_tz_translation = bool(
        use_tau_z_block
        and (translation_plan["use_x"] or translation_plan["use_y"])
        and (requested_backend == "quspin" or requested_engine == "quspin_native")
    )
    if use_tau_z_block and (translation_plan["use_x"] or translation_plan["use_y"]):
        try:
            import quspin_backend as quspin_fused_backend

            quspin_experimental_fused_translation_report = (
                quspin_fused_backend.quspin_fused_translation_api_support_report(
                    geometry_obj,
                    use_tau_z_block=bool(use_tau_z_block),
                    use_z2_block=bool(use_z2_block),
                    requested=bool(experimental_fused_translation_requested),
                )
            )
        except Exception as exc:
            quspin_experimental_fused_translation_report = {
                "status": "checked",
                "experimental": True,
                "requested": bool(experimental_fused_translation_requested),
                "implemented": False,
                "available": False,
                "api_error": str(exc),
                "reason": "Could not validate QuSpin fused-translation API support.",
            }
    fused_translation_available = bool(
        experimental_fused_translation_requested
        and isinstance(quspin_experimental_fused_translation_report, dict)
        and quspin_experimental_fused_translation_report.get("available", False)
        and quspin_experimental_fused_translation_report.get("implemented", False)
    )
    if quspin_requested_with_fused_tz_translation:
        if fused_translation_available:
            warnings_list.append(
                "Using experimental QuSpin fused-site translation path validated against standard_projector."
            )
        else:
            warnings_list.append(fused_translation_tz_reason)

    quspin_native_forces_subset = requested_engine == "quspin_experimental_c3"
    if quspin_native_forces_subset:
        if use_z2_block and z2_kind == "spin_pi_z":
            remove_accepted("z2:")
            use_z2_block = False
            z2_kind = None
            drop(
                "z2",
                "quspin_native cannot represent spin_pi_z in the current tensor-basis path; "
                "use ED_SYMMETRY_ENGINE=standard_projector for this projector.",
            )
        if translation_plan["use_x"]:
            remove_accepted("translation_x")
            translation_plan["use_x"] = False
            drop(
                "translation_x",
                "quspin_native does not use Tx because QuSpin tensor_basis would impose separate "
                "spin/orbital translations, not the fused physical-site translation; "
                "use standard_projector for production fused-site translation blocks.",
            )
        if translation_plan["use_y"]:
            remove_accepted("translation_y")
            translation_plan["use_y"] = False
            drop(
                "translation_y",
                "quspin_native does not use Ty because QuSpin tensor_basis would impose separate "
                "spin/orbital translations, not the fused physical-site translation; "
                "use standard_projector for production fused-site translation blocks.",
            )
        if use_c3_block:
            remove_accepted("combined_c3")
            use_c3_block = False
            c3_reason = (
                "quspin_experimental_c3/native QuSpin pure site maps are not the physical "
                "Yao-Lee combined C3 = lattice rotation times local spin rotation; "
                "use standard_projector for combined C3."
            )
            drop("combined_c3", c3_reason)
        if requested_engine == "quspin_experimental_c3":
            experimental_reason = (
                quspin_experimental_c3_report.get("reason")
                if isinstance(quspin_experimental_c3_report, dict)
                else None
            )
            drop(
                "combined_c3",
                str(
                    experimental_reason
                    or (
                        "quspin_experimental_c3 rejects pure C3 site maps for Yao-Lee; "
                        "combined C3 needs local spin rotation support."
                    )
                ),
            )
            warnings_list.append(
                "quspin_experimental_c3 is experimental and currently disabled for Yao-Lee combined C3: "
                "pure integer C3 maps are rejected; a user_basis/custom phase implementation or "
                "spin-[111] basis encoding plus N=8 standard_projector validation is required."
            )

    needs_standard_projector = bool(
        use_c3_block
        or ((translation_plan["use_x"] or translation_plan["use_y"]) and not fused_translation_available)
        or (use_z2_block and z2_kind == "spin_pi_z")
    )
    quspin_native_safe_subset = bool(
        is_yao_lee_spin_orbital
        and use_tau_z_block
        and not use_sz_block
        and not needs_standard_projector
        and not use_c3_block
        and (not translation_plan["use_x"] or fused_translation_available)
        and (not translation_plan["use_y"] or fused_translation_available)
        and (
            not use_z2_block
            or (z2_kind == "spin_flip" and actual_field_class == "none")
        )
    )
    if requested_engine == "standard_projector":
        effective_engine = "standard_projector"
    elif needs_standard_projector:
        effective_engine = "standard_projector"
    elif requested_engine in ("quspin_native", "quspin_experimental_c3"):
        effective_engine = requested_engine
    elif quspin_native_safe_subset:
        effective_engine = "quspin_native"
    elif requested_backend == "quspin":
        effective_engine = "quspin_native"
    else:
        effective_engine = "standard_projector"
    actual_backend = "quspin" if effective_engine.startswith("quspin") else "standard"
    spin_pi_z_quspin_reason = (
        "QuSpin zblock supports only the tested zero-field spin_flip generator; "
        "spin_pi_z parity requires standard_projector."
    )
    backend_override_reason = (
        fused_translation_tz_reason
        if quspin_requested_with_fused_tz_translation and actual_backend == "standard"
        else (
            spin_pi_z_quspin_reason
            if (
                use_z2_block
                and z2_kind == "spin_pi_z"
                and actual_backend == "standard"
                and (requested_backend == "quspin" or requested_engine == "quspin_native")
            )
            else None
        )
    )
    if not use_z2_block:
        z2_selection_reason = reasons.get("z2", "No ED Z2 generator was selected.")
    elif z2_kind == "spin_pi_z":
        z2_selection_reason = (
            "Using spin_pi_z/Rz(pi) parity. This generator is not implemented by the QuSpin "
            "zblock path, so ED uses standard_projector."
        )
    elif z2_kind == "spin_flip" and actual_field_class == "none":
        z2_selection_reason = (
            "Using the tested zero-field spin_flip generator. QuSpin-native may use it only "
            "when no fused-site translation/C3 projector is active."
        )
        if translation_plan["use_x"] or translation_plan["use_y"]:
            z2_selection_reason += " Translation is requested, so the route is standard_projector."
        elif actual_backend == "quspin":
            z2_selection_reason += " QuSpin-native selected the spin_flip zblock."
        else:
            z2_selection_reason += " QuSpin-native was not selected by the requested backend/engine."
    else:
        z2_selection_reason = (
            f"Z2 generator {z2_kind!r} is not supported for field_class={actual_field_class}."
        )
    quspin_z2_selection_reason = (
        z2_selection_reason
        if z2_kind == "spin_flip" and actual_field_class == "none"
        else "QuSpin-native supports only zero-field spin_flip zblock; it does not implement spin_pi_z parity."
    )
    requested_symmetries: List[str] = [str(item) for item in getattr(args, "symmetry_reductions", ())]
    if translation_plan["requested_x"]:
        requested_symmetries.append("translation_x")
    if translation_plan["requested_y"]:
        requested_symmetries.append("translation_y")
    if z2_requested:
        requested_symmetries.append("z2")
    if c3_requested:
        requested_symmetries.append("combined_c3")

    backend_support_status = {
        "standard_projector": {
            "u1_tz": bool(use_tau_z_block),
            "spin_pi_z": True,
            "spin_flip": False,
            "translation": True,
            "combined_c3": True,
            "reason": (
                "standard ED applies Tz first, then projector-based spin_pi_z and fused-site "
                "translation blocks, then combined C3 q-sector projectors when requested."
            ),
        },
        "quspin_native": {
            "u1_tz": bool(use_tau_z_block),
            "spin_flip": bool(use_z2_block and z2_kind == "spin_flip" and actual_field_class == "none"),
            "spin_pi_z": False,
            "translation": False,
            "combined_c3": False,
            "reason": (
                "current QuSpin tensor-basis path can apply Tz and zero-field spin_flip; "
                "spin_pi_z, fused physical-site translations, and combined C3 are handled by the standard projector engine."
            ),
            "z2_reason": quspin_z2_selection_reason,
        },
        "quspin_experimental_c3": {
            "u1_tz": bool(use_tau_z_block),
            "spin_flip": bool(use_z2_block and z2_kind == "spin_flip" and actual_field_class == "none"),
            "spin_pi_z": False,
            "translation": False,
            "combined_c3": False,
            "has_user_basis": (
                bool(quspin_experimental_c3_report.get("has_user_basis", False))
                if isinstance(quspin_experimental_c3_report, dict)
                else None
            ),
            "combined_c3_implemented": (
                bool(quspin_experimental_c3_report.get("combined_c3_implemented", False))
                if isinstance(quspin_experimental_c3_report, dict)
                else False
            ),
            "phase_scan_allowed": (
                bool(quspin_experimental_c3_report.get("phase_scan_allowed", False))
                if isinstance(quspin_experimental_c3_report, dict)
                else False
            ),
            "reason": (
                "experimental pure site C3 maps are rejected for Yao-Lee; physical combined C3 needs "
                "lattice rotation times local spin rotation and N=8 standard_projector validation."
            ),
        },
        "quspin_experimental_fused_translation": {
            "u1_tz": bool(use_tau_z_block),
            "translation": bool(fused_translation_available),
            "spin_flip": bool(use_z2_block and z2_kind == "spin_flip" and actual_field_class == "none"),
            "spin_pi_z": False,
            "combined_c3": False,
            "requested": bool(experimental_fused_translation_requested),
            "available": bool(fused_translation_available),
            "reason": (
                quspin_experimental_fused_translation_report.get("reason")
                if isinstance(quspin_experimental_fused_translation_report, dict)
                else "QuSpin fused translation was not requested or not checked."
            ),
        },
    }
    if use_z2_block and z2_kind == "spin_pi_z":
        warnings_list.append("ED plan accepts spin_pi_z parity; standard ED applies it with the projector engine.")
    if translation_plan["use_x"] or translation_plan["use_y"]:
        warnings_list.append("ED plan accepts fused-site translation symmetry; standard ED applies it with the projector engine.")
    if use_c3_block:
        warnings_list.append("ED plan accepts combined C3; standard ED applies q-sector projectors after commutator checks.")

    return {
        "status": "resolved",
        "engine": effective_engine,
        "symmetry_engine": effective_engine,
        "requested_engine": requested_engine,
        "effective_engine": effective_engine,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "effective_backend": actual_backend,
        "backend_override_reason": backend_override_reason,
        "engine_selection_reason": (
            backend_override_reason
            if backend_override_reason
            else "projector-only symmetries require the in-repo standard projector path"
            if effective_engine == "standard_projector" and needs_standard_projector
            else (
                "requested QuSpin-native/experimental engine uses only the supported native subset"
                if requested_engine in ("quspin_native", "quspin_experimental_c3")
                else (
                    "auto selected quspin_native because the requested ED symmetries are within "
                    "the validated QuSpin subset (Tz-only or zero-field Tz+spin_flip)"
                    if effective_engine == "quspin_native" and quspin_native_safe_subset
                    else "auto selected the backend-compatible ED symmetry route"
                )
            )
        ),
        "model_family": str(getattr(model_spec_obj, "model_family", "")),
        "orbital_rep": str(getattr(model_spec_obj, "orbital_rep", "")),
        "field_class": field_class,
        "actual_hamiltonian_field_class": actual_field_class,
        "field_classification": field_report,
        "hamiltonian_field_terms": [(float(coefficient), str(op_name)) for coefficient, op_name in hamiltonian_field_terms],
        "requested": {
            "engine": requested_engine,
            "backend": requested_backend,
            "c3_mode": c3_mode,
            "c3_q_blocks": str(getattr(args, "ed_c3_q_blocks", ED_C3_Q_BLOCKS)),
            "z2_mode": z2_mode,
            "z2_kind": z2_kind_request,
            "translation_x": bool(translation_plan["requested_x"]),
            "translation_y": bool(translation_plan["requested_y"]),
            "momentum_x": int(translation_plan["momentum_x"]),
            "momentum_y": int(translation_plan["momentum_y"]),
        },
        "requested_symmetries": requested_symmetries,
        "accepted_symmetries": accepted,
        "dropped_symmetries": dropped,
        "reasons": reasons,
        "use_sz_block": bool(use_sz_block),
        "use_tau_z_block": bool(use_tau_z_block),
        "use_z2_block": bool(use_z2_block),
        "z2_kind": z2_kind,
        "z2_generator": z2_kind,
        "z2_generator_used": z2_kind if use_z2_block else None,
        "z2_selection_reason": z2_selection_reason,
        "quspin_z2_selection_reason": quspin_z2_selection_reason,
        "z2_target_parity": int(z2_target_parity),
        "use_translation_x_block": bool(translation_plan["use_x"]),
        "use_translation_y_block": bool(translation_plan["use_y"]),
        "momentum_x_block": int(translation_plan["momentum_x"]),
        "momentum_y_block": int(translation_plan["momentum_y"]),
        "translation": translation_plan,
        "use_c3_block": bool(use_c3_block),
        "c3_q_blocks": str(getattr(args, "ed_c3_q_blocks", ED_C3_Q_BLOCKS)),
        "c3_commutator_check": {
            "required": bool(use_c3_block),
            "status": "pending_projector_application" if use_c3_block else "not_required",
            "reason": c3_reason,
        },
        "quspin_experimental_c3": quspin_experimental_c3_report,
        "quspin_experimental_fused_translation": quspin_experimental_fused_translation_report,
        "target_sz2": int(target_sz2),
        "target_tz2": int(target_tz2),
        "backend_support_status": backend_support_status,
        "warnings": warnings_list,
    }


def _ed_plan_requires_standard_projector(ed_plan: Dict[str, Any], model_spec_obj: Any) -> bool:
    """Whether ED must use the bitwise projector path instead of QuSpin."""
    if not isinstance(ed_plan, dict) or ed_plan.get("status") != "resolved":
        return False
    effective_engine = _normalize_ed_symmetry_engine(
        ed_plan.get("effective_engine", ed_plan.get("engine", ED_SYMMETRY_ENGINE))
    )
    if effective_engine != "standard_projector":
        return False
    is_yao_lee_spin_orbital = bool(
        str(getattr(model_spec_obj, "model_family", "")).strip().lower() == "yao_lee"
        and str(getattr(model_spec_obj, "spin_rep", "")).strip() == "1/2"
        and str(getattr(model_spec_obj, "orbital_rep", "")).strip() == "1/2"
    )
    if not is_yao_lee_spin_orbital or not bool(ed_plan.get("use_tau_z_block", False)):
        return False
    projector_z2 = bool(
        ed_plan.get("use_z2_block", False)
        and str(ed_plan.get("z2_kind")) == "spin_pi_z"
    )
    return bool(
        projector_z2
        or ed_plan.get("use_translation_x_block", False)
        or ed_plan.get("use_translation_y_block", False)
        or ed_plan.get("use_c3_block", False)
    )


def _ed_projector_reduction_factor_estimate(ed_plan: Dict[str, Any], geometry_obj: Any) -> int:
    factor = 1
    if bool(ed_plan.get("use_z2_block", False)) and str(ed_plan.get("z2_kind")) == "spin_pi_z":
        factor *= 2
    if bool(ed_plan.get("use_translation_x_block", False)):
        factor *= max(1, int(getattr(geometry_obj, "length_x", 1) or 1))
    if bool(ed_plan.get("use_translation_y_block", False)):
        factor *= max(1, int(getattr(geometry_obj, "length_y", 1) or 1))
    if bool(ed_plan.get("use_c3_block", False)):
        factor *= 3
    return int(max(1, factor))


def _m2_values_for_spin_value(spin_value: float) -> List[int]:
    two_s = int(round(2.0 * float(spin_value)))
    if two_s <= 0:
        return [0]
    return [int(two_s - 2 * index) for index in range(two_s + 1)]


def _charge_sector_dimension(single_site_charges: List[int], n_sites: int, target_charge: int) -> int:
    counts: Dict[int, int] = {0: 1}
    charges = [int(charge) for charge in single_site_charges]
    for _site in range(max(0, int(n_sites))):
        next_counts: Dict[int, int] = {}
        for total_charge, count in counts.items():
            for charge in charges:
                new_total = int(total_charge + charge)
                next_counts[new_total] = int(next_counts.get(new_total, 0) + count)
        counts = next_counts
    return int(counts.get(int(target_charge), 0))


def _symmetry_hilbert_dimension_report(
    geometry_obj: Any,
    model_spec_obj: Any,
    symmetry_settings: Dict[str, Any],
) -> Dict[str, Any]:
    n_sites = int(getattr(geometry_obj, "number_of_sites", 0))
    spin_dim = int(getattr(model_spec_obj, "spin_dim", 1))
    orbital_dim = int(getattr(model_spec_obj, "orbital_dim", 1))
    local_dim = int(getattr(model_spec_obj, "physical_dim", max(1, spin_dim * orbital_dim)))
    spin_full_dim = int(spin_dim ** n_sites)
    orbital_full_dim = int(orbital_dim ** n_sites)
    full_dim = int(local_dim ** n_sites)

    use_sz_block = bool(symmetry_settings.get("use_sz_block", False))
    use_tau_z_block = bool(symmetry_settings.get("use_tau_z_block", False))
    use_z2_block = bool(symmetry_settings.get("use_z2_block", False))
    target_sz2 = int(symmetry_settings.get("target_sz2", U1_TARGET_TOTAL_SZ2))
    target_tz2 = int(symmetry_settings.get("target_tz2", U1_TARGET_TOTAL_TZ2))

    spin_m2_values = _m2_values_for_spin_value(float(getattr(model_spec_obj, "spin_value", 0.0)))
    orbital_m2_values = _m2_values_for_spin_value(float(getattr(model_spec_obj, "orbital_value", 0.0)))
    spin_sector_dim = (
        _charge_sector_dimension(spin_m2_values, n_sites, target_sz2)
        if use_sz_block
        else spin_full_dim
    )
    orbital_sector_dim = (
        _charge_sector_dimension(orbital_m2_values, n_sites, target_tz2)
        if use_tau_z_block
        else orbital_full_dim
    )
    u1_effective_dim = int(spin_sector_dim * orbital_sector_dim)
    effective_dim = u1_effective_dim
    z2_dimension_note = None
    if use_z2_block:
        if u1_effective_dim > 0:
            effective_dim = int((u1_effective_dim + 1) // 2)
            z2_dimension_note = (
                "Z2 dimension is reported as a conservative half-sector estimate; "
                "backend-specific bases may record the exact dimension later."
            )
        else:
            effective_dim = 0
            z2_dimension_note = "Z2 was requested but the preceding U1 sector is unreachable."

    reduction_factor = None if effective_dim <= 0 else float(full_dim) / float(effective_dim)
    active_labels: List[str] = []
    if use_sz_block:
        active_labels.append(f"Sz={target_sz2}")
    if use_tau_z_block:
        active_labels.append(f"Tz={target_tz2}")
    if use_z2_block:
        active_labels.append(f"Z2={int(symmetry_settings.get('z2_target_parity', Z2_TARGET_PARITY)) % 2}")
    return {
        "number_of_sites": n_sites,
        "local_dimension": local_dim,
        "spin_full_dimension": spin_full_dim,
        "orbital_full_dimension": orbital_full_dim,
        "full_hilbert_dimension": full_dim,
        "spin_sector_dimension": int(spin_sector_dim),
        "orbital_sector_dimension": int(orbital_sector_dim),
        "u1_effective_hilbert_dimension": int(u1_effective_dim),
        "effective_hilbert_dimension": int(effective_dim),
        "reduction_factor": reduction_factor,
        "active_reductions": active_labels,
        "basis_label": " x ".join(active_labels) if active_labels else "full dense Hilbert space",
        "z2_dimension_note": z2_dimension_note,
    }


def _dimension_ratio_text(dimension_report: Dict[str, Any]) -> str:
    ratio = dimension_report.get("reduction_factor")
    if ratio is None:
        return "unreachable"
    if float(ratio) >= 1000.0:
        return f"{float(ratio):.3e}x"
    return f"{float(ratio):.3f}x"


def _ed_plan_drop_reason(ed_plan: Dict[str, Any], name: str) -> str | None:
    if not isinstance(ed_plan, dict):
        return None
    for item in ed_plan.get("dropped_symmetries", []):
        if not isinstance(item, dict):
            continue
        item_name = str(item.get("name", ""))
        if item_name == name or item_name.startswith(f"{name}:"):
            return str(item.get("reason", "dropped"))
    return None


def _ed_actual_applied_reductions_from_spectrum(
    planned_reductions: List[str],
    spectrum: Dict[str, Any] | None,
) -> List[str]:
    """Prefer the completed ED spectrum over the pre-run plan.

    The standard projector can drop combined C3 at runtime if a memory guard is
    hit.  This keeps summaries from claiming a reduction that was only planned.
    """
    if not isinstance(spectrum, dict):
        return list(planned_reductions)
    applied: List[str] = []
    if bool(spectrum.get("use_sz_block", False)):
        applied.append("sz")
    if bool(spectrum.get("use_tau_z_block", False)):
        applied.append("tz")
    if bool(spectrum.get("use_z2_block", False)):
        applied.append("z2")
    if bool(spectrum.get("use_translation_x_block", False)):
        applied.append("translation_x")
    if bool(spectrum.get("use_translation_y_block", False)):
        applied.append("translation_y")
    if bool(spectrum.get("use_c3_block", False)):
        applied.append("combined_c3")
    return applied or list(planned_reductions)


def _ed_symmetry_status_text(
    ed_plan: Dict[str, Any],
    *,
    spectrum: Dict[str, Any] | None = None,
    max_reason_chars: int = 96,
) -> str:
    if not isinstance(ed_plan, dict):
        return "unavailable"
    completed = isinstance(spectrum, dict)

    def reason_fragment(name: str) -> str:
        reason = _ed_plan_drop_reason(ed_plan, name)
        if not reason:
            return ""
        if len(reason) > int(max_reason_chars):
            reason = reason[: int(max_reason_chars) - 3] + "..."
        return f"({reason})"

    def active(plan_key: str, spectrum_key: str | None = None) -> bool:
        if completed and spectrum_key is not None:
            return bool(spectrum.get(spectrum_key, False))
        return bool(ed_plan.get(plan_key, False))

    entries: List[str] = []
    entries.append(
        "Tz="
        + (
            f"{'applied' if completed else 'planned'}(target={int(ed_plan.get('target_tz2', 0))})"
            if active("use_tau_z_block", "use_tau_z_block")
            else f"dropped{reason_fragment('tz')}"
        )
    )
    if active("use_sz_block", "use_sz_block"):
        entries.append("Sz=" + ("applied" if completed else "planned"))
    elif _ed_plan_drop_reason(ed_plan, "sz"):
        entries.append(f"Sz=dropped{reason_fragment('sz')}")

    if active("use_z2_block", "use_z2_block"):
        z2_kind = (
            spectrum.get("z2_kind")
            if completed and spectrum is not None
            else ed_plan.get("z2_kind")
        )
        entries.append(
            "Z2="
            + ("applied" if completed else "planned")
            + (f"({z2_kind})" if z2_kind else "")
        )
    else:
        entries.append(f"Z2=dropped{reason_fragment('z2')}")

    for axis, label in (("x", "Tx"), ("y", "Ty")):
        plan_key = f"use_translation_{axis}_block"
        spectrum_key = f"use_translation_{axis}_block"
        momentum_key = f"momentum_{axis}_block"
        if active(plan_key, spectrum_key):
            momentum = int(ed_plan.get(momentum_key, 0) or 0)
            entries.append(f"{label}={'applied' if completed else 'planned'}(k={momentum})")
        elif _ed_plan_drop_reason(ed_plan, f"translation_{axis}"):
            entries.append(f"{label}=dropped{reason_fragment(f'translation_{axis}')}")

    if active("use_c3_block", "use_c3_block"):
        q_text = str(ed_plan.get("c3_q_blocks", "all"))
        if completed and spectrum is not None:
            selected_q = spectrum.get("selected_c3_q")
            comm = (spectrum.get("commutator_norms") or {}).get("H_C3")
            verify = "verified" if comm is not None else "applied"
            if selected_q is not None:
                q_text = f"selected_q={selected_q}"
            if comm is not None:
                entries.append(f"C3={verify}({q_text}, ||[H,C3]||={float(comm):.2e})")
            else:
                entries.append(f"C3={verify}({q_text})")
        else:
            entries.append(f"C3=planned(q={q_text}, commutator_check=pending)")
    else:
        entries.append(f"C3=dropped{reason_fragment('combined_c3')}")
    return ", ".join(entries)


def _record_output_status(
    summary: Dict[str, Any],
    key: str,
    filename: str,
    status: str,
    error: str | None = None,
    reason: str | None = None,
) -> None:
    outputs = summary.setdefault("outputs", {})
    outputs[key] = filename
    output_status = summary.setdefault("output_status", {})
    output_status[key] = {"status": status}
    if error is not None:
        output_status[key]["error"] = error
    if reason is not None:
        output_status[key]["reason"] = reason


PLOT_OUTPUT_WARNING_STATUSES = {"failed", "skipped_optional_dependency"}


def _output_warning_keys(summary: Dict[str, Any]) -> List[str]:
    output_status = summary.get("output_status", {})
    if not isinstance(output_status, dict):
        return []
    return [
        str(key)
        for key, item in output_status.items()
        if isinstance(item, dict) and str(item.get("status")) in PLOT_OUTPUT_WARNING_STATUSES
    ]


def _attach_plot_output_warnings(summary: Dict[str, Any], section_key: str | None = None) -> List[str]:
    warning_keys = _output_warning_keys(summary)
    if not warning_keys:
        return []
    warnings_payload = summary.setdefault("plot_output_warnings", {})
    if isinstance(warnings_payload, dict):
        warnings_payload["keys"] = warning_keys
        warnings_payload["note"] = (
            "One or more requested plots were not written. Install matplotlib in the active "
            "environment or rerun with the relevant --no-plot-* flag."
        )
    if section_key is not None and isinstance(summary.get(section_key), dict):
        section = summary[section_key]
        section["plot_output_warnings"] = warning_keys
        if str(section.get("status", "completed")) == "completed":
            section["status"] = "completed_with_warnings"
    return warning_keys


def _all_plaquette_fluxes_from_payload(payload: Any) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    for candidate in (
        payload,
        payload.get("plaquette_flux"),
        payload.get("phase_observables"),
        payload.get("observables"),
        payload.get("spectrum"),
    ):
        if not isinstance(candidate, dict):
            continue
        try:
            fluxes = extract_all_plaquette_fluxes(candidate)
        except Exception:
            fluxes = {}
        if fluxes:
            return {str(index): float(value) for index, value in fluxes.items()}
    return {}


def _record_all_plaquette_fluxes(
    summary: Dict[str, Any],
    method_key: str,
    payload: Any,
) -> Dict[str, float]:
    fluxes = _all_plaquette_fluxes_from_payload(payload)
    if not fluxes:
        return {}
    outputs = summary.setdefault("outputs", {})
    grouped_fluxes = outputs.get("all_plaquette_fluxes")
    if not isinstance(grouped_fluxes, dict):
        grouped_fluxes = {}
    grouped_fluxes[str(method_key)] = fluxes
    outputs["all_plaquette_fluxes"] = grouped_fluxes
    return fluxes


def _normalize_ipeps_result_schema(result: Any) -> Dict[str, Any]:
    """Ensure the PEPS backend summary exposes the common solver fields."""
    if not isinstance(result, dict):
        result = {
            "status": "failed",
            "error": "PEPS backend returned a non-dictionary result.",
        }
    normalized = dict(result)
    normalized["status"] = str(normalized.get("status", "failed"))
    if "energy_per_site" not in normalized:
        normalized["energy_per_site"] = normalized.get("ground_state_energy_per_site")
    if "ground_state_energy_per_site" not in normalized:
        normalized["ground_state_energy_per_site"] = normalized.get("energy_per_site")
    plaquette_flux = normalized.get("plaquette_flux")
    if not isinstance(plaquette_flux, dict):
        plaquette_flux = {
            "available": False,
            "value": None,
            "W_p": None,
            "reason": "PEPS backend did not return a plaquette_flux payload.",
        }
    normalized["plaquette_flux"] = plaquette_flux
    return normalized


def _compact_ipeps_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Small mirror requested by downstream JSON consumers."""
    return {
        "status": str(result.get("status", "failed")),
        "energy_per_site": result.get("energy_per_site"),
        "ground_state_energy_per_site": result.get("ground_state_energy_per_site", result.get("energy_per_site")),
        "plaquette_flux": result.get("plaquette_flux"),
        "phase_label": result.get("phase_label"),
    }


def _record_phase_scan_plaquette_fluxes(
    summary: Dict[str, Any],
    phase_scan_data: Dict[str, Any],
) -> None:
    scan_fluxes: Dict[str, Any] = {}
    for mode_key, mode_data in phase_scan_data.items():
        if not isinstance(mode_data, dict):
            continue
        rows = mode_data.get("rows")
        if not isinstance(rows, list):
            continue
        row_fluxes: Dict[str, Any] = {}
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            fluxes = _all_plaquette_fluxes_from_payload(row)
            if not fluxes:
                continue
            row_fluxes[str(row_index)] = {
                "alpha": row.get("alpha"),
                "beta": row.get("beta"),
                "all_plaquette_fluxes": fluxes,
            }
        if row_fluxes:
            scan_fluxes[str(mode_key)] = row_fluxes
    if not scan_fluxes:
        return
    outputs = summary.setdefault("outputs", {})
    grouped_fluxes = outputs.get("all_plaquette_fluxes")
    if not isinstance(grouped_fluxes, dict):
        grouped_fluxes = {}
    grouped_fluxes["phase_scan"] = scan_fluxes
    outputs["all_plaquette_fluxes"] = grouped_fluxes


def _skip_plot_step(
    summary: Dict[str, Any],
    output_folder: str,
    key: str,
    filename: str,
    reason: str,
) -> None:
    _record_output_status(summary, key, filename, "skipped_disabled", reason=reason)
    _save_summary_checkpoint(output_folder, summary)
    print(f"[output] skip disabled: {os.path.join(output_folder, filename)} :: {reason}")


def _plot_step_status(summary: Dict[str, Any], key: str) -> str | None:
    output_status = summary.get("output_status", {})
    if not isinstance(output_status, dict):
        return None
    item = output_status.get(key)
    if not isinstance(item, dict):
        return None
    status = item.get("status")
    return str(status) if status is not None else None


def _record_combined_plot_alias(
    summary: Dict[str, Any],
    output_folder: str,
    key: str,
    combined_filename: str,
    reason: str,
) -> None:
    _record_output_status(
        summary,
        key,
        combined_filename,
        "saved_in_combined_plot",
        reason=reason,
    )
    _save_summary_checkpoint(output_folder, summary)
    print(f"[output] combined overlay: {os.path.join(output_folder, combined_filename)} :: {reason}")


def _save_plot_step(
    summary: Dict[str, Any],
    output_folder: str,
    key: str,
    filename: str,
    save_callable: Callable[[str], None],
    overwrite_existing: bool,
    continue_on_plot_error: bool,
) -> None:
    filepath = os.path.join(output_folder, filename)
    if os.path.exists(filepath) and not overwrite_existing:
        _record_output_status(summary, key, filename, "skipped_exists")
        _save_summary_checkpoint(output_folder, summary)
        print(f"[output] skip existing: {filepath}")
        return

    try:
        with profile_stage("plotting"):
            save_callable(filepath)
        _record_output_status(summary, key, filename, "saved")
        _save_summary_checkpoint(output_folder, summary)
        print(f"[output] saved: {filepath}")
    except Exception as exc:
        optional_dependency_missing = isinstance(exc, (ImportError, ModuleNotFoundError))
        status = "skipped_optional_dependency" if optional_dependency_missing else "failed"
        _record_output_status(summary, key, filename, status, str(exc))
        if optional_dependency_missing:
            warnings_payload = summary.setdefault("plot_output_warnings", {})
            if isinstance(warnings_payload, dict):
                warnings_payload.setdefault("keys", [])
                if isinstance(warnings_payload["keys"], list) and key not in warnings_payload["keys"]:
                    warnings_payload["keys"].append(key)
                warnings_payload["missing_dependency"] = exc.__class__.__name__
                warnings_payload["note"] = (
                    "A requested plot could not be written because an optional plotting "
                    "dependency is unavailable in the active Python environment."
                )
        _save_summary_checkpoint(output_folder, summary)
        if optional_dependency_missing:
            print(f"[output] skip optional dependency: {filepath} :: {exc}")
        else:
            print(f"[output] failed: {filepath} :: {exc}")
        if not continue_on_plot_error and not optional_dependency_missing:
            raise


_SPLIT_MODULE_BINDINGS_ACTIVE = False


def _load_tenpy_backend_module() -> Any:
    """Load the local TeNPy dense Yao-Lee backend/template."""
    import importlib

    return importlib.import_module("tenpy_backend")


def _optional_dependency_missing_callable(symbol_name: str, module_name: str, reason: str) -> Callable[..., Any]:
    def _missing(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            f"Optional module '{module_name}' is unavailable, so '{symbol_name}' cannot run: {reason}"
        )

    return _missing


def _bind_split_module_implementations() -> None:
    """Import the canonical implementation modules behind this CLI."""
    global _SPLIT_MODULE_BINDINGS_ACTIVE
    if _SPLIT_MODULE_BINDINGS_ACTIVE:
        return

    try:
        import analysis as analysis_tools
        import models as model_defs
        import plot_outputs
        import tenax_backend as tenax_backend_impl
    except Exception as exc:
        raise RuntimeError(f"Could not import the split implementation modules: {exc}") from exc

    ed_backend_impl = None
    ed_backend_import_error: str | None = None
    try:
        import ed_backend as ed_backend_impl_candidate

        ed_backend_impl = ed_backend_impl_candidate
    except Exception as exc:
        ed_backend_import_error = str(exc)
        print(
            "[optional] skip ed_backend.py import: "
            f"{ed_backend_import_error}. Standard sparse ED / finite-temperature ED will be skipped if requested."
        )

    ed_backend_names = (
        "estimate_sz_conserved_dimension",
        "estimate_spin_orbital_u1_dimension",
        "build_sz_conserved_basis",
        "build_spin_orbital_u1_basis",
        "build_sparse_hamiltonian_spin_orbital_u1",
        "run_spin_orbital_u1_exact_spectrum",
        "build_spin_pi_z_operator_in_spin_orbital_u1_basis",
        "build_fused_translation_operator_in_spin_orbital_u1_basis",
        "build_combined_c3_operator_in_spin_orbital_u1_basis",
        "run_spin_orbital_projected_exact_spectrum",
        "build_sparse_hamiltonian_sz_conserved",
        "run_sz_conserved_exact_spectrum",
        "plaquette_flux_from_spin_orbital_u1_ed_state",
        "collect_correlation_matrices_from_spin_orbital_u1_ed",
        "build_spin_orbital_u1_scalar_correlations",
        "collect_uniform_z_observables_from_sz_conserved_ed_state",
        "plaquette_flux_from_sz_conserved_ed_state",
        "collect_correlation_matrices_from_sz_conserved_ed",
        "build_sz_conserved_scalar_correlations",
        "all_bond_energies_sz_conserved",
        "build_exact_hamiltonian",
        "run_small_cluster_exact_spectrum",
        "run_small_cluster_exact_diagonalization",
        "collect_uniform_z_observables_from_ed_state",
        "plaquette_flux_from_ed_state",
        "collect_correlation_matrices_from_ed",
        "build_spin_orbital_scalar_correlations",
        "all_bond_energies",
        "run_finite_temperature_ed",
    )
    bindings: Dict[str, Any] = {}
    module_bindings: List[Tuple[Any, Tuple[str, ...]]] = [
        (
            analysis_tools,
            (
                "get_tenax_api",
                "_get_tqdm",
                "_make_progress_bar",
                "_start_stage",
                "_end_stage",
                "configure_profiling_from_args",
                "finalize_profiling",
                "profile_stage",
                "profiling_enabled",
                "profile_scan_points_enabled",
                "record_scan_point_timing",
                "update_profile_metadata",
                "_entropy_from_probabilities",
                "_entropy_dict_from_singular_values",
                "_summarize_entropy_values",
                "_build_entropy_profile",
                "compute_tenax_finite_mps_entropy_profile",
                "compute_tenpy_finite_mps_entropy_profile",
                "compute_ed_entropy_profile_from_state",
                "compute_tenax_infinite_mps_entropy_profile",
                "build_zero_temperature_dmrg_reference",
                "select_geometric_center_site",
                "build_reference_site_correlation_patterns",
                "resolve_low_energy_spectrum",
                "find_dmrg_excited_state",
                "DEFAULT_PHASE_CLASSIFIER_THRESHOLDS",
                "phase_classifier_thresholds_from_args",
                "_phase_scan_axis_values",
                "_phase_scan_grid_from_args",
                "_random_unit_vectors",
                "_classical_operator_value",
                "_classical_bond_energy_value",
                "_classical_total_energy",
                "_classical_bond_rows",
                "_classical_vector_structure_factor",
                "_classical_structure_factor_rows",
                "_run_classical_product_ground_state",
                "_dominant_structure_channel",
                "_bond_energy_diagnostics",
                "extract_all_plaquette_fluxes",
                "_phase_observable_diagnostics",
                "_classify_phase_from_diagnostics",
                "_phase_scan_quantum_point",
                "_phase_scan_classical_point",
                "run_alpha_beta_phase_scan",
                "validate_yao_lee_symmetry_case",
                "validate_yao_lee_symmetry_rules",
            ),
        ),
        (
            model_defs,
            (
                "ModelSpec",
                "Bond",
                "GeometryData",
                "build_model_spec",
                "is_trivial_orbital",
                "_normalize_symmetry_mode",
                "_is_u1_symmetry_mode",
                "_m2_values_from_spin_value",
                "_get_z2_symmetry_object",
                "_encode_u1_charge_pair",
                "_u1_charge_encoding_summary",
                "_u1_encoded_phys_charges_for_model",
                "_u1_sz_phys_charges_for_model",
                "_u1_tz_phys_charges_for_model",
                "_u1_phys_charges_for_model",
                "_z2_phys_charges_for_model",
                "_u1_basis_charge_table_for_model",
                "_z2_basis_charge_table_for_model",
                "_u1_encoded_target_charge",
                "_u1_target_charge_for_mode",
                "_operator_charge_transfer",
                "_validate_symmetry_conserving_terms",
                "analyze_hamiltonian_symmetries",
                "build_spin_operators",
                "build_site_ops",
                "build_yao_lee_site_ops",
                "build_spin_only_bond_terms",
                "model_terms_for_bond",
                "_normalize_external_field_treatment",
                "_normalize_external_field_axis",
                "external_field_vector",
                "resolve_field_vector",
                "classify_external_field",
                "yao_lee_conserved_symmetries",
                "normalize_requested_symmetry_reductions",
                "external_field_is_active",
                "external_field_terms_for_model",
                "validate_external_field_symmetry_compatibility",
                "external_field_filename_label",
                "external_field_display_label",
                "external_field_construction_summary",
                "_is_zero_coefficient",
                "_real_scalar_if_close",
                "nonzero_bond_terms",
                "nonzero_auto_mpo_terms",
                "_u1_pair_terms_for_bond_terms",
                "auto_mpo_pair_terms_for_bond_terms",
                "honeycomb_real_space_position",
                "snake_y_values",
                "honeycomb_plaquette_flux_operators",
                "select_honeycomb_plaquette_flux_operator",
                "plaquette_flux_close_to_target",
                "build_honeycomb_cylinder_geometry",
                "square_real_space_position",
                "build_square_cylinder_geometry",
                "triangular_bravais_vectors",
                "triangular_real_space_position",
                "build_triangular_cylinder_geometry",
                "build_lattice_geometry",
                "kron_all",
                "build_global_operator_cache",
                "build_global_operator_cache_for_model",
                "build_exact_hamiltonian",
                "run_small_cluster_exact_spectrum",
                "run_small_cluster_exact_diagonalization",
                "one_point_expectation_from_state",
                "collect_uniform_z_observables_from_ed_state",
                "plaquette_flux_from_ed_state",
                "two_point_expectation_from_state",
                "collect_correlation_matrices_from_ed",
                "build_spin_orbital_scalar_correlations",
                "bond_energy_from_correlations",
                "all_bond_energies",
                "mps_path_quality",
                "lattice_display_name",
                "_safe_filename_token",
                "_rep_filename_token",
                "model_simplified_name",
                "model_display_short_name",
                "geometry_size_filename_label",
                "geometry_size_display_label",
                "run_output_prefix",
                "run_title_label",
                "labeled_output_filename",
                "reciprocal_lattice_vectors",
                "default_high_symmetry_momenta",
                "structure_factor_from_scalar_correlation",
                "all_high_symmetry_structure_factors",
                "finite_temperature_grid",
                "run_finite_temperature_ed",
            ),
        ),
        (
            tenax_backend_impl,
            (
                "_build_auto_mpo_from_terms",
                "_empty_tenax_hamiltonian_message",
                "build_tenax_model_mpo",
                "build_tenax_yao_lee_mpo",
                "_extract_dmrg_result",
                "_sz_zero_product_state_indices_from_charges",
                "_build_product_symmetric_mps",
                "_build_random_symmetric_mps",
                "run_tenax_cylindrical_dmrg",
                "_build_dense_bulk_mpo_tensor",
                "build_idmrg_bulk_mpo_from_finite_mpo",
                "run_tenax_idmrg_x_from_finite_mpo",
                "evaluate_expectation_value",
                "collect_uniform_z_observables_from_tenax",
                "collect_correlation_matrices_from_tenax",
            ),
        ),
        (
            plot_outputs,
            (
                "titled_for_run",
                "ensure_folder_exists",
                "_geometry_positions",
                "_bond_i_j_gamma",
                "save_geometry_diagram",
                "save_bond_energy_diagram",
                "save_structure_factor_plot",
                "save_scalar_correlation_heatmaps",
                "plot_real_space_pattern",
                "save_real_space_pattern_diagram",
                "save_phase_representative_pattern",
                "save_flux_crystal_pattern",
                "save_multi_method_energy_comparison",
                "plot_peps_vs_ed_comparison",
                "save_peps_vs_ed_comparison",
                "save_entropy_profiles_comparison",
                "save_entropy_method_means_comparison",
                "save_dmrg_ed_energy_comparison",
                "save_low_energy_spectrum_comparison",
                "save_multi_method_structure_comparison",
                "save_dmrg_ed_structure_comparison",
                "save_finite_temperature_observables_plot",
                "save_finite_temperature_correlations_plot",
                "save_finite_temperature_structure_factors_plot",
                "save_phase_observable_heatmap",
                "save_phase_diagram_plot",
                "save_energy_b_scan_plot",
            ),
        ),
    ]
    if ed_backend_impl is not None:
        module_bindings.append((ed_backend_impl, ed_backend_names))
    elif ed_backend_import_error is not None:
        for name in ed_backend_names:
            bindings.setdefault(
                name,
                _optional_dependency_missing_callable(name, "ed_backend.py", ed_backend_import_error),
            )

    for module, names in module_bindings:
        for name in names:
            if hasattr(module, name):
                bindings[name] = getattr(module, name)

    globals().update(bindings)
    _SPLIT_MODULE_BINDINGS_ACTIVE = True


def main() -> None:
    _bind_split_module_implementations()
    args = parse_command_line()
    _validate_solver_resource_args(args)
    args.backend = _normalize_backend(args.backend)
    args.method = _normalize_calculation_method(getattr(args, "method", METHOD), args.backend)
    args.ipeps_contraction_method = _normalize_ipeps_contraction_method(
        getattr(args, "ipeps_contraction_method", IPEPS_CONTRACTION_METHOD)
    )
    args.ed_backend = _normalize_ed_backend(args.ed_backend)
    symmetry_request_from_cli = bool(
        args.symmetry_reductions is not None
        or args.symmetry_mode is not None
        or getattr(args, "use_sz_conserved", None) is not None
    )
    args.symmetry_reductions = _normalize_symmetry_reductions(
        args.symmetry_reductions,
        args.symmetry_mode,
    )
    if getattr(args, "use_sz_conserved", None) is not None:
        args.symmetry_reductions = ("sz",) if bool(args.use_sz_conserved) else ("none",)
    args.symmetry_mode = _legacy_symmetry_mode_from_reductions(args.symmetry_reductions)
    args.use_sz_block = False
    args.use_tau_z_block = False
    args.use_z2_block = False
    raw_phase_scan_mode = args.phase_scan_mode
    args.phase_scan_mode = _normalize_phase_scan_mode(args.phase_scan_mode)
    if args.phase_scan_mode == "classical":
        args.phase_scan_quantum_methods = []
    else:
        args.phase_scan_quantum_methods = _normalize_phase_scan_quantum_methods(
            args.phase_scan_methods,
            raw_phase_scan_mode,
        )
    args.phase_scan_methods = _selected_phase_scan_methods(
        args.phase_scan_mode,
        args.phase_scan_quantum_methods,
    )
    if bool(args.phase_scan_only):
        # Phase-scan-only has priority: it is the scan/plot workflow, with the
        # single-point workflow skipped later.
        args.phase_diagram = True
        args.run_phase_scan = True
        args.plot_phase_scan = True
    if args.backend == "quimb" and args.method in ("peps", "ipeps"):
        # quimb PEPS/iPEPS paths are benchmark workflows: always attempt the
        # finite ED reference, then let the existing ED eligibility caps decide
        # whether the selected cluster is feasible.
        args.run_ed = True
    elif bool(args.phase_diagram):
        # The phase-diagram switch is the combined user-facing option. Turning
        # it on always means calculate, plot, and save the phase diagrams.
        args.run_phase_scan = True
        args.plot_phase_scan = True
    else:
        if args.run_phase_scan is None:
            args.run_phase_scan = False
        if args.plot_phase_scan is None:
            args.plot_phase_scan = bool(args.run_phase_scan)
    args.output_folder = _resolve_output_folder(args.output_folder)
    args.profile_output_folder = _resolve_output_folder(
        getattr(args, "profile_output_folder", PROFILE_OUTPUT_FOLDER)
    )
    ensure_folder_exists(args.output_folder)
    configure_profiling_from_args(args)
    update_profile_metadata(
        requested_backend=args.backend,
        requested_method=args.method,
        requested_ed_backend=args.ed_backend,
        requested_ed_symmetry_engine=args.ed_symmetry_engine,
        output_folder=args.output_folder,
        profile_output_folder=args.profile_output_folder,
    )
    print(f"[output] folder: {args.output_folder}")
    show_progress = bool(args.progress)
    overwrite_existing = bool(args.overwrite_plots)
    continue_on_plot_error = not bool(args.strict_plot_errors)
    calculate_entanglement = bool(args.calculate_entanglement)
    calculate_uniform_observables = bool(args.calculate_uniform_observables)
    calculate_bond_energies = bool(args.calculate_bond_energies)
    calculate_structure_factors = bool(args.calculate_structure_factors)
    calculate_real_space_patterns = bool(
        args.calculate_real_space_patterns
        or args.plot_real_space_patterns
        or args.plot_bond_energies
    )
    calculate_correlations = bool(
        args.calculate_correlations
        or calculate_bond_energies
        or calculate_structure_factors
        or calculate_real_space_patterns
    )
    observable_controls = {
        "calculate": {
            "correlations": bool(calculate_correlations),
            "correlations_requested": bool(args.calculate_correlations),
            "bond_energies": bool(calculate_bond_energies),
            "structure_factors": bool(calculate_structure_factors),
            "entanglement": bool(calculate_entanglement),
            "uniform_observables": bool(calculate_uniform_observables),
            "real_space_patterns": bool(calculate_real_space_patterns),
            "reference_site_idx": args.reference_site_idx,
            "note": (
                "Bond energies, structure factors, and real-space patterns require correlation matrices; "
                "bond-energy plots also request the compact spin-vector overlay, "
                "so calculate_correlations is promoted to true when any of them is enabled."
            ),
        },
        "plot": {
            "geometry": bool(args.plot_geometry),
            "bond_energies": bool(args.plot_bond_energies),
            "structure_factors": bool(args.plot_structure_factors),
            "correlation_heatmaps": bool(args.plot_correlation_heatmaps),
            "real_space_patterns": bool(args.plot_real_space_patterns),
            "entanglement": bool(args.plot_entanglement),
            "energy_comparison": bool(args.plot_energy_comparison),
            "low_energy_spectrum": bool(args.plot_low_energy_spectrum),
            "finite_temperature": bool(args.plot_finite_temperature),
            "phase_scan": bool(args.plot_phase_scan),
        },
        "phase_diagram": {
            "enabled": bool(args.phase_diagram),
            "run": bool(args.run_phase_scan),
            "plot": bool(args.plot_phase_scan),
            "channels": str(args.phase_scan_channels),
            "mode": str(args.phase_scan_mode),
            "quantum_methods": list(args.phase_scan_quantum_methods),
            "selected_outputs": list(args.phase_scan_methods),
            "note": (
                "phase_scan_only has priority and promotes phase_diagram on; "
                "phase_diagram=true always runs, plots, and saves the selected phase scans."
            ),
        },
    }
    lattice_name = str(args.lattice).lower()
    circumference_x = bool(args.circumference_x)
    circumference_y = bool(args.circumference_y)
    args.symmetry_mode = _normalize_symmetry_mode(args.symmetry_mode)
    with profile_stage("model_spec construction"):
        model_spec = build_model_spec(
            spin_rep=args.spin_rep,
            orbital_rep=args.orbital_rep,
            model_family=args.model_family,
            ising_axis=args.ising_axis,
        )
    # Normalize legacy alias "1" -> "0" in recorded parameters.
    args.orbital_rep = model_spec.orbital_rep
    with profile_stage("external field construction"):
        args.external_field_treatment = _normalize_external_field_treatment(args.external_field_treatment)
        args.external_field_axis = _normalize_external_field_axis(args.external_field_axis)
        resolved_field_vector = external_field_vector(
            axis=args.external_field_axis,
            strength=args.external_field_strength,
            hx=args.field_hx,
            hy=args.field_hy,
            hz=args.field_hz,
        )
        hamiltonian_external_field_terms = (
            external_field_terms_for_model(
                resolved_field_vector,
                mu_b=args.mu_b,
                field_sign=args.field_sign,
                sigma_factor=args.field_sigma_factor,
            )
            if args.external_field_treatment == "hamiltonian"
            else []
        )
    model_symmetry_selection = normalize_requested_symmetry_reductions(
        args.symmetry_reductions,
        model_spec,
        args.external_field_treatment,
        resolved_field_vector,
        backend=args.backend,
        strict=bool(args.strict_symmetry_selection_rules),
        allow_dense_fallback=bool(args.symmetry_allow_dense_fallback),
        requested_from_default=not symmetry_request_from_cli,
        target_sz2=int(args.u1_target_sz2),
        target_tz2=int(args.u1_target_tz2),
        z2_target_parity=int(args.z2_target_parity),
    )
    if model_symmetry_selection.get("errors"):
        raise ValueError("; ".join(str(item) for item in model_symmetry_selection["errors"]))
    args.model_symmetry_selection = model_symmetry_selection
    args.symmetry_reductions = tuple(model_symmetry_selection.get("safe_reductions", ["none"]))
    args.symmetry_mode = _normalize_symmetry_mode(_legacy_symmetry_mode_from_reductions(args.symmetry_reductions))
    if show_progress:
        for warning in model_symmetry_selection.get("warnings", []):
            print(f"[symmetry] {warning}")
    try:
        validate_external_field_symmetry_compatibility(
            hamiltonian_external_field_terms,
            symmetry_mode=args.symmetry_mode,
            model_family=model_spec.model_family,
            external_field_treatment=args.external_field_treatment,
            z2_generator=model_symmetry_selection.get("z2_generator"),
        )
    except ValueError as exc:
        if not bool(args.symmetry_allow_dense_fallback):
            raise
        if show_progress:
            print(f"[symmetry] external-field symmetry warning: {exc}")
    args.u1_target_sz2 = int(args.u1_target_sz2)
    args.u1_target_tz2 = int(args.u1_target_tz2)
    args.z2_target_parity = int(args.z2_target_parity) % 2
    args.use_translation_x_block = bool(args.use_translation_x_block)
    args.use_translation_y_block = bool(args.use_translation_y_block)
    args.momentum_x_block = int(args.momentum_x_block)
    args.momentum_y_block = int(args.momentum_y_block)
    args.reflection_block = int(args.reflection_block)
    args.ed_symmetry_engine = _normalize_ed_symmetry_engine(args.ed_symmetry_engine)
    args.ed_quspin_experimental_fused_translation = bool(args.ed_quspin_experimental_fused_translation)
    args.ed_c3_mode = str(args.ed_c3_mode).strip().lower()
    args.ed_c3_q_blocks = str(args.ed_c3_q_blocks).strip().lower()
    args.ed_z2_mode = str(args.ed_z2_mode).strip().lower()
    args.ed_z2_kind = str(args.ed_z2_kind).strip().lower()
    args.phase_scan_ed_max_sites = (
        int(args.max_ed_sites)
        if args.phase_scan_ed_max_sites is None
        else int(args.phase_scan_ed_max_sites)
    )
    args.phase_scan_ed_max_hilbert_dim = (
        int(args.max_ed_hilbert_dim)
        if args.phase_scan_ed_max_hilbert_dim is None
        else int(args.phase_scan_ed_max_hilbert_dim)
    )
    symmetry_reduction_settings = _symmetry_reduction_settings_from_report(args, None)
    args.ed_symmetry_plan = {"status": "pending", "reason": "geometry has not been built yet"}
    quspin_ed_settings = {
        "symmetry_reductions": symmetry_reduction_settings,
        "ed_symmetry_plan": args.ed_symmetry_plan,
        "check_symmetries": bool(args.quspin_check_symmetries),
        "check_hermiticity": bool(args.quspin_check_hermiticity),
        "check_particle_conservation": bool(args.quspin_check_particle_conservation),
    }

    def refresh_symmetry_reduction_settings(report: Dict[str, Any] | None) -> Dict[str, Any]:
        nonlocal symmetry_reduction_settings, quspin_ed_settings
        symmetry_reduction_settings = _symmetry_reduction_settings_from_report(args, report)
        args.use_sz_block = bool(symmetry_reduction_settings.get("use_sz_block", False))
        args.use_tau_z_block = bool(symmetry_reduction_settings.get("use_tau_z_block", False))
        args.use_z2_block = bool(symmetry_reduction_settings.get("use_z2_block", False))
        args.use_sz_conserved = bool(args.use_sz_block)
        quspin_ed_settings = {
            "symmetry_reductions": symmetry_reduction_settings,
            "ed_symmetry_plan": getattr(args, "ed_symmetry_plan", {"status": "pending"}),
            "check_symmetries": bool(args.quspin_check_symmetries),
            "check_hermiticity": bool(args.quspin_check_hermiticity),
            "check_particle_conservation": bool(args.quspin_check_particle_conservation),
        }
        return symmetry_reduction_settings

    def _geometry_cache_key(geometry_obj: Any) -> Tuple[Any, ...]:
        cell_indices = tuple(
            tuple(int(value) for value in cell)
            for cell in list(getattr(geometry_obj, "cell_indices", []))
        )
        sublattice_indices = tuple(
            int(value) for value in list(getattr(geometry_obj, "sublattice_indices", []))
        )
        return (
            int(getattr(geometry_obj, "number_of_sites", 0)),
            int(getattr(geometry_obj, "length_x", args.length_x)),
            int(getattr(geometry_obj, "length_y", args.length_y)),
            bool(getattr(geometry_obj, "circumference_x", circumference_x)),
            bool(getattr(geometry_obj, "circumference_y", circumference_y)),
            cell_indices,
            sublattice_indices,
        )

    def _hashable_cache_value(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((str(key), _hashable_cache_value(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(_hashable_cache_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(_hashable_cache_value(item) for item in value))
        return value

    ed_symmetry_plan_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def refresh_ed_symmetry_plan(geometry_obj: Any) -> Dict[str, Any]:
        nonlocal quspin_ed_settings
        with profile_stage("ED symmetry plan resolution"):
            cache_key = (
                _geometry_cache_key(geometry_obj),
                model_spec,
                tuple(args.symmetry_reductions),
                str(args.ed_backend),
                str(args.ed_symmetry_engine),
                bool(args.ed_quspin_experimental_fused_translation),
                str(args.ed_c3_mode),
                str(args.ed_c3_q_blocks),
                str(args.ed_z2_mode),
                str(args.ed_z2_kind),
                bool(args.use_translation_x_block),
                bool(args.use_translation_y_block),
                int(args.momentum_x_block),
                int(args.momentum_y_block),
                int(args.z2_target_parity),
                int(args.u1_target_sz2),
                int(args.u1_target_tz2),
                tuple(float(value) for value in resolved_field_vector),
                tuple((float(coefficient), str(op_name)) for coefficient, op_name in hamiltonian_external_field_terms),
                _hashable_cache_value(symmetry_reduction_settings),
            )
            cached_plan = ed_symmetry_plan_cache.get(cache_key)
            if cached_plan is None:
                cached_plan = _resolve_ed_symmetry_plan(
                    args=args,
                    model_spec_obj=model_spec,
                    geometry_obj=geometry_obj,
                    resolved_field_vector=resolved_field_vector,
                    hamiltonian_field_terms=hamiltonian_external_field_terms,
                    shared_symmetry_settings=symmetry_reduction_settings,
                )
                ed_symmetry_plan_cache[cache_key] = copy.deepcopy(cached_plan)
            args.ed_symmetry_plan = copy.deepcopy(cached_plan)
            quspin_ed_settings = dict(quspin_ed_settings)
            quspin_ed_settings["symmetry_reductions"] = symmetry_reduction_settings
            quspin_ed_settings["ed_symmetry_plan"] = args.ed_symmetry_plan
            return args.ed_symmetry_plan

    def attach_symmetry_dimension_report(report: Dict[str, Any], geometry_obj: Any) -> Dict[str, Any]:
        dimension_report = _symmetry_hilbert_dimension_report(
            geometry_obj,
            model_spec,
            symmetry_reduction_settings,
        )
        report["hilbert_space_dimension"] = dimension_report
        return dimension_report

    refresh_symmetry_reduction_settings(None)
    symmetry_preflight_report: Dict[str, Any] | None = None
    effective_symmetry_mode = str(args.symmetry_mode)

    def _run_symmetry_preflight_for_geometry_impl(geometry_obj: Any) -> Dict[str, Any]:
        if not bool(args.symmetry_precheck):
            disabled_effective = "none" if str(args.symmetry_mode) == "auto" else str(args.symmetry_mode)
            report = {
                "status": "disabled",
                "requested_mode": str(args.symmetry_mode),
                "requested_reductions": list(args.symmetry_reductions),
                "model_aware_selection": getattr(args, "model_symmetry_selection", None),
                "effective_mode_for_tenax": disabled_effective,
            }
            refresh_symmetry_reduction_settings(report)
            ed_plan = refresh_ed_symmetry_plan(geometry_obj)
            attach_symmetry_dimension_report(report, geometry_obj)
            report["effective_reductions"] = {
                "sz": bool(args.use_sz_block),
                "tz": bool(args.use_tau_z_block),
                "z2": bool(args.use_z2_block),
            }
            report["ed_symmetry_plan"] = ed_plan
            return report
        report = analyze_hamiltonian_symmetries(
            geometry=geometry_obj,
            model_spec=model_spec,
            alpha=args.alpha,
            beta=args.beta,
            coupling_j=args.coupling_j,
            jx=args.jx,
            jy=args.jy,
            jz=args.jz,
            external_field_terms=hamiltonian_external_field_terms,
            requested_symmetry_mode=args.symmetry_mode,
            u1_target_total_sz2=args.u1_target_sz2,
            u1_target_total_tz2=args.u1_target_tz2,
            z2_target_parity=args.z2_target_parity,
        )
        report["status"] = "completed"
        report["requested_reductions"] = list(args.symmetry_reductions)
        report["model_aware_selection"] = getattr(args, "model_symmetry_selection", None)
        failures: List[str] = []
        requested_mode = str(report.get("requested_mode", args.symmetry_mode))
        reduction_set = set(args.symmetry_reductions)
        if "auto" not in reduction_set and "none" not in reduction_set:
            if {"sz", "tz"}.issubset(reduction_set):
                if not _u1_report_supports_sector(report, "u1"):
                    failures.append(
                        "requested combined reductions sz+tz require simultaneous conservation of total Sz "
                        "and total tau_z with reachable target sectors"
                    )
            elif "sz" in reduction_set and not _u1_report_supports_sector(report, "u1_sz"):
                failures.append("requested sz reduction is not conserved or its target sector is unreachable")
            elif "tz" in reduction_set and not _u1_report_supports_sector(report, "u1_tz"):
                failures.append("requested tz reduction is not conserved or its target sector is unreachable")
            if "z2" in reduction_set and not _z2_report_supports_sector(report):
                failures.append("requested z2 reduction is not conserved or its target sector is unreachable")
        recommended_mode = str(report.get("recommended_mode_for_tenax", "none"))
        effective_mode = requested_mode
        if requested_mode == "auto":
            effective_mode = recommended_mode
        elif requested_mode == "z2" and "z2" in reduction_set:
            effective_mode = "none"
        elif failures and bool(args.symmetry_allow_dense_fallback):
            effective_mode = recommended_mode
        report["effective_mode_for_tenax"] = effective_mode
        report["strict_precheck_failures"] = failures
        refresh_symmetry_reduction_settings(report)
        ed_plan = refresh_ed_symmetry_plan(geometry_obj)
        dimension_report = attach_symmetry_dimension_report(report, geometry_obj)
        report["effective_reductions"] = {
            "sz": bool(args.use_sz_block),
            "tz": bool(args.use_tau_z_block),
            "z2": bool(args.use_z2_block),
        }
        report["ed_symmetry_plan"] = ed_plan
        if show_progress:
            u1_info = report.get("u1", {}) if isinstance(report.get("u1"), dict) else {}
            u1_sz_info = report.get("u1_sz", {}) if isinstance(report.get("u1_sz"), dict) else {}
            u1_tz_info = report.get("u1_tz", {}) if isinstance(report.get("u1_tz"), dict) else {}
            z2_info = report.get("z2", {}) if isinstance(report.get("z2"), dict) else {}
            applied_shared_reductions = [
                name for name, enabled in report["effective_reductions"].items() if bool(enabled)
            ]
            dropped_shared_reductions = list(
                model_symmetry_selection.get("dropped_reductions", [])
                if isinstance(model_symmetry_selection, dict)
                else []
            )
            print(
                "[symmetry] model rules: "
                f"method={getattr(args, 'backend', 'auto')}/{getattr(args, 'method', 'auto')}, "
                f"requested={list(args.symmetry_reductions)}, "
                f"conserved={{Sz:{bool(u1_sz_info.get('conserved_total_Sz', False))}, "
                f"Tz:{bool(u1_tz_info.get('conserved_total_Tz', False))}, "
                f"Z2:{bool(z2_info.get('conserved_global_parity', False))}}}, "
                f"eligible={applied_shared_reductions}, "
                f"dropped_by_model={dropped_shared_reductions}, "
                f"field_class={model_symmetry_selection.get('field_class') if isinstance(model_symmetry_selection, dict) else None}, "
                f"full_dim={int(dimension_report['full_hilbert_dimension']):,}, "
                f"model_effective_dim={int(dimension_report['effective_hilbert_dimension']):,}, "
                f"model_reduction={_dimension_ratio_text(dimension_report)}, "
                f"basis={dimension_report['basis_label']}"
            )
            if bool(ed_plan.get("use_tau_z_block", False)):
                ed_u1_dim = _spin_orbital_symmetry_reduced_dimension(
                    int(geometry_obj.number_of_sites),
                    False,
                    int(ed_plan.get("target_sz2", args.u1_target_sz2)),
                    True,
                    int(ed_plan.get("target_tz2", args.u1_target_tz2)),
                )
                ed_projector_factor = _ed_projector_reduction_factor_estimate(ed_plan, geometry_obj)
                ed_projector_dim = int(max(1, int(ed_u1_dim) // max(1, ed_projector_factor)))
                ed_total_reduction = (
                    float(dimension_report["full_hilbert_dimension"]) / float(ed_projector_dim)
                    if int(ed_projector_dim) > 0
                    else None
                )
                ed_engine_label = str(ed_plan.get("effective_engine", ed_plan.get("engine", args.ed_symmetry_engine)))
                ed_dropped = [item.get("name") for item in ed_plan.get("dropped_symmetries", [])]
                print(
                    "[ed-symmetry] plan: "
                    f"backend={ed_engine_label}, "
                    f"status={_ed_symmetry_status_text(ed_plan)}, "
                    f"dropped={ed_dropped}, "
                    f"field_class={ed_plan.get('actual_hamiltonian_field_class')}, "
                    f"full_dim={int(dimension_report['full_hilbert_dimension']):,}, "
                    f"Tz_parent_dim={int(ed_u1_dim):,}, "
                    f"projected_dim_estimate={int(ed_projector_dim):,}, "
                    f"Tz_reduction={float(dimension_report['full_hilbert_dimension']) / float(ed_u1_dim):.3f}x, "
                    f"projector_reduction~={ed_projector_factor}x, "
                    f"total_reduction~={(f'{ed_total_reduction:.3f}x' if ed_total_reduction is not None else 'unreachable')}"
                )
            if failures:
                print(f"[symmetry] strict precheck issue: {'; '.join(failures)}")
        if (
            failures
            and bool(args.strict_symmetry_precheck)
            and not bool(args.symmetry_allow_dense_fallback)
        ):
            raise ValueError(
                "Strict symmetry precheck failed: "
                + "; ".join(failures)
                + ". Use conserving sectors, switch --symmetry-reductions none, "
                "or pass --symmetry-allow-dense-fallback to continue without symmetry speedup."
            )
        return report

    def run_symmetry_preflight_for_geometry(geometry_obj: Any) -> Dict[str, Any]:
        with profile_stage("symmetry precheck"):
            return _run_symmetry_preflight_for_geometry_impl(geometry_obj)

    # Keep outputs consistent with current settings: if the output folder already
    # contains results from a different parameter set, auto-overwrite stale plots.
    previous_summary = _load_previous_summary(args.output_folder)
    if (
        (not overwrite_existing)
        and previous_summary is not None
        and isinstance(previous_summary.get("parameters"), dict)
    ):
        previous_signature = _extract_consistency_signature(previous_summary["parameters"])
        current_signature = _extract_consistency_signature(vars(args))
        if previous_signature != current_signature:
            overwrite_existing = True
            print(
                "[output] Detected existing plots from a different parameter set; "
                "auto-enabling overwrite for consistency."
            )

    tenpy_allowed_symmetry_modes = ("auto", "u1_tz", "none")

    def tenpy_backend_compatibility_issue() -> str | None:
        if lattice_name != "honeycomb":
            return (
                "TeNPy backend currently supports only --lattice honeycomb."
            )
        if args.symmetry_mode not in tenpy_allowed_symmetry_modes:
            return (
                "The local TeNPy backend uses the Yao-Lee total-Tz U1 implementation and supports "
                "only --symmetry-mode u1_tz, auto, or none."
            )
        if not (
            model_spec.spin_rep == "1/2"
            and model_spec.orbital_rep == "1/2"
            and model_spec.model_family == "yao_lee"
            and model_spec.ising_axis == "z"
        ):
            return (
                "TeNPy backend currently supports only the legacy default model "
                "(spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z)."
            )
        return None

    tenpy_backend_issue = tenpy_backend_compatibility_issue()
    if args.backend == "tenpy" and tenpy_backend_issue is not None:
        raise ValueError(
            f"{tenpy_backend_issue} Use --backend tenax for unsupported lattices/models "
            "or dense non-symmetric experiments."
        )

    geometry = None
    dmrg_info: Dict[str, Any] = {}
    dmrg_energy = 0.0
    dmrg_scalar_correlations: Dict[str, np.ndarray] = {}
    dmrg_real_space_patterns: Dict[str, Any] = {}
    dmrg_bond_rows: List[Dict[str, Any]] = []
    dmrg_structure_factor_rows: List[Dict[str, Any]] = []
    dmrg_uniform_observables: Dict[str, Any] = {}
    dmrg_state_obj: Any = None
    tenax_mpo = None
    backend_used = None
    backend_warning = None
    entanglement_warning: str | None = None
    entropy_profiles: Dict[str, Dict[str, Any]] = {}
    ed_spectrum: Dict[str, Any] | None = None
    geometry_plot_status = "not_attempted"
    geometry_plot_error: str | None = None
    run_file_prefix: str | None = None
    run_plot_title_label: str | None = None
    run_summary_filename = "run_summary.json"
    output_filename_cache: Dict[str, str] = {}
    plot_title_cache: Dict[str, str] = {}

    def configure_run_output_names(geometry_obj: Any) -> None:
        nonlocal run_file_prefix, run_plot_title_label, run_summary_filename
        if run_file_prefix is not None and run_plot_title_label is not None:
            return
        run_file_prefix = run_output_prefix(
            model_spec=model_spec,
            geometry=geometry_obj,
            lattice=lattice_name,
            length_x=args.length_x,
            length_y=args.length_y,
            circumference_x=circumference_x,
            circumference_y=circumference_y,
        )
        field_file_label = external_field_filename_label(
            args.external_field_treatment,
            args.external_field_axis,
            resolved_field_vector,
        )
        if field_file_label:
            run_file_prefix = _safe_filename_token(f"{run_file_prefix}_{field_file_label}")
        run_plot_title_label = run_title_label(
            model_spec=model_spec,
            geometry=geometry_obj,
            lattice=lattice_name,
            length_x=args.length_x,
            length_y=args.length_y,
            circumference_x=circumference_x,
            circumference_y=circumference_y,
        )
        field_title_label = external_field_display_label(
            args.external_field_treatment,
            args.external_field_axis,
            resolved_field_vector,
        )
        if field_title_label:
            run_plot_title_label = f"{run_plot_title_label}\n{field_title_label}"
        run_summary_filename = labeled_output_filename(run_file_prefix, "run_summary.json")

    def output_filename(base_filename: str) -> str:
        if run_file_prefix is None:
            return base_filename
        cached = output_filename_cache.get(base_filename)
        if cached is None:
            cached = labeled_output_filename(run_file_prefix, base_filename)
            output_filename_cache[base_filename] = cached
        return cached

    def plot_title(base_title: str) -> str:
        cached = plot_title_cache.get(base_title)
        if cached is None:
            cached = titled_for_run(base_title, run_plot_title_label)
            plot_title_cache[base_title] = cached
        return cached

    def save_flux_crystal_output(
        summary_obj: Dict[str, Any],
        geometry_obj: Any,
        payload: Any,
        output_key: str,
        base_filename: str,
        title: str,
    ) -> None:
        all_fluxes = _all_plaquette_fluxes_from_payload(payload)
        if not all_fluxes:
            return
        filename = output_filename(base_filename)
        if not bool(args.plot_real_space_patterns):
            _skip_plot_step(
                summary_obj,
                args.output_folder,
                output_key,
                filename,
                "plot_real_space_patterns is false",
            )
            return
        _save_plot_step(
            summary_obj,
            args.output_folder,
            output_key,
            filename,
            lambda path, fluxes=all_fluxes, plot_text=title: save_flux_crystal_pattern(
                geometry_obj,
                fluxes,
                path,
                plot_title(plot_text),
            ),
            overwrite_existing,
            continue_on_plot_error,
        )

    def save_geometry_before_sweep(geometry_obj: Any) -> None:
        nonlocal geometry_plot_status, geometry_plot_error
        configure_run_output_names(geometry_obj)
        filename = output_filename("geometry_diagram.png")
        filepath = os.path.join(args.output_folder, filename)
        if not bool(args.plot_geometry):
            geometry_plot_status = "skipped_disabled"
            print(f"[geometry] plot disabled (plot_geometry=false); not writing {filepath}")
            return
        if os.path.exists(filepath) and not overwrite_existing:
            geometry_plot_status = "skipped_exists"
            print(f"[output] skip existing: {filepath}")
            return
        try:
            with profile_stage("plotting"):
                save_geometry_diagram(
                    geometry_obj,
                    filepath,
                    lattice_name,
                    title_label=run_plot_title_label,
                    external_field_vector=resolved_field_vector,
                )
            geometry_plot_status = "saved"
            print(f"[output] saved: {filepath}")
        except Exception as exc:
            optional_dependency_missing = isinstance(exc, (ImportError, ModuleNotFoundError))
            geometry_plot_status = "skipped_optional_dependency" if optional_dependency_missing else "failed"
            geometry_plot_error = str(exc)
            if optional_dependency_missing:
                print(f"[output] skip optional dependency: {filepath} :: {exc}")
            else:
                print(f"[output] failed: {filepath} :: {exc}")
            if not continue_on_plot_error and not optional_dependency_missing:
                raise

    def _phase_scan_representative_rows(
        rows: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        completed: List[Tuple[int, Dict[str, Any]]] = [
            (idx, row)
            for idx, row in enumerate(rows)
            if str(row.get("status", "completed")) == "completed" and "phase_label" in row
        ]
        if len(completed) == 0:
            return []
        alpha_values = [float(row["alpha"]) for _, row in completed]
        beta_values = [float(row["beta"]) for _, row in completed]
        alpha_span = max(max(alpha_values) - min(alpha_values), 1.0)
        beta_span = max(max(beta_values) - min(beta_values), 1.0)
        phase_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        for idx, row in completed:
            phase_groups.setdefault(str(row["phase_label"]), []).append((idx, row))

        representatives: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for phase_label, phase_rows in phase_groups.items():
            phase_alphas = np.asarray([float(row["alpha"]) for _, row in phase_rows], dtype=float)
            phase_betas = np.asarray([float(row["beta"]) for _, row in phase_rows], dtype=float)
            alpha_center = float(np.median(phase_alphas))
            beta_center = float(np.median(phase_betas))

            def representative_score(item: Tuple[int, Dict[str, Any]]) -> Tuple[float, float, int]:
                idx, row = item
                alpha_distance = (float(row["alpha"]) - alpha_center) / alpha_span
                beta_distance = (float(row["beta"]) - beta_center) / beta_span
                energy_value = abs(float(row.get("energy_per_site", row.get("energy", 0.0))))
                return (alpha_distance * alpha_distance + beta_distance * beta_distance, energy_value, idx)

            row_index, representative_row = min(phase_rows, key=representative_score)
            representative = {
                "phase_label": str(phase_label),
                "row_index": int(row_index),
                "alpha": float(representative_row["alpha"]),
                "beta": float(representative_row["beta"]),
                "model_beta": float(representative_row.get("model_beta", representative_row["beta"])),
                "field_strength": (
                    float(representative_row["field_strength"])
                    if representative_row.get("field_strength") is not None
                    else None
                ),
                "scan_axes": representative_row.get("scan_axes"),
                "energy": (
                    float(representative_row["energy"])
                    if "energy" in representative_row and representative_row["energy"] is not None
                    else None
                ),
                "energy_per_site": (
                    float(representative_row["energy_per_site"])
                    if "energy_per_site" in representative_row and representative_row["energy_per_site"] is not None
                    else None
                ),
                "outputs": {},
                "selection": "row closest to the median alpha/beta coordinate of this phase label",
            }
            representatives.append((representative, representative_row))
        return representatives

    def _phase_scan_representative_filename(
        mode_key: str,
        representative: Dict[str, Any],
        suffix: str,
    ) -> str:
        phase_token = _safe_filename_token(str(representative["phase_label"]).lower())
        alpha_token = _safe_filename_token(f"{float(representative['alpha']):.6g}")
        if representative.get("field_strength") is not None:
            field_token = _safe_filename_token(f"{float(representative['field_strength']):.6g}")
            beta_token = _safe_filename_token(f"{float(representative.get('model_beta', representative['beta'])):.6g}")
            return output_filename(
                f"{mode_key}_{phase_token}_alpha_{alpha_token}_B_{field_token}_modelbeta_{beta_token}_{suffix}"
            )
        beta_token = _safe_filename_token(f"{float(representative['beta']):.6g}")
        return output_filename(
            f"{mode_key}_{phase_token}_alpha_{alpha_token}_beta_{beta_token}_{suffix}"
        )

    def _effective_quspin_spatial_flags(geometry_obj: Any) -> Dict[str, Any]:
        package_available = importlib.util.find_spec("quspin") is not None
        requested_translation_x = bool(args.use_translation_x_block)
        requested_translation_y = bool(args.use_translation_y_block)
        translation_x_supported = False
        translation_y_supported = False
        translation_x_reason = None
        translation_y_reason = None
        if bool(requested_translation_x or requested_translation_y):
            if not package_available:
                reason = "QuSpin package is not installed, so translation blocks cannot be checked."
                translation_x_reason = reason if requested_translation_x else None
                translation_y_reason = reason if requested_translation_y else None
            else:
                try:
                    import quspin_backend as quspin_validation_backend

                    support = quspin_validation_backend.quspin_translation_block_support(geometry_obj)
                    x_support = support.get("x", {})
                    y_support = support.get("y", {})
                    translation_x_supported = bool(requested_translation_x and x_support.get("supported", False))
                    translation_y_supported = bool(requested_translation_y and y_support.get("supported", False))
                    translation_x_reason = x_support.get("reason") if requested_translation_x else None
                    translation_y_reason = y_support.get("reason") if requested_translation_y else None
                except Exception as exc:
                    reason = str(exc)
                    translation_x_supported = False
                    translation_y_supported = False
                    translation_x_reason = reason if requested_translation_x else None
                    translation_y_reason = reason if requested_translation_y else None
        reflection_reason = None
        if bool(args.use_reflection_block) or int(args.reflection_block) != 0:
            reflection_reason = (
                "QuSpin reflection/C3 blocks are not applied for the bond-directional Yao-Lee Hamiltonian; "
                "they can permute x/y/z bond types unless a gauge map is implemented."
            )
        return {
            "package_available": bool(package_available),
            "requested_translation_x_block": bool(requested_translation_x),
            "requested_translation_y_block": bool(requested_translation_y),
            "use_translation_block": bool(translation_x_supported or translation_y_supported),
            "use_translation_x_block": bool(translation_x_supported),
            "use_translation_y_block": bool(translation_y_supported),
            "translation_reason": {
                "x": translation_x_reason,
                "y": translation_y_reason,
            },
            "translation_x_reason": translation_x_reason,
            "translation_y_reason": translation_y_reason,
            "use_reflection_block": False,
            "reflection_reason": reflection_reason,
        }

    def _classical_representative_payload(
        geometry_obj: Any,
        row: Dict[str, Any],
        alpha: float,
        beta: float,
    ) -> Dict[str, Any]:
        spin_vectors = np.asarray(row.get("spin_vectors"), dtype=float)
        orbital_vectors = np.asarray(row.get("orbital_vectors"), dtype=float)
        if spin_vectors.ndim != 2 or orbital_vectors.ndim != 2 or spin_vectors.shape != orbital_vectors.shape:
            raise ValueError("Representative classical row is missing valid spin/orbital vectors.")
        spin_scalar = spin_vectors @ spin_vectors.T
        orbital_scalar = orbital_vectors @ orbital_vectors.T
        scalar_correlations = {
            "S": spin_scalar,
            "T": orbital_scalar,
            "ST": spin_scalar * orbital_scalar,
        }
        bond_rows = _classical_bond_rows(
            geometry_obj,
            spin_vectors,
            orbital_vectors,
            model_spec,
            float(alpha),
            float(beta),
            args.coupling_j,
            args.jx,
            args.jy,
            args.jz,
        )
        return {
            "bond_energies": bond_rows,
            "real_space_patterns": build_reference_site_correlation_patterns(
                geometry_obj,
                scalar_correlations,
                reference_site_idx=args.reference_site_idx,
            ),
            "recalculation_method": "classical_vectors_from_representative_row",
        }

    def _quantum_ed_representative_payload(
        geometry_obj: Any,
        row: Dict[str, Any],
        alpha: float,
        beta: float,
        field_terms: List[Tuple[float, str]] | None = None,
    ) -> Dict[str, Any]:
        active_field_terms = list(hamiltonian_external_field_terms if field_terms is None else field_terms)
        ed_backend_name = str(row.get("ed_backend", args.ed_backend)).strip().lower()
        if ed_backend_name == "ed":
            ed_backend_name = "standard"
        row_ed_plan = (
            row.get("ed_symmetry_plan")
            if isinstance(row.get("ed_symmetry_plan"), dict)
            else getattr(args, "ed_symmetry_plan", {})
        )
        row_uses_projector = bool(
            str(row.get("basis_type", "")) == "bitwise_spin_orbital_tz_projector_block"
            or row.get("use_translation_x_conserved", False)
            or row.get("use_translation_y_conserved", False)
            or row.get("use_c3_conserved", False)
            or (
                row.get("use_z2_conserved", False)
                and str(row.get("z2_kind", row_ed_plan.get("z2_kind") if isinstance(row_ed_plan, dict) else "")) == "spin_pi_z"
            )
        )
        if row_uses_projector and ed_backend_name == "quspin":
            ed_backend_name = "standard"
        if ed_backend_name == "quspin":
            import quspin_backend as quspin_ed_backend

            use_sz_block = bool(symmetry_reduction_settings.get("use_sz_block", False))
            use_tau_z_block = bool(symmetry_reduction_settings.get("use_tau_z_block", False))
            use_z2_block = bool(symmetry_reduction_settings.get("use_z2_block", False))
            spatial_flags = _effective_quspin_spatial_flags(geometry_obj)
            spectrum, vectors = quspin_ed_backend.run_small_cluster_exact_spectrum(
                geometry=geometry_obj,
                model_spec=model_spec,
                alpha=float(alpha),
                beta=float(beta),
                coupling_j=args.coupling_j,
                eigenstate_count=max(1, int(args.ed_max_eigenstates)),
                check_ground_state_degeneracy=False,
                jx=args.jx,
                jy=args.jy,
                jz=args.jz,
                external_field_terms=active_field_terms,
                show_progress=show_progress,
                solver=args.ed_solver,
                sparse_tol=float(args.ed_sparse_tol),
                sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
                use_sz_block=use_sz_block,
                target_sz2=int(symmetry_reduction_settings.get("target_sz2", args.u1_target_sz2)),
                use_tau_z_block=use_tau_z_block,
                target_tz2=int(symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2)),
                use_z2_block=use_z2_block,
                z2_generator=symmetry_reduction_settings.get("z2_generator"),
                z2_target_parity=int(args.z2_target_parity),
                use_translation_block=bool(spatial_flags["use_translation_block"]),
                use_translation_x_block=bool(spatial_flags["use_translation_x_block"]),
                use_translation_y_block=bool(spatial_flags["use_translation_y_block"]),
                momentum_block_1=int(args.momentum_x_block),
                momentum_block_2=int(args.momentum_y_block),
                momentum_x_block=int(args.momentum_x_block),
                momentum_y_block=int(args.momentum_y_block),
                use_reflection_block=False,
                reflection_block=0,
                check_symm=bool(args.quspin_check_symmetries),
                check_herm=bool(args.quspin_check_hermiticity),
                check_pcon=bool(args.quspin_check_particle_conservation),
            )
            basis_use_sz_block = bool(spectrum.get("use_sz_block", use_sz_block))
            basis_target_sz2 = int(spectrum.get("target_sz2", int(symmetry_reduction_settings.get("target_sz2", args.u1_target_sz2))))
            basis_use_z2_block = bool(spectrum.get("use_z2_block", use_z2_block))
            basis = quspin_ed_backend.build_quspin_yao_lee_basis(
                int(geometry_obj.number_of_sites),
                geometry=geometry_obj,
                use_sz_block=basis_use_sz_block,
                target_sz2=basis_target_sz2,
                use_tau_z_block=use_tau_z_block,
                target_tz2=int(symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2)),
                use_z2_block=basis_use_z2_block,
                z2_generator=symmetry_reduction_settings.get("z2_generator"),
                z2_target_parity=int(args.z2_target_parity),
                use_translation_block=bool(spatial_flags["use_translation_block"]),
                use_translation_x_block=bool(spatial_flags["use_translation_x_block"]),
                use_translation_y_block=bool(spatial_flags["use_translation_y_block"]),
                momentum_block_1=int(args.momentum_x_block),
                momentum_block_2=int(args.momentum_y_block),
                momentum_x_block=int(args.momentum_x_block),
                momentum_y_block=int(args.momentum_y_block),
                use_reflection_block=False,
                reflection_block=0,
            )
            scalar_correlations = quspin_ed_backend.build_spin_orbital_scalar_correlations(
                basis,
                np.asarray(vectors[:, 0], dtype=np.complex128),
                int(geometry_obj.number_of_sites),
            )
            bond_rows = quspin_ed_backend.all_bond_energies(
                geometry_obj,
                scalar_correlations,
                float(alpha),
                float(beta),
                args.coupling_j,
            )
            energy = float(spectrum.get("ground_state_energy", np.nan))
            method_label = "quspin_representative_ed"
        elif bool(symmetry_reduction_settings.get("use_tau_z_block", False)) and not bool(row.get("use_sz_conserved", False)):
            if row_uses_projector:
                spectrum, vectors, basis_list, basis_map = run_spin_orbital_projected_exact_spectrum(
                    geometry=geometry_obj,
                    model_spec=model_spec,
                    alpha=float(alpha),
                    beta=float(beta),
                    coupling_j=args.coupling_j,
                    eigenstate_count=max(3, min(int(args.ed_max_eigenstates), 8)),
                    check_ground_state_degeneracy=False,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    external_field_terms=active_field_terms,
                    show_progress=show_progress,
                    sparse_tol=float(args.ed_sparse_tol),
                    sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
                    target_tz2=int(
                        row.get(
                            "selected_target_tz2",
                            row_ed_plan.get("target_tz2", symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2))
                            if isinstance(row_ed_plan, dict)
                            else symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2),
                        )
                    ),
                    use_spin_pi_z=bool(
                        row.get("use_z2_conserved", False)
                        and str(row.get("z2_kind", row_ed_plan.get("z2_kind") if isinstance(row_ed_plan, dict) else "")) == "spin_pi_z"
                    ),
                    z2_target_parity=int(
                        row_ed_plan.get("z2_target_parity", args.z2_target_parity)
                        if isinstance(row_ed_plan, dict)
                        else args.z2_target_parity
                    ),
                    use_translation_x=bool(
                        row.get(
                            "use_translation_x_conserved",
                            row_ed_plan.get("use_translation_x_block", False) if isinstance(row_ed_plan, dict) else False,
                        )
                    ),
                    use_translation_y=bool(
                        row.get(
                            "use_translation_y_conserved",
                            row_ed_plan.get("use_translation_y_block", False) if isinstance(row_ed_plan, dict) else False,
                        )
                    ),
                    momentum_x=int(row_ed_plan.get("momentum_x_block", args.momentum_x_block)) if isinstance(row_ed_plan, dict) else int(args.momentum_x_block),
                    momentum_y=int(row_ed_plan.get("momentum_y_block", args.momentum_y_block)) if isinstance(row_ed_plan, dict) else int(args.momentum_y_block),
                    use_combined_c3=bool(
                        row.get(
                            "use_c3_conserved",
                            row_ed_plan.get("use_c3_block", False) if isinstance(row_ed_plan, dict) else False,
                        )
                    ),
                    c3_q_blocks=str(row_ed_plan.get("c3_q_blocks", args.ed_c3_q_blocks)) if isinstance(row_ed_plan, dict) else str(args.ed_c3_q_blocks),
                    strict_projector_memory=False,
                    allow_drop_c3_on_memory=True,
                )
                method_label = "standard_projector_representative_ed"
            else:
                spectrum, vectors, basis_list, basis_map = run_spin_orbital_u1_exact_spectrum(
                    geometry=geometry_obj,
                    model_spec=model_spec,
                    alpha=float(alpha),
                    beta=float(beta),
                    coupling_j=args.coupling_j,
                    eigenstate_count=max(3, min(int(args.ed_max_eigenstates), 8)),
                    check_ground_state_degeneracy=False,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    external_field_terms=active_field_terms,
                    show_progress=show_progress,
                    sparse_tol=float(args.ed_sparse_tol),
                    sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
                    use_sz_block=False,
                    target_sz2=int(symmetry_reduction_settings.get("target_sz2", args.u1_target_sz2)),
                    use_tau_z_block=True,
                    target_tz2=int(symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2)),
                )
                method_label = "standard_tz_representative_ed"
            correlations = collect_correlation_matrices_from_spin_orbital_u1_ed(
                geometry_obj,
                np.asarray(vectors[:, 0], dtype=np.complex128),
                basis_list,
                basis_map,
                show_progress=show_progress,
            )
            scalar_correlations = build_spin_orbital_u1_scalar_correlations(correlations)
            bond_rows = all_bond_energies_sz_conserved(
                geometry_obj,
                correlations,
                float(alpha),
                float(beta),
                args.coupling_j,
                show_progress=show_progress,
                progress_desc="Tz-ED bond energies",
            )
            energy = float(spectrum.get("ground_state_energy", np.nan))
        elif bool(row.get("use_sz_conserved", False)):
            spectrum, vectors, basis_list, basis_map = run_sz_conserved_exact_spectrum(
                geometry=geometry_obj,
                alpha=float(alpha),
                beta=float(beta),
                coupling_j=args.coupling_j,
                eigenstate_count=max(3, min(int(args.ed_max_eigenstates), 8)),
                check_ground_state_degeneracy=False,
                external_field_terms=active_field_terms,
                show_progress=show_progress,
                sparse_tol=float(args.ed_sparse_tol),
                sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
                target_sz2=int(symmetry_reduction_settings.get("target_sz2", args.u1_target_sz2)),
            )
            correlations = collect_correlation_matrices_from_sz_conserved_ed(
                geometry_obj,
                np.asarray(vectors[:, 0], dtype=np.complex128),
                basis_list,
                basis_map,
                show_progress=show_progress,
            )
            scalar_correlations = build_sz_conserved_scalar_correlations(correlations)
            bond_rows = all_bond_energies_sz_conserved(
                geometry_obj,
                correlations,
                float(alpha),
                float(beta),
                args.coupling_j,
                show_progress=show_progress,
            )
            energy = float(spectrum.get("ground_state_energy", np.nan))
            method_label = "standard_sz_representative_ed"
        else:
            spectrum, vectors = run_small_cluster_exact_spectrum(
                geometry=geometry_obj,
                model_spec=model_spec,
                alpha=float(alpha),
                beta=float(beta),
                coupling_j=args.coupling_j,
                eigenstate_count=max(1, int(args.ed_max_eigenstates)),
                check_ground_state_degeneracy=False,
                jx=args.jx,
                jy=args.jy,
                jz=args.jz,
                external_field_terms=active_field_terms,
                show_progress=show_progress,
                solver=args.ed_solver,
                sparse_tol=float(args.ed_sparse_tol),
                sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
            )
            correlations = collect_correlation_matrices_from_ed(
                geometry_obj,
                np.asarray(vectors[:, 0], dtype=np.complex128),
                model_spec=model_spec,
                show_progress=show_progress,
            )
            scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
            bond_rows = all_bond_energies(
                geometry_obj,
                correlations,
                model_spec,
                float(alpha),
                float(beta),
                args.coupling_j,
                jx=args.jx,
                jy=args.jy,
                jz=args.jz,
                show_progress=show_progress,
            )
            energy = float(spectrum.get("ground_state_energy", np.nan))
            method_label = "standard_full_representative_ed"
        return {
            "bond_energies": bond_rows,
            "real_space_patterns": build_reference_site_correlation_patterns(
                geometry_obj,
                scalar_correlations,
                reference_site_idx=args.reference_site_idx,
            ),
            "energy": energy,
            "recalculation_method": method_label,
        }

    def _tenpy_dmrg_representative_payload(
        geometry_obj: Any,
        alpha: float,
        beta: float,
        field_terms: List[Tuple[float, str]] | None = None,
    ) -> Dict[str, Any]:
        active_field_terms = list(hamiltonian_external_field_terms if field_terms is None else field_terms)
        yl_scan = _load_tenpy_backend_module()
        psi, _mpo, info = yl_scan.run_cylindrical_dmrg(
            geometry=geometry_obj,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=args.coupling_j,
            max_bond_dimension=args.max_bond_dimension,
            max_sweeps=args.max_sweeps,
            truncation_cutoff=args.truncation_cutoff,
            svd_min=args.dmrg_svd_min,
            initial_state=None,
            compute_phase_observables=False,
            external_field_terms=active_field_terms,
            symmetry_reductions=symmetry_reduction_settings,
            show_progress=show_progress,
        )
        correlations = yl_scan.collect_correlation_matrices_from_dmrg(psi, show_progress=show_progress)
        scalar_correlations = yl_scan.build_spin_orbital_scalar_correlations(correlations)
        return {
            "bond_energies": yl_scan.all_bond_energies(
                geometry_obj,
                correlations,
                float(alpha),
                float(beta),
                args.coupling_j,
                show_progress=show_progress,
            ),
            "real_space_patterns": build_reference_site_correlation_patterns(
                geometry_obj,
                scalar_correlations,
                reference_site_idx=args.reference_site_idx,
            ),
            "energy": float(info.get("E", np.nan)),
            "recalculation_method": "tenpy_finite_dmrg_representative_rerun",
        }

    def _tenpy_idmrg_representative_payload(
        geometry_obj: Any,
        alpha: float,
        beta: float,
        field_terms: List[Tuple[float, str]] | None = None,
    ) -> Dict[str, Any]:
        active_field_terms = list(hamiltonian_external_field_terms if field_terms is None else field_terms)
        yl_scan = _load_tenpy_backend_module()
        _rows, psi = yl_scan.run_alpha_scan_idmrg_with_adiabatic_state_passing(
            geometry=geometry_obj,
            alpha_values=[float(alpha)],
            beta=float(beta),
            coupling_j=args.coupling_j,
            max_bond_dimension=args.idmrg_max_bond_dimension,
            max_iterations=args.idmrg_max_iterations,
            truncation_cutoff=args.truncation_cutoff,
            svd_min=args.idmrg_svd_min,
            initial_state=None,
            classifier_thresholds=phase_classifier_thresholds_from_args(args),
            external_field_terms=active_field_terms,
            show_progress=show_progress,
            progress_bar=None,
        )
        if psi is None:
            raise RuntimeError("Representative iDMRG rerun did not return an optimized MPS.")
        correlations = yl_scan.collect_correlation_matrices_from_dmrg(psi, show_progress=show_progress)
        scalar_correlations = yl_scan.build_spin_orbital_scalar_correlations(correlations)
        return {
            "bond_energies": yl_scan.all_bond_energies(
                geometry_obj,
                correlations,
                float(alpha),
                float(beta),
                args.coupling_j,
                show_progress=show_progress,
            ),
            "real_space_patterns": build_reference_site_correlation_patterns(
                geometry_obj,
                scalar_correlations,
                reference_site_idx=args.reference_site_idx,
            ),
            "recalculation_method": "tenpy_idmrg_representative_rerun",
        }

    def _quimb_peps_representative_payload(
        geometry_obj: Any,
        alpha: float,
        beta: float,
        field_terms: List[Tuple[float, str]] | None = None,
    ) -> Dict[str, Any]:
        active_field_terms = list(hamiltonian_external_field_terms if field_terms is None else field_terms)
        import peps_backend as quimb_peps_backend

        result = quimb_peps_backend.run_quimb_peps_calculation(
            geometry=geometry_obj,
            model_spec=model_spec,
            lattice_name=lattice_name,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=args.coupling_j,
            jx=args.jx,
            jy=args.jy,
            jz=args.jz,
            external_field_terms=active_field_terms,
            max_sites=args.max_peps_sites,
            max_bond_dimension=args.peps_max_bond_dimension,
            max_sweeps=args.peps_max_sweeps,
            truncation_cutoff=args.truncation_cutoff,
            tau=args.peps_tau,
            random_seed=args.phase_scan_random_seed,
            initial_state_style=args.initial_state,
            ctm_chi=args.peps_ctm_chi,
            entanglement_max_dense_dim=args.peps_entanglement_max_dense_dim,
            classifier_thresholds=phase_classifier_thresholds_from_args(args),
            compute_correlations=True,
            compute_bond_energies=True,
            compute_structure_factors=False,
            compute_uniform_observables=False,
            compute_entanglement=False,
            show_progress=show_progress,
            args=args,
            symmetry_reductions=symmetry_reduction_settings,
            use_sz_conserved=bool(args.use_sz_conserved),
            symmetric=False,
            peps_symmetry_mode=args.peps_symmetry_mode,
            peps_strict_symmetry=bool(args.peps_strict_symmetry),
            peps_allow_dense_fallback=bool(args.peps_allow_dense_fallback),
        )
        info = result.get("info", {})
        scalar_correlations = result.get("scalar_correlations", {})
        return {
            "bond_energies": result.get("bond_rows", []),
            "real_space_patterns": build_reference_site_correlation_patterns(
                geometry_obj,
                scalar_correlations,
                reference_site_idx=args.reference_site_idx,
            ),
            "energy": info.get("ground_state_energy", info.get("E")),
            "recalculation_method": "quimb_peps_representative_rerun",
        }

    def _recalculate_phase_representative_payload(
        geometry_obj: Any,
        mode_key: str,
        representative: Dict[str, Any],
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        alpha = float(representative["alpha"])
        beta = float(representative.get("model_beta", representative["beta"]))
        field_terms = (
            [(float(coefficient), str(op_name)) for coefficient, op_name in row.get("external_field_terms", [])]
            if isinstance(row.get("external_field_terms"), list)
            else None
        )
        field_text = (
            f", B={float(representative['field_strength']):.6g}"
            if representative.get("field_strength") is not None
            else ""
        )
        print(
            "[phase] representative recalculation: "
            f"method={mode_key}, phase={representative['phase_label']}, alpha={alpha:.6g}, beta={beta:.6g}{field_text}"
        )
        if mode_key == "classical_product":
            return _classical_representative_payload(geometry_obj, row, alpha, beta)
        if mode_key == "quantum_ed":
            return _quantum_ed_representative_payload(geometry_obj, row, alpha, beta, field_terms=field_terms)
        if mode_key == "tenpy_dmrg":
            return _tenpy_dmrg_representative_payload(geometry_obj, alpha, beta, field_terms=field_terms)
        if mode_key == "quimb_peps":
            return _quimb_peps_representative_payload(geometry_obj, alpha, beta, field_terms=field_terms)
        if mode_key == "tenpy_idmrg":
            return _tenpy_idmrg_representative_payload(geometry_obj, alpha, beta, field_terms=field_terms)
        raise ValueError(f"Unsupported representative recalculation method '{mode_key}'.")

    def _save_phase_representative_outputs(
        summary_obj: Dict[str, Any],
        geometry_obj: Any,
        mode_key: str,
        representative: Dict[str, Any],
        row: Dict[str, Any],
    ) -> None:
        outputs = representative.setdefault("outputs", {})
        phase_label = str(representative["phase_label"])
        alpha = float(representative["alpha"])
        beta = float(representative["beta"])
        model_beta = float(representative.get("model_beta", beta))
        field_strength = representative.get("field_strength")
        row_field_vector = row.get("external_field_vector")
        representative_field_vector = (
            tuple(float(value) for value in row_field_vector)
            if isinstance(row_field_vector, (list, tuple)) and len(row_field_vector) == 3
            else resolved_field_vector
        )
        phase_token = _safe_filename_token(phase_label.lower())
        output_key_prefix = f"{mode_key}_{phase_token}_{int(representative['row_index'])}"
        row_bond_rows = row.get("bond_energies") if isinstance(row.get("bond_energies"), list) else None
        recalculation_mode_key = (
            mode_key[len("external_"):]
            if str(mode_key).startswith("external_")
            else mode_key
        )
        recalculation_supported = recalculation_mode_key in ("classical_product", "quantum_ed", "tenpy_dmrg", "quimb_peps", "tenpy_idmrg")
        needs_recalculation = bool(
            recalculation_supported
            and (args.plot_real_space_patterns or args.plot_bond_energies)
        )
        payload: Dict[str, Any] = {}
        if needs_recalculation:
            try:
                payload = _recalculate_phase_representative_payload(
                    geometry_obj,
                    recalculation_mode_key,
                    representative,
                    row,
                )
                representative["recalculation"] = {
                    "status": "completed",
                    "method": str(payload.get("recalculation_method", "representative_recalculation")),
                    "stored_payload": "filenames and compact diagnostics only; correlation arrays are not stored",
                }
                if "energy" in payload and payload["energy"] is not None:
                    representative["recalculated_energy"] = float(payload["energy"])
            except Exception as exc:
                representative["recalculation"] = {"status": "failed", "error": str(exc)}
                outputs["recalculation_error"] = str(exc)
                print(f"[phase] representative recalculation failed: {mode_key} {phase_label} :: {exc}")
                if not continue_on_plot_error:
                    raise

        bond_rows = payload.get("bond_energies")
        if not isinstance(bond_rows, list) or len(bond_rows) == 0:
            bond_rows = row_bond_rows if isinstance(row_bond_rows, list) else []
        pattern_payload = payload.get("real_space_patterns")
        correlations = (
            pattern_payload.get("correlations")
            if isinstance(pattern_payload, dict)
            else None
        )
        reference_site_idx = (
            pattern_payload.get("reference_site_idx")
            if isinstance(pattern_payload, dict)
            else None
        )
        if isinstance(pattern_payload, dict):
            representative["reference_site_idx"] = pattern_payload.get("reference_site_idx")
            if isinstance(pattern_payload.get("max_abs_correlation"), dict):
                representative["max_abs_correlation"] = pattern_payload["max_abs_correlation"]

        if (
            bool(args.plot_real_space_patterns or args.plot_bond_energies)
            and isinstance(correlations, dict)
            and "S" in correlations
            and reference_site_idx is not None
        ):
            filename = _phase_scan_representative_filename(
                mode_key,
                representative,
                "representative_pattern.png",
            )
            _save_plot_step(
                summary_obj,
                args.output_folder,
                f"{output_key_prefix}_representative_pattern_png",
                filename,
                lambda path, values=correlations["S"], ref_idx=int(reference_site_idx), rows_for_plot=bond_rows, title_phase=phase_label, title_alpha=alpha, title_beta=model_beta, title_field=field_strength, field_vector=representative_field_vector: save_phase_representative_pattern(
                    geometry_obj,
                    np.asarray(values, dtype=float),
                    ref_idx,
                    rows_for_plot,
                    path,
                    plot_title(
                        f"{mode_key} {title_phase} spin pattern + resolved bonds "
                        + (
                            f"(alpha={title_alpha:.6g}, B={float(title_field):.6g}, beta={title_beta:.6g})"
                            if title_field is not None
                            else f"(alpha={title_alpha:.6g}, beta={title_beta:.6g})"
                        )
                    ),
                    external_field_vector=field_vector,
                ),
                overwrite_existing,
                continue_on_plot_error,
            )
            outputs["representative_pattern_png"] = filename
            outputs["representative_pattern_note"] = (
                "Single combined plot: spin arrows show the relative sign pattern "
                "projected onto a chosen horizontal direction; bond overlays use "
                "resolved spin/orbital channel energies when available."
            )
        elif isinstance(correlations, dict):
            outputs["representative_pattern_note"] = "representative combined plotting is disabled"
        else:
            outputs["representative_pattern_note"] = (
                str(pattern_payload.get("warning"))
                if isinstance(pattern_payload, dict) and pattern_payload.get("warning")
                else "no spin reference-site pattern row available"
            )

    def save_phase_scan_outputs(
        summary_obj: Dict[str, Any],
        phase_scan_data: Dict[str, Any],
        geometry_obj: Any,
        mode_keys: set[str] | None = None,
    ) -> None:
        selected_mode_keys = set(mode_keys) if mode_keys is not None else None
        _record_phase_scan_plaquette_fluxes(summary_obj, phase_scan_data)
        phase_scan_filename = output_filename("phase_scan_summary.json")
        phase_scan_filepath = os.path.join(args.output_folder, phase_scan_filename)
        write_json(phase_scan_filepath, phase_scan_data)
        _record_output_status(summary_obj, "phase_scan_summary_json", phase_scan_filename, "saved")
        _save_summary_checkpoint(args.output_folder, summary_obj)
        print(f"[output] saved: {phase_scan_filepath}")

        if selected_mode_keys is None or "energy_b_scan" in selected_mode_keys:
            energy_b_data = phase_scan_data.get("energy_b_scan")
            if isinstance(energy_b_data, dict):
                base_name = "energy_b_scan.png"
                output_key = "energy_b_scan_png"
                if not bool(args.plot_phase_scan):
                    _skip_plot_step(
                        summary_obj,
                        args.output_folder,
                        output_key,
                        output_filename(base_name),
                        "plot_phase_scan is false",
                    )
                elif len(list(energy_b_data.get("rows", []))) == 0:
                    _record_output_status(
                        summary_obj,
                        output_key,
                        output_filename(base_name),
                        "skipped",
                        str(energy_b_data.get("reason", "No Energy-B scan rows available.")),
                    )
                    _save_summary_checkpoint(args.output_folder, summary_obj)
                else:
                    _save_plot_step(
                        summary_obj,
                        args.output_folder,
                        output_key,
                        output_filename(base_name),
                        lambda path, scan_payload=energy_b_data: save_energy_b_scan_plot(
                            scan_payload,
                            path,
                            title="DMRG Ground State and ED Bands vs External Field",
                            title_label=run_plot_title_label,
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )

        for mode_key, title in (
            ("classical_product", "Classical Product-State Phase Diagram"),
            ("quantum_ed", "Quantum ED Phase Diagram"),
            ("external_classical_product", "External-Field Classical Alpha-B Phase Diagram"),
            ("external_quantum_ed", "External-Field Quantum ED Alpha-B Phase Diagram"),
            ("tenax_dmrg", "Tenax Finite-DMRG Phase Diagram"),
            ("tenpy_dmrg", "TeNPy Finite-DMRG Phase Diagram"),
            ("tenax_idmrg", "Tenax iDMRG Phase Diagram"),
            ("tenpy_idmrg", "TeNPy iDMRG Phase Diagram"),
            ("quimb_peps", "quimb PEPS Phase Diagram"),
            ("quimb_ipeps", "quimb iPEPS Phase Diagram"),
        ):
            if selected_mode_keys is not None and mode_key not in selected_mode_keys:
                continue
            mode_data = phase_scan_data.get(mode_key)
            if not isinstance(mode_data, dict):
                continue
            scan_axes = mode_data.get("scan_axes") if isinstance(mode_data.get("scan_axes"), dict) else {}
            is_alpha_b_scan = str(scan_axes.get("y", "")).lower() in ("field_strength", "b")
            base_name = (
                (
                    "classical_alpha_b_phase_diagram.png"
                    if is_alpha_b_scan
                    else "classical_phase_diagram.png"
                )
                if mode_key == "classical_product"
                else (
                    ("quantum_alpha_b_phase_diagram.png" if is_alpha_b_scan else "quantum_phase_diagram.png")
                    if mode_key == "quantum_ed"
                    else f"{mode_key}_{'alpha_b_' if is_alpha_b_scan else ''}phase_diagram.png"
                )
            )
            output_key = f"{mode_key}_phase_diagram_png"
            rows = list(mode_data.get("rows", []))
            if len(rows) == 0:
                _record_output_status(
                    summary_obj,
                    output_key,
                    output_filename(base_name),
                    "skipped",
                    str(mode_data.get("reason", "No phase-scan rows available for this solver.")),
                )
                _save_summary_checkpoint(args.output_folder, summary_obj)
                continue
            completed_rows = [
                row for row in rows
                if str(row.get("status", "completed")) == "completed" and "phase_label" in row
            ]
            if len(completed_rows) == 0:
                _record_output_status(
                    summary_obj,
                    output_key,
                    output_filename(base_name),
                    "skipped",
                    "No completed phase-scan points available for this solver.",
                )
                _save_summary_checkpoint(args.output_folder, summary_obj)
                continue
            representative_pairs = _phase_scan_representative_rows(rows)
            representative_outputs_saved = bool(mode_data.get("representative_outputs_saved", False))
            if not representative_outputs_saved:
                mode_data["phase_representatives"] = [
                    representative for representative, _row in representative_pairs
                ]
                for representative, representative_row in representative_pairs:
                    _save_phase_representative_outputs(
                        summary_obj,
                        geometry_obj,
                        mode_key,
                        representative,
                        representative_row,
                    )
                mode_data["representative_outputs_saved"] = True
            if not bool(args.plot_phase_scan):
                _skip_plot_step(
                    summary_obj,
                    args.output_folder,
                    output_key,
                    output_filename(base_name),
                    "plot_phase_scan is false",
                )
                continue
            _save_plot_step(
                summary_obj,
                args.output_folder,
                output_key,
                output_filename(base_name),
                lambda path, scan_rows=completed_rows, scan_title=title, alpha_b=bool(is_alpha_b_scan): save_phase_diagram_plot(
                    scan_rows,
                    path,
                    ("Alpha-B " + scan_title) if alpha_b else scan_title,
                    title_label=run_plot_title_label,
                    y_label=(r"External field strength $B$" if alpha_b else r"$\beta$"),
                ),
                overwrite_existing,
                continue_on_plot_error,
            )

        for mode_key, title_prefix in (
            ("tenpy_dmrg", "TeNPy finite-DMRG"),
            ("tenpy_idmrg", "TeNPy iDMRG"),
            ("quimb_peps", "quimb PEPS"),
            ("quimb_ipeps", "quimb iPEPS"),
        ):
            if selected_mode_keys is not None and mode_key not in selected_mode_keys:
                continue
            mode_data = phase_scan_data.get(mode_key)
            if not isinstance(mode_data, dict):
                continue
            rows = list(mode_data.get("rows", []))
            observable_specs = [
                ("S_E", ("observables", "S_E"), "Center-bond entanglement entropy"),
                ("Sz_center", ("observables", "local_order_parameters", "Sz_center_mean"), "Center-site <Sz>"),
                ("tau_z_center", ("observables", "local_order_parameters", "tau_z_center_mean"), "Center-site <tau_z>"),
                ("W_p", ("observables", "W_p"), "Plaquette flux W_p"),
            ]
            if mode_key == "tenpy_idmrg":
                observable_specs.append(("xi", ("observables", "xi"), "Correlation length xi"))
            for observable_name, observable_path, colorbar_label in observable_specs:
                base_name = f"{mode_key}_{observable_name}_phase_observable.png"
                output_key = f"{mode_key}_{observable_name}_phase_observable_png"
                if len(rows) == 0:
                    _record_output_status(
                        summary_obj,
                        output_key,
                        output_filename(base_name),
                        "skipped",
                        str(mode_data.get("reason", "No phase-scan rows available for this solver.")),
                    )
                    _save_summary_checkpoint(args.output_folder, summary_obj)
                    continue
                if not bool(args.plot_phase_scan):
                    _skip_plot_step(
                        summary_obj,
                        args.output_folder,
                        output_key,
                        output_filename(base_name),
                        "plot_phase_scan is false",
                    )
                    continue
                _save_plot_step(
                    summary_obj,
                    args.output_folder,
                    output_key,
                    output_filename(base_name),
                    lambda path, scan_rows=rows, obs_path=observable_path, plot_title_text=f"{title_prefix} {colorbar_label}", cbar_label=colorbar_label: save_phase_observable_heatmap(
                        scan_rows,
                        path,
                        obs_path,
                        plot_title_text,
                        cbar_label,
                        title_label=run_plot_title_label,
                    ),
                    overwrite_existing,
                    continue_on_plot_error,
                )

        write_json(phase_scan_filepath, phase_scan_data)
        _record_output_status(summary_obj, "phase_scan_summary_json", phase_scan_filename, "saved")
        _save_summary_checkpoint(args.output_folder, summary_obj)

    def phase_scan_axis_values(axis_min: float, axis_max: float, points: int) -> List[float]:
        if int(points) <= 1:
            return [float(axis_min)]
        return [float(value) for value in np.linspace(float(axis_min), float(axis_max), int(points))]

    def run_requested_phase_scan_for_geometry(
        geometry_obj: Any,
        incremental_summary_obj: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        methods = list(args.phase_scan_methods)
        alpha_values = phase_scan_axis_values(
            args.phase_scan_alpha_min,
            args.phase_scan_alpha_max,
            args.phase_scan_alpha_points,
        )
        beta_values = phase_scan_axis_values(
            args.phase_scan_beta_min,
            args.phase_scan_beta_max,
            args.phase_scan_beta_points,
        )
        classifier_thresholds = phase_classifier_thresholds_from_args(args)
        scan_points_profiled = profile_scan_points_enabled()

        def _attach_scan_point_profile(
            row: Dict[str, Any],
            mode: str,
            alpha: float | None,
            beta: float | None,
            point_index: int,
            elapsed: float,
            extra: Dict[str, Any] | None = None,
        ) -> None:
            if not scan_points_profiled:
                return
            profile_extra: Dict[str, Any] = dict(extra or {})
            diagnostics = row.get("diagnostics") if isinstance(row, dict) else None
            flux_candidates: List[Any] = []
            if isinstance(row, dict):
                flux_candidates.extend([row.get("plaquette_flux"), row.get("observables")])
            if isinstance(diagnostics, dict):
                flux_candidates.extend([diagnostics.get("plaquette_flux"), diagnostics])
            for flux_candidate in flux_candidates:
                if not isinstance(flux_candidate, dict):
                    continue
                flux_value = flux_candidate.get("W_p", flux_candidate.get("value"))
                if flux_value is None:
                    continue
                try:
                    profile_extra.setdefault("W_p", float(flux_value))
                except Exception:
                    profile_extra.setdefault("W_p", flux_value)
                break
            profile_extra.setdefault("backend", row.get("backend") if isinstance(row, dict) else None)
            profile_entry = record_scan_point_timing(
                mode=mode,
                alpha=alpha,
                beta=beta,
                status=str(row.get("status", "unknown")),
                wall_time_seconds=float(elapsed),
                point_index=int(point_index),
                extra=profile_extra,
            )
            row_profile = row.setdefault("profiling", {})
            if isinstance(row_profile, dict):
                row_profile["wall_time_seconds"] = float(elapsed)
                row_profile["scan_point_timing"] = profile_entry

        def _tenax_phase_scan_row(
            alpha: float,
            beta: float,
            alpha_index: int,
            beta_index: int,
            point_index: int,
        ) -> Dict[str, Any]:
            point_seed = int(args.phase_scan_random_seed) + int(point_index)
            try:
                tenax_mps, _tenax_mpo, tenax_info = run_tenax_cylindrical_dmrg(
                    geometry=geometry_obj,
                    model_spec=model_spec,
                    alpha=float(alpha),
                    beta=float(beta),
                    coupling_j=args.coupling_j,
                    external_field_terms=hamiltonian_external_field_terms,
                    max_bond_dimension=args.max_bond_dimension,
                    max_sweeps=args.max_sweeps,
                    truncation_cutoff=args.truncation_cutoff,
                    svd_min=args.dmrg_svd_min,
                    random_seed=point_seed,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    symmetry_mode=effective_symmetry_mode,
                    u1_target_total_sz2=args.u1_target_sz2,
                    u1_target_total_tz2=args.u1_target_tz2,
                    z2_target_parity=args.z2_target_parity,
                    strict_symmetry_selection_rules=args.strict_symmetry_selection_rules,
                    allow_symmetry_fallback_to_dense=args.symmetry_allow_dense_fallback,
                    initial_state_style=args.initial_state,
                    show_progress=False,
                )
                correlations = collect_correlation_matrices_from_tenax(
                    tenax_mps,
                    geometry_obj,
                    model_spec=model_spec,
                    show_progress=False,
                )
                scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
                structure_rows = all_high_symmetry_structure_factors(
                    scalar_correlations,
                    geometry_obj,
                    lattice=lattice_name,
                    show_progress=False,
                    progress_desc="Tenax phase-scan structure factors",
                )
                bond_rows = all_bond_energies(
                    geometry_obj,
                    correlations,
                    model_spec,
                    float(alpha),
                    float(beta),
                    args.coupling_j,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    show_progress=False,
                    progress_desc="Tenax phase-scan bond energies",
                )
                diagnostics = _phase_observable_diagnostics(
                    structure_rows,
                    bond_rows,
                    geometry_obj.number_of_sites,
                    plaquette_flux=(tenax_info.get("phase_observables") or {}).get("plaquette_flux"),
                )
                phase_label = _classify_phase_from_diagnostics(
                    diagnostics,
                    float(alpha),
                    float(beta),
                    "tenax_dmrg",
                    classifier_thresholds,
                )
                energy = float(tenax_info.get("E", np.nan))
                return {
                    "status": "completed",
                    "backend": "tenax",
                    "alpha_index": int(alpha_index),
                    "beta_index": int(beta_index),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "energy": energy,
                    "energy_per_site": energy / float(max(1, geometry_obj.number_of_sites)),
                    "phase_label": phase_label,
                    "diagnostics": diagnostics,
                    "structure_factors": structure_rows,
                    "bond_energies": bond_rows,
                    "dmrg_info": tenax_info,
                    "symmetry_mode": str(tenax_info.get("symmetry_mode", effective_symmetry_mode)),
                    "requested_symmetry_reductions": list(args.symmetry_reductions),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "random_seed": int(point_seed),
                }
            except Exception as exc:
                return {
                    "status": "failed",
                    "backend": "tenax",
                    "alpha_index": int(alpha_index),
                    "beta_index": int(beta_index),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "error": str(exc),
                    "symmetry_mode": str(effective_symmetry_mode),
                    "requested_symmetry_reductions": list(args.symmetry_reductions),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "random_seed": int(point_seed),
                }

        def _run_tenax_dmrg_phase_scan() -> Dict[str, Any]:
            stage_start = _start_stage("Tenax finite-DMRG phase scan", show_progress)
            rows: List[Dict[str, Any]] = []
            progress_bar = _make_progress_bar(
                enabled=show_progress,
                total=len(alpha_values) * len(beta_values),
                desc="tenax dmrg scan",
                unit="point",
                leave=False,
            )
            point_index = 0
            for beta_index, beta in enumerate(beta_values):
                for alpha_index, alpha in enumerate(alpha_values):
                    point_start = time.perf_counter() if scan_points_profiled else None
                    row = _tenax_phase_scan_row(alpha, beta, alpha_index, beta_index, point_index)
                    if scan_points_profiled and point_start is not None:
                        point_elapsed = time.perf_counter() - point_start
                        _attach_scan_point_profile(
                            row,
                            "tenax_dmrg",
                            float(alpha),
                            float(beta),
                            point_index,
                            point_elapsed,
                            {
                                "alpha_index": int(alpha_index),
                                "beta_index": int(beta_index),
                                "symmetry_engine": "tenax",
                            },
                        )
                    rows.append(row)
                    point_index += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
            if progress_bar is not None:
                progress_bar.close()
            _end_stage("Tenax finite-DMRG phase scan", stage_start, show_progress)
            failed_count = int(sum(1 for row in rows if row.get("status") == "failed"))
            return {
                "status": "completed_with_warnings" if failed_count > 0 else "completed",
                "backend": "tenax",
                "scan_type": "finite_dmrg_observable_scan",
                "requested_backend": str(args.backend),
                "symmetry_mode": str(effective_symmetry_mode),
                "symmetry_reductions": symmetry_reduction_settings,
                "alpha_values": [float(value) for value in alpha_values],
                "beta_values": [float(value) for value in beta_values],
                "rows": rows,
                "completed_points": int(sum(1 for row in rows if row.get("status") == "completed")),
                "failed_points": failed_count,
                "skipped_points": 0,
            }

        def _run_selected_dmrg_phase_scan() -> Tuple[str, Dict[str, Any]]:
            backend_request = str(args.backend).strip().lower()
            tenpy_scan_issue = tenpy_backend_compatibility_issue()
            n_sites = int(geometry_obj.number_of_sites)
            if backend_request == "quimb":
                reason = (
                    "phase_scan_methods=dmrg now means finite DMRG only. "
                    "Use phase_scan_methods=peps for finite quimb PEPS."
                )
                if show_progress:
                    print(f"[phase-scan] skip finite DMRG with backend=quimb: {reason}")
                return "tenpy_dmrg", {
                    "status": "skipped",
                    "backend": "quimb",
                    "scan_type": "finite_dmrg_observable_scan",
                    "requested_backend": str(args.backend),
                    "selected_via": "phase_scan_method_dmrg",
                    "reason": reason,
                    "symmetry_mode": str(effective_symmetry_mode),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "alpha_values": [float(value) for value in alpha_values],
                    "beta_values": [float(value) for value in beta_values],
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(len(alpha_values) * len(beta_values)),
                }
            if n_sites > int(args.max_dmrg_sites):
                output_key = (
                    "tenpy_dmrg"
                    if backend_request == "tenpy" or (backend_request == "auto" and tenpy_scan_issue is None)
                    else "tenax_dmrg"
                )
                return output_key, {
                    "status": "skipped",
                    "backend": backend_request,
                    "scan_type": "finite_dmrg_observable_scan",
                    "requested_backend": str(args.backend),
                    "reason": (
                        f"Finite DMRG phase scan limited to {int(args.max_dmrg_sites)} sites, "
                        f"but geometry has N={n_sites}."
                    ),
                    "symmetry_mode": str(effective_symmetry_mode),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "alpha_values": [float(value) for value in alpha_values],
                    "beta_values": [float(value) for value in beta_values],
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(len(alpha_values) * len(beta_values)),
                }

            def _run_tenpy_dmrg_phase_scan() -> Tuple[str, Dict[str, Any]]:
                if tenpy_scan_issue is not None:
                    raise ValueError(f"TeNPy DMRG phase scan is not compatible: {tenpy_scan_issue}")
                yl_scan = _load_tenpy_backend_module()
                tenpy_data = yl_scan.run_alpha_beta_dmrg_observable_scan(
                    geometry=geometry_obj,
                    alpha_values=alpha_values,
                    beta_values=beta_values,
                    coupling_j=args.coupling_j,
                    max_bond_dimension=args.max_bond_dimension,
                    max_sweeps=args.max_sweeps,
                    truncation_cutoff=args.truncation_cutoff,
                    svd_min=args.dmrg_svd_min,
                    carry_state_between_betas=False,
                    classifier_thresholds=classifier_thresholds,
                    external_field_terms=hamiltonian_external_field_terms,
                    symmetry_reductions=symmetry_reduction_settings,
                    show_progress=show_progress,
                )
                tenpy_data["requested_backend"] = str(args.backend)
                tenpy_data["symmetry_reductions"] = symmetry_reduction_settings
                tenpy_data["symmetry_note"] = (
                    "TeNPy scan uses the real total-Tz U1 YaoLeeSite."
                    if bool(symmetry_reduction_settings.get("use_tau_z_block", False))
                    else "TeNPy scan uses dense/no-symmetry YaoLeeSite tensors."
                )
                return "tenpy_dmrg", tenpy_data

            if backend_request == "tenpy":
                return _run_tenpy_dmrg_phase_scan()
            if backend_request == "auto" and tenpy_scan_issue is None:
                try:
                    return _run_tenpy_dmrg_phase_scan()
                except Exception as tenpy_exc:
                    if show_progress:
                        print(f"[phase-scan] TeNPy DMRG scan failed; trying Tenax fallback. Reason: {tenpy_exc}")
                    tenax_data = _run_tenax_dmrg_phase_scan()
                    tenax_data["fallback_from"] = "tenpy"
                    tenax_data["fallback_reason"] = str(tenpy_exc)
                    return "tenax_dmrg", tenax_data
            if backend_request == "auto" and tenpy_scan_issue is not None and show_progress:
                print(f"[phase-scan] auto selects Tenax DMRG because TeNPy is not compatible: {tenpy_scan_issue}")
            if backend_request in ("tenax", "auto"):
                return "tenax_dmrg", _run_tenax_dmrg_phase_scan()
            raise ValueError(f"Unsupported DMRG phase-scan backend: {args.backend}")

        def _run_quimb_peps_phase_scan() -> Tuple[str, Dict[str, Any]]:
            try:
                import peps_backend as quimb_peps_backend

                peps_data = quimb_peps_backend.run_quimb_peps_scan(
                    geometry=geometry_obj,
                    alpha_values=alpha_values,
                    beta_values=beta_values,
                    coupling_j=args.coupling_j,
                    max_sites=args.max_peps_sites,
                    max_bond_dimension=args.peps_max_bond_dimension,
                    max_sweeps=args.peps_max_sweeps,
                    truncation_cutoff=args.truncation_cutoff,
                    tau=args.peps_tau,
                    carry_state_between_betas=False,
                    classifier_thresholds=classifier_thresholds,
                    external_field_terms=hamiltonian_external_field_terms,
                    show_progress=show_progress,
                    model_spec=model_spec,
                    lattice_name=lattice_name,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    random_seed=args.phase_scan_random_seed,
                    initial_state_style=args.initial_state,
                    ctm_chi=args.peps_ctm_chi,
                    entanglement_max_dense_dim=args.peps_entanglement_max_dense_dim,
                    symmetry_reductions=symmetry_reduction_settings,
                    args=args,
                    use_sz_conserved=bool(args.use_sz_conserved),
                    symmetric=False,
                    peps_symmetry_mode=args.peps_symmetry_mode,
                    peps_strict_symmetry=bool(args.peps_strict_symmetry),
                    peps_allow_dense_fallback=bool(args.peps_allow_dense_fallback),
                )
                peps_data["requested_backend"] = str(args.backend)
                peps_data["phase_scan_method"] = "peps"
                peps_data["selected_via"] = "phase_scan_method_peps"
                return "quimb_peps", peps_data
            except Exception as peps_exc:
                error_text = str(peps_exc) or peps_exc.__class__.__name__
                optional_dependency_missing = isinstance(peps_exc, (ImportError, ModuleNotFoundError))
                if show_progress:
                    detail = (
                        f"optional package unavailable :: {error_text}"
                        if optional_dependency_missing
                        else error_text
                    )
                    print(f"[phase-scan] skip quimb PEPS: {detail}")
                return "quimb_peps", {
                    "status": "skipped" if optional_dependency_missing else "failed",
                    "backend": "quimb_peps",
                    "requested_backend": str(args.backend),
                    "scan_type": "finite_peps_observable_scan",
                    "phase_scan_method": "peps",
                    "selected_via": "phase_scan_method_peps",
                    "reason": error_text,
                    "error": error_text,
                    "alpha_values": [float(value) for value in alpha_values],
                    "beta_values": [float(value) for value in beta_values],
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0 if optional_dependency_missing else int(len(alpha_values) * len(beta_values)),
                    "skipped_points": int(len(alpha_values) * len(beta_values)) if optional_dependency_missing else 0,
                    "energy_per_site": None,
                    "ground_state_energy_per_site": None,
                    "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": error_text},
                    "phase_label": "Weak/undetermined",
                    "all_plaquette_fluxes": {},
                    "symmetry_reductions": symmetry_reduction_settings,
                }

        def _run_quimb_ipeps_phase_scan() -> Tuple[str, Dict[str, Any]]:
            try:
                import peps_backend as quimb_ipeps_backend

                ipeps_data = quimb_ipeps_backend.run_quimb_ipeps_scan(
                    geometry=geometry_obj,
                    alpha_values=alpha_values,
                    beta_values=beta_values,
                    coupling_j=args.coupling_j,
                    max_unit_cell_sites=args.max_ipeps_unit_cell_sites,
                    max_bond_dimension=args.ipeps_max_bond_dimension,
                    max_iterations=args.ipeps_max_iterations,
                    truncation_cutoff=args.truncation_cutoff,
                    carry_state_between_betas=False,
                    classifier_thresholds=classifier_thresholds,
                    external_field_terms=hamiltonian_external_field_terms,
                    show_progress=show_progress,
                    model_spec=model_spec,
                    lattice_name=lattice_name,
                    random_seed=args.phase_scan_random_seed,
                    initial_state_style=args.initial_state,
                    tau=args.ipeps_tau,
                    ctm_chi=args.ipeps_ctm_chi,
                    symmetry_reductions=symmetry_reduction_settings,
                    args=args,
                    use_sz_conserved=bool(args.use_sz_conserved),
                    symmetric=False,
                    ipeps_symmetry_mode=args.ipeps_symmetry_mode,
                    ipeps_strict_symmetry=bool(args.ipeps_strict_symmetry),
                    ipeps_allow_dense_fallback=bool(args.ipeps_allow_dense_fallback),
                    unit_cell_kind=args.ipeps_unit_cell_kind,
                    use_translation_symmetry=bool(args.ipeps_use_translation_symmetry),
                    contraction_method=args.ipeps_contraction_method,
                )
                ipeps_data["requested_backend"] = str(args.backend)
                ipeps_data["phase_scan_method"] = "ipeps"
                ipeps_data["selected_via"] = "phase_scan_method_ipeps"
                return "quimb_ipeps", ipeps_data
            except Exception as ipeps_exc:
                error_text = str(ipeps_exc) or ipeps_exc.__class__.__name__
                optional_dependency_missing = isinstance(ipeps_exc, (ImportError, ModuleNotFoundError))
                if show_progress and optional_dependency_missing:
                    print(f"[phase-scan] skip quimb iPEPS: optional package unavailable :: {error_text}")
                return "quimb_ipeps", {
                    "status": "skipped" if optional_dependency_missing else "failed",
                    "backend": "quimb_ipeps",
                    "requested_backend": str(args.backend),
                    "scan_type": "ipeps_observable_scan",
                    "phase_scan_method": "ipeps",
                    "selected_via": "phase_scan_method_ipeps",
                    "alpha_values": [float(value) for value in alpha_values],
                    "beta_values": [float(value) for value in beta_values],
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0 if optional_dependency_missing else int(len(alpha_values) * len(beta_values)),
                    "skipped_points": int(len(alpha_values) * len(beta_values)) if optional_dependency_missing else 0,
                    "energy_per_site": None,
                    "ground_state_energy_per_site": None,
                    "plaquette_flux": {"available": False, "value": None, "W_p": None, "reason": error_text},
                    "phase_label": "Weak/undetermined",
                    "all_plaquette_fluxes": {},
                    "reason": error_text,
                    "error": error_text,
                    "symmetry_reductions": symmetry_reduction_settings,
                }

        def _external_scan_field_values() -> List[float]:
            return phase_scan_axis_values(
                float(args.external_scan_field_min),
                float(args.external_scan_field_max),
                int(args.external_scan_field_points),
            )

        def _external_scan_field_vector(field_strength: float) -> Tuple[float, float, float]:
            axis_mode = str(args.external_field_axis).strip().lower()
            if axis_mode in ("111", "001"):
                return external_field_vector(
                    axis=axis_mode,
                    strength=float(field_strength),
                    hx=0.0,
                    hy=0.0,
                    hz=0.0,
                )
            base = np.asarray(resolved_field_vector, dtype=float)
            norm = float(np.linalg.norm(base))
            if norm <= 1.0e-14:
                base = np.asarray([0.0, 0.0, 1.0], dtype=float)
                norm = 1.0
            scaled = base * (float(field_strength) / norm)
            return float(scaled[0]), float(scaled[1]), float(scaled[2])

        def _external_scan_field_terms(field_strength: float) -> List[Tuple[float, str]]:
            if str(args.external_field_treatment) != "hamiltonian":
                return []
            return list(
                external_field_terms_for_model(
                    _external_scan_field_vector(field_strength),
                    mu_b=args.mu_b,
                    field_sign=args.field_sign,
                    sigma_factor=args.field_sigma_factor,
                )
            )

        def _annotate_alpha_b_row(
            row: Dict[str, Any],
            *,
            model_beta: float,
            field_strength: float,
            field_terms: List[Tuple[float, str]],
            alpha_index: int,
            field_index: int,
            method_key: str,
        ) -> Dict[str, Any]:
            row["alpha_index"] = int(alpha_index)
            row["beta_index"] = int(field_index)
            row["field_index"] = int(field_index)
            row["field_strength"] = float(field_strength)
            row["B"] = float(field_strength)
            row["model_beta"] = float(model_beta)
            row["beta"] = float(field_strength)
            row["external_field_vector"] = [float(value) for value in _external_scan_field_vector(field_strength)]
            row["external_field_terms"] = [(float(coefficient), str(op_name)) for coefficient, op_name in field_terms]
            row["scan_axes"] = {"x": "alpha", "y": "field_strength", "model_beta": float(model_beta)}
            row["phase_scan_method"] = str(method_key)
            return row

        def _external_alpha_b_modes() -> List[str]:
            mode = str(args.external_scan_mode)
            if mode == "alpha_b_classical":
                return ["classical_product"]
            if mode == "alpha_b_quantum":
                return ["quantum_ed"]
            if mode in ("alpha_b_both", "alpha_b_all"):
                return ["quantum_ed", "classical_product"]
            return []

        def _run_external_alpha_b_phase_scan() -> Dict[str, Any]:
            field_values = _external_scan_field_values()
            selected_modes = _external_alpha_b_modes()
            model_beta = float(args.beta)
            output_alpha_b: Dict[str, Any] = {
                "status": "running",
                "mode": str(args.external_scan_mode),
                "external_scan_mode": str(args.external_scan_mode),
                "selected_outputs": list(selected_modes),
                "grid": {
                    "alpha_min": float(min(alpha_values)),
                    "alpha_max": float(max(alpha_values)),
                    "alpha_points": int(len(alpha_values)),
                    "alpha_values": [float(value) for value in alpha_values],
                    "field_min": float(min(field_values)),
                    "field_max": float(max(field_values)),
                    "field_points": int(len(field_values)),
                    "field_values": [float(value) for value in field_values],
                    "model_beta": float(model_beta),
                },
                "solver_controls": {
                    "external_scan_mode": str(args.external_scan_mode),
                    "external_field_treatment": str(args.external_field_treatment),
                    "external_field_axis": str(args.external_field_axis),
                    "external_scan_field_min": float(args.external_scan_field_min),
                    "external_scan_field_max": float(args.external_scan_field_max),
                    "external_scan_field_points": int(args.external_scan_field_points),
                    "model_beta": float(model_beta),
                    "quantum_ed_max_sites": int(args.phase_scan_ed_max_sites),
                    "quantum_ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
                    "classifier_thresholds": classifier_thresholds,
                },
            }
            progress_bar = _make_progress_bar(
                enabled=show_progress,
                total=len(selected_modes) * len(alpha_values) * len(field_values),
                desc="alpha-B scan",
                unit="point",
                leave=False,
            )
            point_index = 0
            for mode_key in selected_modes:
                rows: List[Dict[str, Any]] = []
                for field_index, field_strength in enumerate(field_values):
                    field_terms = _external_scan_field_terms(field_strength)
                    for alpha_index, alpha in enumerate(alpha_values):
                        point_start = time.perf_counter() if scan_points_profiled else None
                        try:
                            if mode_key == "quantum_ed":
                                row = _phase_scan_quantum_point(
                                    geometry_obj,
                                    model_spec,
                                    lattice_name,
                                    float(alpha),
                                    model_beta,
                                    args,
                                    field_terms,
                                    classifier_thresholds,
                                    show_progress=False,
                                )
                            else:
                                row = _phase_scan_classical_point(
                                    geometry_obj,
                                    model_spec,
                                    lattice_name,
                                    float(alpha),
                                    model_beta,
                                    args,
                                    point_index,
                                    field_terms,
                                    classifier_thresholds,
                                )
                            row = _annotate_alpha_b_row(
                                row,
                                model_beta=model_beta,
                                field_strength=float(field_strength),
                                field_terms=field_terms,
                                alpha_index=alpha_index,
                                field_index=field_index,
                                method_key=mode_key,
                            )
                        except Exception as exc:
                            row = _annotate_alpha_b_row(
                                {
                                    "status": "failed",
                                    "alpha": float(alpha),
                                    "error": str(exc),
                                },
                                model_beta=model_beta,
                                field_strength=float(field_strength),
                                field_terms=field_terms,
                                alpha_index=alpha_index,
                                field_index=field_index,
                                method_key=mode_key,
                            )
                        if scan_points_profiled and point_start is not None:
                            point_elapsed = time.perf_counter() - point_start
                            _attach_scan_point_profile(
                                row,
                                str(mode_key),
                                float(alpha),
                                float(model_beta),
                                point_index,
                                point_elapsed,
                                {
                                    "alpha_index": int(alpha_index),
                                    "field_index": int(field_index),
                                    "field_strength": float(field_strength),
                                    "B": float(field_strength),
                                },
                            )
                        rows.append(row)
                        point_index += 1
                        if progress_bar is not None:
                            progress_bar.update(1)
                failed_count = int(sum(1 for row in rows if row.get("status") == "failed"))
                output_alpha_b[mode_key] = {
                    "status": "completed_with_warnings" if failed_count > 0 else "completed",
                    "scan_type": "alpha_b_phase_scan",
                    "scan_axes": {"x": "alpha", "y": "field_strength", "model_beta": float(model_beta)},
                    "rows": rows,
                    "completed_points": int(sum(1 for row in rows if row.get("status") == "completed")),
                    "failed_points": failed_count,
                    "skipped_points": int(sum(1 for row in rows if row.get("status") == "skipped")),
                    "alpha_values": [float(value) for value in alpha_values],
                    "field_values": [float(value) for value in field_values],
                    "beta_values": [float(value) for value in field_values],
                    "model_beta": float(model_beta),
                }
            if progress_bar is not None:
                progress_bar.close()
            child_statuses = [
                item.get("status")
                for item in output_alpha_b.values()
                if isinstance(item, dict) and "status" in item
            ]
            output_alpha_b["status"] = (
                "completed_with_warnings"
                if any(status in ("failed", "completed_with_warnings") for status in child_statuses)
                else "completed"
            )
            return output_alpha_b

        def _run_energy_b_ed_point(field_strength: float, field_terms: List[Tuple[float, str]]) -> Dict[str, Any]:
            if int(geometry_obj.number_of_sites) > int(args.phase_scan_ed_max_sites):
                return {
                    "status": "skipped",
                    "reason": f"ED Energy-B scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
                    "energies": [],
                }
            ed_plan_for_point = (
                getattr(args, "ed_symmetry_plan", {})
                if isinstance(getattr(args, "ed_symmetry_plan", None), dict)
                else {}
            )
            if _ed_plan_requires_standard_projector(ed_plan_for_point, model_spec):
                parent_dim = _spin_orbital_symmetry_reduced_dimension(
                    int(geometry_obj.number_of_sites),
                    False,
                    int(ed_plan_for_point.get("target_sz2", args.u1_target_sz2)),
                    True,
                    int(ed_plan_for_point.get("target_tz2", args.u1_target_tz2)),
                )
                projector_factor = _ed_projector_reduction_factor_estimate(ed_plan_for_point, geometry_obj)
                projected_dim_estimate = int(max(1, int(parent_dim) // max(1, projector_factor)))
                if projected_dim_estimate > int(args.phase_scan_ed_max_hilbert_dim):
                    return {
                        "status": "skipped",
                        "reason": (
                            "ED Energy-B standard projector dimension estimate "
                            f"{projected_dim_estimate} exceeds {int(args.phase_scan_ed_max_hilbert_dim)}."
                        ),
                        "solver_mode": "spin_orbital_tz_projector",
                        "basis_type": "bitwise_spin_orbital_tz_projector_block",
                        "hilbert_dim": int(projected_dim_estimate),
                        "u1_parent_hilbert_dim": int(parent_dim),
                        "energies": [],
                    }
                spectrum, _vectors, _basis_list, _basis_map = run_spin_orbital_projected_exact_spectrum(
                    geometry=geometry_obj,
                    model_spec=model_spec,
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    coupling_j=args.coupling_j,
                    eigenstate_count=max(1, int(args.external_scan_ed_bands)),
                    check_ground_state_degeneracy=False,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    external_field_terms=field_terms,
                    show_progress=False,
                    ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                    ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                    sparse_tol=float(args.ed_sparse_tol),
                    sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
                    target_tz2=int(ed_plan_for_point.get("target_tz2", args.u1_target_tz2)),
                    use_spin_pi_z=bool(
                        ed_plan_for_point.get("use_z2_block", False)
                        and str(ed_plan_for_point.get("z2_kind")) == "spin_pi_z"
                    ),
                    z2_target_parity=int(ed_plan_for_point.get("z2_target_parity", args.z2_target_parity)),
                    use_translation_x=bool(ed_plan_for_point.get("use_translation_x_block", False)),
                    use_translation_y=bool(ed_plan_for_point.get("use_translation_y_block", False)),
                    momentum_x=int(ed_plan_for_point.get("momentum_x_block", args.momentum_x_block)),
                    momentum_y=int(ed_plan_for_point.get("momentum_y_block", args.momentum_y_block)),
                    use_combined_c3=bool(ed_plan_for_point.get("use_c3_block", False)),
                    c3_q_blocks=str(ed_plan_for_point.get("c3_q_blocks", args.ed_c3_q_blocks)),
                    strict_projector_memory=False,
                    allow_drop_c3_on_memory=True,
                )
                energies = [float(value) for value in list(spectrum.get("energies", []))]
                return {
                    "status": "completed",
                    "ed_backend": "standard",
                    "requested_ed_backend": str(args.ed_backend),
                    "backend_override_reason": (
                        "Energy-B ED used standard projector symmetries because the ED plan "
                        "contains fused translations/C3 or spin_pi_z."
                    ),
                    "solver_mode": spectrum.get("solver_mode"),
                    "basis_type": spectrum.get("basis_type", "bitwise_spin_orbital_tz_projector_block"),
                    "hilbert_dim": spectrum.get("hilbert_dim", projected_dim_estimate),
                    "u1_parent_hilbert_dim": int(parent_dim),
                    "projector_reduced_dimension": spectrum.get("projector_reduced_dimension"),
                    "projector_strategy": spectrum.get("projector_strategy"),
                    "memory_estimate_MB": spectrum.get("memory_estimate_MB"),
                    "dropped_symmetries": spectrum.get("dropped_symmetries", []),
                    "drop_reasons": spectrum.get("drop_reasons", {}),
                    "applied_reductions": [
                        name
                        for name, active in (
                            ("tz", spectrum.get("use_tau_z_block", True)),
                            ("z2", spectrum.get("use_z2_block", False)),
                            ("translation_x", spectrum.get("use_translation_x_block", False)),
                            ("translation_y", spectrum.get("use_translation_y_block", False)),
                            ("combined_c3", spectrum.get("use_c3_block", False)),
                        )
                        if bool(active)
                    ],
                    "commutator_norms": spectrum.get("commutator_norms", {}),
                    "selected_c3_q": spectrum.get("selected_c3_q"),
                    "c3_sector_energies": spectrum.get("c3_sector_energies"),
                    "energies": energies,
                    "energies_per_site": [
                        float(value) / float(max(1, int(geometry_obj.number_of_sites)))
                        for value in energies
                    ],
                }
            if str(ed_plan_for_point.get("effective_engine", "")).startswith("quspin"):
                try:
                    import quspin_backend as quspin_ed_backend
                except Exception as exc:
                    return {
                        "status": "skipped",
                        "reason": f"QuSpin-native Energy-B ED requested but quspin_backend is unavailable: {exc}",
                        "energies": [],
                    }
                use_quspin_z2 = bool(
                    ed_plan_for_point.get("use_z2_block", False)
                    and ed_plan_for_point.get("z2_kind") == "spin_flip"
                )
                spectrum, _vectors = quspin_ed_backend.run_small_cluster_exact_spectrum(
                    geometry=geometry_obj,
                    model_spec=model_spec,
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    coupling_j=args.coupling_j,
                    eigenstate_count=max(1, int(args.external_scan_ed_bands)),
                    check_ground_state_degeneracy=False,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    external_field_terms=field_terms,
                    show_progress=False,
                    solver=args.ed_solver,
                    sparse_tol=float(args.ed_sparse_tol),
                    sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
                    use_sz_block=False,
                    target_sz2=int(ed_plan_for_point.get("target_sz2", args.u1_target_sz2)),
                    use_tau_z_block=bool(ed_plan_for_point.get("use_tau_z_block", False)),
                    target_tz2=int(ed_plan_for_point.get("target_tz2", args.u1_target_tz2)),
                    use_z2_block=use_quspin_z2,
                    z2_generator=("spin_flip" if use_quspin_z2 else None),
                    z2_target_parity=int(ed_plan_for_point.get("z2_target_parity", args.z2_target_parity)),
                    use_translation_block=False,
                    use_translation_x_block=False,
                    use_translation_y_block=False,
                    momentum_block_1=0,
                    momentum_block_2=0,
                    momentum_x_block=0,
                    momentum_y_block=0,
                    use_reflection_block=False,
                    reflection_block=0,
                    check_symm=bool(args.quspin_check_symmetries),
                    check_herm=bool(args.quspin_check_hermiticity),
                    check_pcon=bool(args.quspin_check_particle_conservation),
                )
                energies = [float(value) for value in list(spectrum.get("energies", []))]
                return {
                    "status": "completed",
                    "ed_backend": "quspin",
                    "symmetry_engine": ed_plan_for_point.get("effective_engine", "quspin_native"),
                    "solver_mode": spectrum.get("solver_mode"),
                    "basis_type": spectrum.get("basis_type"),
                    "hilbert_dim": spectrum.get("hilbert_dim", spectrum.get("hilbert_dimension")),
                    "use_tau_z_block": bool(spectrum.get("use_tau_z_block", ed_plan_for_point.get("use_tau_z_block", False))),
                    "use_z2_block": bool(spectrum.get("use_z2_block", use_quspin_z2)),
                    "z2_kind": spectrum.get("z2_kind", "spin_flip" if use_quspin_z2 else None),
                    "metadata_note": (
                        "QuSpin-native Energy-B ED uses only the supported subset; "
                        "fused translations and combined C3 are not represented by this path."
                    ),
                    "energies": energies,
                    "energies_per_site": [
                        float(value) / float(max(1, int(geometry_obj.number_of_sites)))
                        for value in energies
                    ],
                }
            spectrum, _vectors = run_small_cluster_exact_spectrum(
                geometry=geometry_obj,
                model_spec=model_spec,
                alpha=float(args.alpha),
                beta=float(args.beta),
                coupling_j=args.coupling_j,
                eigenstate_count=max(1, int(args.external_scan_ed_bands)),
                check_ground_state_degeneracy=False,
                jx=args.jx,
                jy=args.jy,
                jz=args.jz,
                external_field_terms=field_terms,
                show_progress=False,
                solver=args.ed_solver,
                sparse_tol=float(args.ed_sparse_tol),
                sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
            )
            energies = [float(value) for value in list(spectrum.get("energies", []))]
            return {
                "status": "completed",
                "solver_mode": spectrum.get("solver_mode"),
                "hilbert_dim": spectrum.get("hilbert_dim"),
                "energies": energies,
                "energies_per_site": [
                    float(value) / float(max(1, int(geometry_obj.number_of_sites)))
                    for value in energies
                ],
            }

        def _run_energy_b_dmrg_point(field_strength: float, field_terms: List[Tuple[float, str]], point_index: int) -> Dict[str, Any]:
            if int(geometry_obj.number_of_sites) > int(args.max_dmrg_sites):
                return {
                    "status": "skipped",
                    "reason": f"DMRG Energy-B scan limited to {int(args.max_dmrg_sites)} sites.",
                }
            backend_request = str(args.backend).strip().lower()
            if backend_request == "quimb":
                return {"status": "skipped", "reason": "backend=quimb does not run finite DMRG; use backend=auto/tenpy/tenax."}
            tenpy_scan_issue = tenpy_backend_compatibility_issue()
            try:
                if backend_request == "tenpy" or (backend_request == "auto" and tenpy_scan_issue is None):
                    yl_scan = _load_tenpy_backend_module()
                    _psi, _mpo, info = yl_scan.run_cylindrical_dmrg(
                        geometry=geometry_obj,
                        alpha=float(args.alpha),
                        beta=float(args.beta),
                        coupling_j=args.coupling_j,
                        max_bond_dimension=args.max_bond_dimension,
                        max_sweeps=args.max_sweeps,
                        truncation_cutoff=args.truncation_cutoff,
                        svd_min=args.dmrg_svd_min,
                        random_seed=int(args.phase_scan_random_seed) + int(point_index),
                        product_state_style=args.initial_state,
                        compute_phase_observables=False,
                        external_field_terms=field_terms,
                        symmetry_reductions=symmetry_reduction_settings,
                        show_progress=False,
                    )
                    energy = float(info.get("E", info.get("energy", np.nan)))
                    return {
                        "status": "completed",
                        "backend": "tenpy",
                        "energy": energy,
                        "energy_per_site": energy / float(max(1, int(geometry_obj.number_of_sites))),
                    }
                mps, _mpo, info = run_tenax_cylindrical_dmrg(
                    geometry=geometry_obj,
                    model_spec=model_spec,
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    coupling_j=args.coupling_j,
                    external_field_terms=field_terms,
                    max_bond_dimension=args.max_bond_dimension,
                    max_sweeps=args.max_sweeps,
                    truncation_cutoff=args.truncation_cutoff,
                    svd_min=args.dmrg_svd_min,
                    random_seed=int(args.phase_scan_random_seed) + int(point_index),
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    symmetry_mode=effective_symmetry_mode,
                    u1_target_total_sz2=args.u1_target_sz2,
                    u1_target_total_tz2=args.u1_target_tz2,
                    z2_target_parity=args.z2_target_parity,
                    strict_symmetry_selection_rules=args.strict_symmetry_selection_rules,
                    allow_symmetry_fallback_to_dense=args.symmetry_allow_dense_fallback,
                    initial_state_style=args.initial_state,
                    show_progress=False,
                )
                del mps
                energy = float(info.get("E", info.get("energy", np.nan)))
                return {
                    "status": "completed",
                    "backend": "tenax",
                    "energy": energy,
                    "energy_per_site": energy / float(max(1, int(geometry_obj.number_of_sites))),
                }
            except Exception as exc:
                return {"status": "failed", "error": str(exc)}

        def _run_external_energy_b_scan() -> Dict[str, Any]:
            field_values = _external_scan_field_values()
            rows: List[Dict[str, Any]] = []
            progress_bar = _make_progress_bar(
                enabled=show_progress,
                total=len(field_values),
                desc="Energy-B scan",
                unit="point",
                leave=False,
            )
            for field_index, field_strength in enumerate(field_values):
                field_terms = _external_scan_field_terms(field_strength)
                point_start = time.perf_counter() if scan_points_profiled else None
                ed_result = _run_energy_b_ed_point(float(field_strength), field_terms)
                dmrg_result = _run_energy_b_dmrg_point(float(field_strength), field_terms, field_index)
                row = {
                    "status": (
                        "completed"
                        if ed_result.get("status") == "completed" or dmrg_result.get("status") == "completed"
                        else "skipped"
                    ),
                    "field_index": int(field_index),
                    "field_strength": float(field_strength),
                    "B": float(field_strength),
                    "alpha": float(args.alpha),
                    "beta": float(args.beta),
                    "model_beta": float(args.beta),
                    "external_field_vector": [float(value) for value in _external_scan_field_vector(field_strength)],
                    "external_field_terms": [(float(coefficient), str(op_name)) for coefficient, op_name in field_terms],
                    "ed_status": ed_result.get("status"),
                    "ed_energies": ed_result.get("energies", []),
                    "ed_energies_per_site": ed_result.get("energies_per_site", []),
                    "ed": ed_result,
                    "dmrg_status": dmrg_result.get("status"),
                    "dmrg_energy": dmrg_result.get("energy"),
                    "dmrg_energy_per_site": dmrg_result.get("energy_per_site"),
                    "dmrg": dmrg_result,
                }
                if row["status"] == "skipped":
                    row["reason"] = "; ".join(
                        str(item.get("reason", item.get("error", "")))
                        for item in (ed_result, dmrg_result)
                        if item.get("reason") or item.get("error")
                    )
                if scan_points_profiled and point_start is not None:
                    point_elapsed = time.perf_counter() - point_start
                    _attach_scan_point_profile(
                        row,
                        "energy_b",
                        float(args.alpha),
                        float(args.beta),
                        field_index,
                        point_elapsed,
                        {
                            "field_index": int(field_index),
                            "field_strength": float(field_strength),
                            "B": float(field_strength),
                            "ed_status": ed_result.get("status"),
                            "dmrg_status": dmrg_result.get("status"),
                        },
                    )
                rows.append(row)
                if progress_bar is not None:
                    progress_bar.update(1)
            if progress_bar is not None:
                progress_bar.close()
            completed_count = int(sum(1 for row in rows if row.get("status") == "completed"))
            return {
                "status": "completed" if completed_count > 0 else "skipped",
                "external_scan_mode": "e_b",
                "scan_type": "energy_b_scan",
                "rows": rows,
                "completed_points": completed_count,
                "failed_points": int(sum(1 for row in rows if row.get("status") == "failed")),
                "skipped_points": int(sum(1 for row in rows if row.get("status") == "skipped")),
                "field_values": [float(value) for value in field_values],
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "ed_bands": int(args.external_scan_ed_bands),
                "note": "DMRG ground-state energy and the lowest ED bands are plotted together versus external field strength B.",
            }

        def _external_field_phase_scan_is_available() -> bool:
            return bool(
                external_field_is_active(args.external_field_treatment, resolved_field_vector)
                and str(args.external_field_treatment) != "off"
                and str(args.external_scan_mode) != "none"
            )

        def _run_selected_external_phase_scan() -> Dict[str, Any]:
            if str(args.external_scan_mode) == "e_b":
                return {
                    "status": "completed",
                    "mode": "external_e_b",
                    "external_scan_mode": str(args.external_scan_mode),
                    "selected_outputs": ["energy_b_scan"],
                    "energy_b_scan": _run_external_energy_b_scan(),
                }
            return _run_external_alpha_b_phase_scan()

        def _copy_external_phase_scan_into_output(
            output_obj: Dict[str, Any],
            external_data: Dict[str, Any],
        ) -> List[str]:
            """Attach external scan payloads using non-conflicting output keys."""
            output_obj["external_scan"] = {
                "status": str(external_data.get("status", "unknown")),
                "mode": str(external_data.get("mode", "external")),
                "external_scan_mode": str(external_data.get("external_scan_mode", args.external_scan_mode)),
                "selected_outputs": list(external_data.get("selected_outputs", [])),
                "reason": external_data.get("reason"),
            }
            copied_keys: List[str] = ["external_scan"]
            if isinstance(external_data.get("energy_b_scan"), dict):
                output_obj["energy_b_scan"] = external_data["energy_b_scan"]
                copied_keys.append("energy_b_scan")
            for source_key, target_key in (
                ("quantum_ed", "external_quantum_ed"),
                ("classical_product", "external_classical_product"),
            ):
                if isinstance(external_data.get(source_key), dict):
                    mode_payload = external_data[source_key]
                    mode_payload["external_scan_mode"] = str(external_data.get("external_scan_mode", args.external_scan_mode))
                    mode_payload["phase_scan_channels"] = str(external_data.get("phase_scan_channels", "external"))
                    output_obj[target_key] = mode_payload
                    copied_keys.append(target_key)
            output_obj["selected_outputs"] = list(dict.fromkeys(
                list(output_obj.get("selected_outputs", []))
                + [
                    key
                    for key in ("energy_b_scan", "external_quantum_ed", "external_classical_product")
                    if isinstance(output_obj.get(key), dict)
                ]
            ))
            return copied_keys

        external_scan_available = _external_field_phase_scan_is_available()
        requested_phase_scan_channels = str(args.phase_scan_channels)
        if requested_phase_scan_channels == "auto":
            resolved_phase_scan_channels = "external" if external_scan_available else "normal"
        else:
            resolved_phase_scan_channels = requested_phase_scan_channels

        if resolved_phase_scan_channels == "none":
            return {
                "status": "skipped",
                "mode": str(args.phase_scan_mode),
                "phase_scan_channels": "none",
                "external_scan_mode": str(args.external_scan_mode),
                "selected_outputs": [],
                "reason": "phase_scan_channels=none disables both normal and external phase-scan calculations.",
            }

        external_phase_scan_data: Dict[str, Any] | None = None
        if resolved_phase_scan_channels in ("external", "both"):
            if external_scan_available:
                external_phase_scan_data = _run_selected_external_phase_scan()
            else:
                external_phase_scan_data = {
                    "status": "skipped",
                    "mode": "external",
                    "external_scan_mode": str(args.external_scan_mode),
                    "selected_outputs": [],
                    "reason": (
                        "External phase scan requested, but no active external scan is available "
                        "(requires nonzero field, treatment != off, and external_scan_mode != none)."
                    ),
                }
            external_phase_scan_data["phase_scan_channels"] = str(resolved_phase_scan_channels)
            external_phase_scan_data["requested_phase_scan_channels"] = str(args.phase_scan_channels)
            if resolved_phase_scan_channels == "external":
                return external_phase_scan_data
            if resolved_phase_scan_channels == "both" and incremental_summary_obj is not None:
                external_checkpoint_output: Dict[str, Any] = {
                    "status": str(external_phase_scan_data.get("status", "unknown")),
                    "mode": str(args.phase_scan_mode),
                    "phase_scan_channels": str(resolved_phase_scan_channels),
                    "requested_phase_scan_channels": str(args.phase_scan_channels),
                    "quantum_methods": list(args.phase_scan_quantum_methods),
                    "selected_outputs": [],
                    "grid": {
                        "alpha_min": float(min(alpha_values)),
                        "alpha_max": float(max(alpha_values)),
                        "alpha_points": int(len(alpha_values)),
                        "alpha_values": [float(value) for value in alpha_values],
                    },
                    "solver_controls": {
                        "phase_scan_channels": str(args.phase_scan_channels),
                        "external_scan_mode": str(args.external_scan_mode),
                        "external_scan_field_min": float(args.external_scan_field_min),
                        "external_scan_field_max": float(args.external_scan_field_max),
                        "external_scan_field_points": int(args.external_scan_field_points),
                        "external_scan_ed_bands": int(args.external_scan_ed_bands),
                        "symmetry_reductions": symmetry_reduction_settings,
                    },
                    "note": (
                        "External phase-scan checkpoint saved immediately after the alpha-B/Energy-B "
                        "calculation finished; normal alpha-beta scans may continue afterward."
                    ),
                }
                copied_external_keys = _copy_external_phase_scan_into_output(
                    external_checkpoint_output,
                    external_phase_scan_data,
                )
                if show_progress:
                    print(
                        "[phase-scan] checkpoint: external scan finished; saving plots and "
                        "representative outputs before starting normal phase scan."
                    )
                incremental_summary_obj["phase_scan"] = external_checkpoint_output
                incremental_summary_obj.setdefault("stages", {})["phase_scan"] = "running"
                save_phase_scan_outputs(
                    incremental_summary_obj,
                    external_checkpoint_output,
                    geometry_obj,
                    mode_keys={key for key in copied_external_keys if key != "external_scan"},
                )
                _save_summary_checkpoint(args.output_folder, incremental_summary_obj)

        output: Dict[str, Any] = {
            "status": "running",
            "mode": str(args.phase_scan_mode),
            "phase_scan_channels": str(resolved_phase_scan_channels),
            "requested_phase_scan_channels": str(args.phase_scan_channels),
            "quantum_methods": list(args.phase_scan_quantum_methods),
            "selected_outputs": methods,
            "grid": {
                "alpha_min": float(min(alpha_values)),
                "alpha_max": float(max(alpha_values)),
                "alpha_points": int(len(alpha_values)),
                "alpha_values": alpha_values,
                "beta_min": float(min(beta_values)),
                "beta_max": float(max(beta_values)),
                "beta_points": int(len(beta_values)),
                "beta_values": beta_values,
            },
            "solver_controls": {
                "phase_scan_channels": str(args.phase_scan_channels),
                "mode": str(args.phase_scan_mode),
                "quantum_methods": list(args.phase_scan_quantum_methods),
                "selected_outputs": methods,
                "ed_max_sites": int(args.phase_scan_ed_max_sites),
                "ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
                "ed_backend": str(args.ed_backend),
                "ed_symmetry_plan": getattr(args, "ed_symmetry_plan", {"status": "pending"}),
                "dmrg_backend": str(args.backend),
                "dmrg_effective_symmetry_mode": str(effective_symmetry_mode),
                "symmetry_reductions": symmetry_reduction_settings,
                "translation_reductions": {
                    "use_x": bool(args.use_translation_x_block),
                    "use_y": bool(args.use_translation_y_block),
                    "momentum_x": int(args.momentum_x_block),
                    "momentum_y": int(args.momentum_y_block),
                },
                "dmrg_max_bond_dimension": int(args.max_bond_dimension),
                "dmrg_max_sweeps": int(args.max_sweeps),
                "peps_max_sites": int(args.max_peps_sites),
                "peps_max_bond_dimension": int(args.peps_max_bond_dimension),
                "peps_bond_dimension_cap": int(args.peps_bond_dimension_cap),
                "peps_max_sweeps": int(args.peps_max_sweeps),
                "peps_sweep_cap": int(args.peps_sweep_cap),
                "peps_ctm_chi": int(args.peps_ctm_chi),
                "peps_ctm_chi_cap": int(args.peps_ctm_chi_cap),
                "peps_tau": float(args.peps_tau),
                "peps_entanglement_max_dense_dim": int(args.peps_entanglement_max_dense_dim),
                "peps_symmetry_mode": str(args.peps_symmetry_mode),
                "peps_strict_symmetry": bool(args.peps_strict_symmetry),
                "peps_allow_dense_fallback": bool(args.peps_allow_dense_fallback),
                "idmrg_max_bond_dimension": int(args.idmrg_max_bond_dimension),
                "idmrg_max_iterations": int(args.idmrg_max_iterations),
                "idmrg_use_translation_symmetry": bool(args.idmrg_use_translation_symmetry),
                "ipeps_max_unit_cell_sites": int(args.max_ipeps_unit_cell_sites),
                "ipeps_max_bond_dimension": int(args.ipeps_max_bond_dimension),
                "ipeps_bond_dimension_cap": int(args.ipeps_bond_dimension_cap),
                "ipeps_max_iterations": int(args.ipeps_max_iterations),
                "ipeps_iteration_cap": int(args.ipeps_iteration_cap),
                "ipeps_ctm_chi": int(args.ipeps_ctm_chi),
                "ipeps_ctm_chi_cap": int(args.ipeps_ctm_chi_cap),
                "ipeps_tau": float(args.ipeps_tau),
                "ipeps_symmetry_mode": str(args.ipeps_symmetry_mode),
                "ipeps_strict_symmetry": bool(args.ipeps_strict_symmetry),
                "ipeps_allow_dense_fallback": bool(args.ipeps_allow_dense_fallback),
                "ipeps_unit_cell_kind": str(args.ipeps_unit_cell_kind),
                "ipeps_use_translation_symmetry": bool(args.ipeps_use_translation_symmetry),
                "ipeps_contraction_method": str(args.ipeps_contraction_method),
                "ipeps_ctmrg_enabled": str(args.ipeps_contraction_method) == "ctmrg",
                "external_scan_mode": str(args.external_scan_mode),
                "external_scan_field_min": float(args.external_scan_field_min),
                "external_scan_field_max": float(args.external_scan_field_max),
                "external_scan_field_points": int(args.external_scan_field_points),
                "external_scan_ed_bands": int(args.external_scan_ed_bands),
                "truncation_cutoff": float(args.truncation_cutoff),
                "tenpy_symmetry_mode": str(effective_symmetry_mode),
                "classifier_thresholds": classifier_thresholds,
            },
        }

        def _save_incremental_completed_modes(completed_mode_keys: List[str], note: str) -> None:
            if incremental_summary_obj is None:
                return
            mode_key_set = {
                str(key)
                for key in completed_mode_keys
                if isinstance(output.get(str(key)), dict)
            }
            if len(mode_key_set) == 0:
                return
            if show_progress:
                print(
                    "[phase-scan] checkpoint: saving completed "
                    f"{', '.join(sorted(mode_key_set))} outputs before continuing to {note}."
                )
            incremental_summary_obj["phase_scan"] = output
            incremental_summary_obj.setdefault("stages", {})["phase_scan"] = "running"
            save_phase_scan_outputs(
                incremental_summary_obj,
                output,
                geometry_obj,
                mode_keys=mode_key_set,
            )

        legacy_mode = _phase_scan_legacy_mode_from_methods(methods)
        if legacy_mode is not None:
            args_for_legacy = argparse.Namespace(**vars(args))
            args_for_legacy.phase_scan_mode = legacy_mode
            legacy_data = run_alpha_beta_phase_scan(
                geometry=geometry_obj,
                model_spec=model_spec,
                lattice_name=lattice_name,
                args=args_for_legacy,
                hamiltonian_external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
            )
            completed_legacy_keys: List[str] = []
            for key in ("quantum_ed", "classical_product"):
                if key in legacy_data:
                    output[key] = legacy_data[key]
                    completed_legacy_keys.append(key)
            remaining_after_legacy = [method for method in methods if method not in ("ed", "classical")]
            if remaining_after_legacy:
                _save_incremental_completed_modes(completed_legacy_keys, "quantum tensor-network scans")
        elif "ed" in methods or "classical" in methods:
            ed_classical_methods = [method for method in methods if method in ("ed", "classical")]
            args_for_legacy = argparse.Namespace(**vars(args))
            args_for_legacy.phase_scan_mode = _phase_scan_legacy_mode_from_methods(ed_classical_methods)
            legacy_data = run_alpha_beta_phase_scan(
                geometry=geometry_obj,
                model_spec=model_spec,
                lattice_name=lattice_name,
                args=args_for_legacy,
                hamiltonian_external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
            )
            completed_legacy_keys = []
            for key in ("quantum_ed", "classical_product"):
                if key in legacy_data:
                    output[key] = legacy_data[key]
                    completed_legacy_keys.append(key)
            remaining_after_legacy = [method for method in methods if method not in ("ed", "classical")]
            if remaining_after_legacy:
                _save_incremental_completed_modes(completed_legacy_keys, "quantum tensor-network scans")

        if "dmrg" in methods:
            dmrg_key, dmrg_data = _run_selected_dmrg_phase_scan()
            output[dmrg_key] = dmrg_data
            if any(method in methods for method in ("peps", "idmrg", "ipeps")):
                _save_incremental_completed_modes([dmrg_key], "remaining phase scans")

        if "peps" in methods:
            peps_key, peps_data = _run_quimb_peps_phase_scan()
            output[peps_key] = peps_data
            if any(method in methods for method in ("idmrg", "ipeps")):
                _save_incremental_completed_modes([peps_key], "remaining phase scans")

        if "idmrg" in methods:
            backend_request = str(args.backend).strip().lower()
            tenpy_scan_issue = tenpy_backend_compatibility_issue()
            if not bool(args.idmrg_use_translation_symmetry):
                reason = "iDMRG translation symmetry was disabled by --no-idmrg-use-translation-symmetry."
                output["tenpy_idmrg"] = {
                    "status": "skipped",
                    "backend": backend_request,
                    "requested_backend": str(args.backend),
                    "scan_type": "idmrg_observable_scan",
                    "selected_via": "phase_scan_method_idmrg",
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(len(alpha_values) * len(beta_values)),
                    "reason": reason,
                    "translation_symmetry": {"enabled": False},
                    "symmetry_reductions": symmetry_reduction_settings,
                    "symmetry_mode": str(effective_symmetry_mode),
                }
                _save_incremental_completed_modes(["tenpy_idmrg"], "final output collation")
            elif backend_request == "quimb":
                reason = (
                    "phase_scan_methods=idmrg now means tensor-network iDMRG only. "
                    "Use phase_scan_methods=ipeps for quimb iPEPS."
                )
                if show_progress:
                    print(f"[phase-scan] skip iDMRG with backend=quimb: {reason}")
                output["tenpy_idmrg"] = {
                    "status": "skipped",
                    "backend": "quimb",
                    "requested_backend": str(args.backend),
                    "scan_type": "idmrg_observable_scan",
                    "selected_via": "phase_scan_method_idmrg",
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(len(alpha_values) * len(beta_values)),
                    "reason": reason,
                    "symmetry_reductions": symmetry_reduction_settings,
                    "symmetry_mode": str(effective_symmetry_mode),
                }
                _save_incremental_completed_modes(["tenpy_idmrg"], "final output collation")
            elif backend_request == "tenax" or (backend_request == "auto" and tenpy_scan_issue is not None):
                output["tenax_idmrg"] = {
                    "status": "skipped",
                    "backend": "tenax",
                    "requested_backend": str(args.backend),
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(len(alpha_values) * len(beta_values)),
                    "reason": (
                        "Tenax iDMRG phase-scan classification is not implemented yet. "
                        + (
                            f"Auto selected this skip because TeNPy is not compatible: {tenpy_scan_issue}"
                            if backend_request == "auto" and tenpy_scan_issue is not None
                            else "The single-point iDMRG workflow still uses Tenax when the finite DMRG backend is Tenax."
                        )
                    ),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "symmetry_mode": str(effective_symmetry_mode),
                }
                _save_incremental_completed_modes(["tenax_idmrg"], "final output collation")
            else:
                if tenpy_scan_issue is not None:
                    raise ValueError(f"TeNPy iDMRG phase scan is not compatible: {tenpy_scan_issue}")
                yl_scan = _load_tenpy_backend_module()
                output["tenpy_idmrg"] = yl_scan.run_alpha_beta_idmrg_observable_scan(
                    geometry=geometry_obj,
                    alpha_values=alpha_values,
                    beta_values=beta_values,
                    coupling_j=args.coupling_j,
                    max_unit_cell_sites=args.max_ipeps_unit_cell_sites,
                    max_bond_dimension=args.idmrg_max_bond_dimension,
                    max_iterations=args.idmrg_max_iterations,
                    truncation_cutoff=args.truncation_cutoff,
                    svd_min=args.idmrg_svd_min,
                    carry_state_between_betas=False,
                    classifier_thresholds=classifier_thresholds,
                    external_field_terms=hamiltonian_external_field_terms,
                    symmetry_reductions=symmetry_reduction_settings,
                    show_progress=show_progress,
                )
                output["tenpy_idmrg"]["requested_backend"] = str(args.backend)
                output["tenpy_idmrg"]["symmetry_reductions"] = symmetry_reduction_settings
                output["tenpy_idmrg"]["translation_symmetry"] = {
                    "enabled": bool(args.idmrg_use_translation_symmetry),
                    "implemented_as": "infinite repeated MPS unit cell along x",
                }
                output["tenpy_idmrg"]["symmetry_note"] = (
                    "TeNPy iDMRG scan uses the real total-Tz U1 YaoLeeSite."
                    if bool(symmetry_reduction_settings.get("use_tau_z_block", False))
                    else "TeNPy iDMRG scan uses dense/no-symmetry YaoLeeSite tensors."
                )
                _save_incremental_completed_modes(["tenpy_idmrg"], "final output collation")

        if "ipeps" in methods:
            ipeps_key, ipeps_data = _run_quimb_ipeps_phase_scan()
            output[ipeps_key] = ipeps_data
            _save_incremental_completed_modes([ipeps_key], "final output collation")

        if external_phase_scan_data is not None:
            _copy_external_phase_scan_into_output(output, external_phase_scan_data)
        child_statuses = [
            value.get("status")
            for value in output.values()
            if isinstance(value, dict) and "status" in value
        ]
        if any(status in ("failed", "completed_with_warnings") for status in child_statuses):
            output["status"] = "completed_with_warnings"
        else:
            output["status"] = "completed"
        return output

    def finalize_summary_with_profiling(summary_obj: Dict[str, Any]) -> None:
        if not profiling_enabled():
            return
        update_profile_metadata(
            actual_backend=backend_used,
            actual_method=args.method,
            actual_ed_backend=args.ed_backend,
            symmetry_engine=getattr(args, "ed_symmetry_engine", None),
        )
        profiling_payload = finalize_profiling(
            summary_obj,
            output_folder=args.profile_output_folder,
        )
        if profiling_payload:
            summary_obj["profiling"] = profiling_payload

    if args.phase_scan_only:
        with profile_stage("geometry construction"):
            geometry = build_lattice_geometry(
                lattice=lattice_name,
                length_x=args.length_x,
                length_y=args.length_y,
                circumference_x=circumference_x,
                circumference_y=circumference_y,
            )
        symmetry_preflight_report = run_symmetry_preflight_for_geometry(geometry)
        effective_symmetry_mode = str(
            symmetry_preflight_report.get("effective_mode_for_tenax", effective_symmetry_mode)
        )
        configure_run_output_names(geometry)
        if geometry_plot_status == "not_attempted":
            save_geometry_before_sweep(geometry)
        scan_external_field_summary = external_field_construction_summary(
            treatment=args.external_field_treatment,
            axis=args.external_field_axis,
            field_vector=resolved_field_vector,
            mu_b=args.mu_b,
            field_sign=args.field_sign,
            sigma_factor=args.field_sigma_factor,
            field_terms=hamiltonian_external_field_terms,
        )
        scan_summary: Dict[str, Any] = {
            "model_name": f"{lattice_display_name(lattice_name)} phase scan",
            "run_output_prefix": run_file_prefix,
            "monitor_data_name": run_summary_filename,
            "plot_title_label": run_plot_title_label,
            "resource_profile": ACTIVE_RESOURCE_PROFILE,
            "active_resource_settings": ACTIVE_RESOURCE_SETTINGS,
            "external_field": scan_external_field_summary,
            "run_status": "running",
            "parameters": vars(args),
            "observable_controls": observable_controls,
            "model_construction_annotations": {
                "phase_scan": {
                    "phase_diagram_enabled": bool(args.phase_diagram),
                    "channels": str(args.phase_scan_channels),
                    "mode": str(args.phase_scan_mode),
                    "quantum_methods": list(args.phase_scan_quantum_methods),
                    "selected_outputs": list(args.phase_scan_methods),
                    "alpha_points": int(args.phase_scan_alpha_points),
                    "beta_points": int(args.phase_scan_beta_points),
                    "external_scan_mode": str(args.external_scan_mode),
                    "external_scan_field_min": float(args.external_scan_field_min),
                    "external_scan_field_max": float(args.external_scan_field_max),
                    "external_scan_field_points": int(args.external_scan_field_points),
                    "external_scan_ed_bands": int(args.external_scan_ed_bands),
                    "quantum_ed_max_sites": int(args.phase_scan_ed_max_sites),
                    "quantum_ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
                    "quantum_ed_solver": str(args.ed_solver),
                    "quantum_ed_use_sz_block": bool(args.use_sz_block),
                    "dmrg_backend": str(args.backend),
                    "dmrg_svd_min": float(args.dmrg_svd_min),
                    "idmrg_svd_min": float(args.idmrg_svd_min),
                    "truncation_cutoff": float(args.truncation_cutoff),
                    "dmrg_effective_symmetry_mode": str(effective_symmetry_mode),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "quantum_ed_sparse_tol": float(args.ed_sparse_tol),
                    "quantum_ed_sparse_maxiter": (
                        int(args.ed_sparse_maxiter)
                        if int(args.ed_sparse_maxiter) > 0
                        else None
                    ),
                    "classical_restarts": int(args.phase_scan_classical_restarts),
                    "classical_sweeps": int(args.phase_scan_classical_sweeps),
                    "classifier_thresholds": {
                        "quantum_weak_order": float(args.phase_scan_quantum_weak_order_threshold),
                        "classical_weak_order": float(args.phase_scan_classical_weak_order_threshold),
                        "quantum_bond_nematicity": float(args.phase_scan_quantum_nematicity_threshold),
                        "classical_bond_nematicity": float(args.phase_scan_classical_nematicity_threshold),
                        "plaquette_flux_target": float(args.phase_scan_plaquette_flux_target),
                        "plaquette_flux_tolerance": float(args.phase_scan_plaquette_flux_tolerance),
                    },
                    "note": (
                        "Phase-scan mode chooses quantum, classical, or both. Quantum methods are explicit: "
                        "ed, dmrg, idmrg, peps, ipeps, or all. PEPS/iPEPS are requested directly."
                    ),
                },
                "symmetry": {
                    "requested_reductions": list(args.symmetry_reductions),
                    "applied_reductions": symmetry_reduction_settings,
                    "requested_mode": str(args.symmetry_mode),
                    "effective_mode_for_tenax": str(effective_symmetry_mode),
                    "precheck": symmetry_preflight_report,
                    "allow_dense_fallback": bool(args.symmetry_allow_dense_fallback),
                    "strict_precheck": bool(args.strict_symmetry_precheck),
                },
                "quspin_ed": {
                    "settings": quspin_ed_settings,
                    "note": "QuSpin consumes the same shared symmetry reductions as standard ED and phase scans when implemented.",
                },
                "external_field": scan_external_field_summary,
            },
            "model_spec": {
                "spin_rep": model_spec.spin_rep,
                "orbital_rep": model_spec.orbital_rep,
                "model_family": model_spec.model_family,
                "ising_axis": model_spec.ising_axis,
                "physical_dim": model_spec.physical_dim,
            },
            "geometry": {
                "lattice": lattice_name,
                "length_x": int(args.length_x),
                "length_y": int(args.length_y),
                "circumference_x": bool(circumference_x),
                "circumference_y": bool(circumference_y),
                "number_of_sites": geometry.number_of_sites,
                "number_of_bonds": len(geometry.bond_list),
                "mps_path": mps_path_quality(geometry),
            },
            "stages": {"phase_scan": "running"},
            "outputs": {
                "run_summary_json": run_summary_filename,
                "monitor_data_json": run_summary_filename,
            },
        }
        _record_output_status(
            scan_summary,
            "geometry_diagram_png",
            output_filename("geometry_diagram.png"),
            geometry_plot_status,
            geometry_plot_error,
        )
        _save_summary_checkpoint(args.output_folder, scan_summary)
        try:
            phase_scan_data = run_requested_phase_scan_for_geometry(
                geometry,
                incremental_summary_obj=scan_summary,
            )
            scan_summary["phase_scan"] = phase_scan_data
            scan_summary["stages"]["phase_scan"] = "completed"
            save_phase_scan_outputs(scan_summary, phase_scan_data, geometry)
            if _attach_plot_output_warnings(scan_summary, "phase_scan"):
                scan_summary["stages"]["phase_scan"] = "completed_with_warnings"
        except Exception as exc:
            scan_summary["phase_scan"] = {"status": "failed", "error": str(exc)}
            scan_summary["stages"]["phase_scan"] = "failed"
            _save_summary_checkpoint(args.output_folder, scan_summary)
            if not continue_on_plot_error:
                raise
        output_warning_keys = _attach_plot_output_warnings(scan_summary, "phase_scan")
        scan_summary["run_status"] = (
            "completed_with_warnings"
            if output_warning_keys
            or (
                isinstance(scan_summary.get("phase_scan"), dict)
                and scan_summary["phase_scan"].get("status") in ("failed", "completed_with_warnings")
            )
            else "completed"
        )
        finalize_summary_with_profiling(scan_summary)
        _save_summary_checkpoint(args.output_folder, scan_summary)
        print(
            "[run] phase-scan finished: "
            f"status={scan_summary['run_status']}, summary={os.path.join(args.output_folder, run_summary_filename)}"
        )
        return

    def run_tenax_dmrg_path() -> None:
        nonlocal geometry, tenax_mpo, dmrg_info, dmrg_energy, dmrg_state_obj
        nonlocal dmrg_scalar_correlations, dmrg_bond_rows, dmrg_structure_factor_rows
        nonlocal dmrg_uniform_observables, dmrg_real_space_patterns
        nonlocal backend_used, symmetry_preflight_report, effective_symmetry_mode

        dmrg_scalar_correlations = {}
        dmrg_bond_rows = []
        dmrg_structure_factor_rows = []
        dmrg_uniform_observables = {}
        dmrg_real_space_patterns = {}
        with profile_stage("geometry construction"):
            geometry = build_lattice_geometry(
                lattice=lattice_name,
                length_x=args.length_x,
                length_y=args.length_y,
                circumference_x=circumference_x,
                circumference_y=circumference_y,
            )
        if geometry.number_of_sites > args.max_dmrg_sites:
            raise ValueError(
                f"Finite DMRG safety cap for profile '{ACTIVE_RESOURCE_PROFILE}' is N <= {args.max_dmrg_sites}, "
                f"but requested N={geometry.number_of_sites}. Increase --max-dmrg-sites "
                "only for aragorn/beehive or a dedicated run."
            )
        symmetry_preflight_report = run_symmetry_preflight_for_geometry(geometry)
        effective_symmetry_mode = str(
            symmetry_preflight_report.get("effective_mode_for_tenax", effective_symmetry_mode)
        )
        if geometry_plot_status == "not_attempted":
            save_geometry_before_sweep(geometry)
        tenax_mps, tenax_mpo, dmrg_info = run_tenax_cylindrical_dmrg(
            geometry=geometry,
            model_spec=model_spec,
            alpha=args.alpha,
            beta=args.beta,
            coupling_j=args.coupling_j,
            external_field_terms=hamiltonian_external_field_terms,
            max_bond_dimension=args.max_bond_dimension,
            max_sweeps=args.max_sweeps,
            truncation_cutoff=args.truncation_cutoff,
            svd_min=args.dmrg_svd_min,
            random_seed=args.seed,
            jx=args.jx,
            jy=args.jy,
            jz=args.jz,
            symmetry_mode=effective_symmetry_mode,
            u1_target_total_sz2=args.u1_target_sz2,
            u1_target_total_tz2=args.u1_target_tz2,
            z2_target_parity=args.z2_target_parity,
            strict_symmetry_selection_rules=args.strict_symmetry_selection_rules,
            allow_symmetry_fallback_to_dense=args.symmetry_allow_dense_fallback,
            initial_state_style=args.initial_state,
            show_progress=show_progress,
        )
        dmrg_energy = float(dmrg_info["E"])
        dmrg_state_obj = tenax_mps
        dmrg_correlations: Dict[str, np.ndarray] = {}
        with profile_stage("observables"):
            if calculate_correlations:
                dmrg_correlations = collect_correlation_matrices_from_tenax(
                    tenax_mps,
                    geometry,
                    model_spec=model_spec,
                    show_progress=show_progress,
                )
                dmrg_scalar_correlations = build_spin_orbital_scalar_correlations(dmrg_correlations)
                if calculate_bond_energies:
                    dmrg_bond_rows = all_bond_energies(
                        geometry,
                        dmrg_correlations,
                        model_spec,
                        args.alpha,
                        args.beta,
                        args.coupling_j,
                        jx=args.jx,
                        jy=args.jy,
                        jz=args.jz,
                        show_progress=show_progress,
                        progress_desc="DMRG bond energies",
                    )
                if calculate_structure_factors:
                    dmrg_structure_factor_rows = all_high_symmetry_structure_factors(
                        dmrg_scalar_correlations,
                        geometry,
                        lattice=lattice_name,
                        show_progress=show_progress,
                        progress_desc="DMRG structure factors",
                    )
            if calculate_uniform_observables:
                try:
                    dmrg_uniform_observables = collect_uniform_z_observables_from_tenax(
                        tenax_mps,
                        geometry,
                        model_spec=model_spec,
                    )
                except Exception as exc:
                    dmrg_uniform_observables = {
                        "warning": f"Failed to compute DMRG uniform z observables: {exc}"
                    }
        backend_used = "tenax"

    def run_tenpy_dmrg_path(tenpy_path_label: str, backend_label: str) -> None:
        nonlocal geometry, dmrg_info, dmrg_energy, dmrg_state_obj
        nonlocal dmrg_scalar_correlations, dmrg_bond_rows, dmrg_structure_factor_rows
        nonlocal dmrg_uniform_observables, dmrg_real_space_patterns
        nonlocal backend_used, symmetry_preflight_report, effective_symmetry_mode

        compatibility_issue = tenpy_backend_compatibility_issue()
        if compatibility_issue is not None:
            raise RuntimeError(f"{tenpy_path_label} is not compatible: {compatibility_issue}")
        dmrg_scalar_correlations = {}
        dmrg_bond_rows = []
        dmrg_structure_factor_rows = []
        dmrg_uniform_observables = {}
        dmrg_real_space_patterns = {}
        yl = _load_tenpy_backend_module()
        stage_start = _start_stage(f"{tenpy_path_label} DMRG", show_progress)
        try:
            with profile_stage("geometry construction"):
                geometry = yl.build_honeycomb_cylinder_geometry(
                    length_x=args.length_x,
                    length_y=args.length_y,
                    circumference_x=circumference_x,
                    circumference_y=circumference_y,
                )
            if geometry.number_of_sites > args.max_dmrg_sites:
                raise ValueError(
                    f"Finite DMRG safety cap for profile '{ACTIVE_RESOURCE_PROFILE}' is N <= {args.max_dmrg_sites}, "
                    f"but requested N={geometry.number_of_sites}. Increase --max-dmrg-sites "
                    "only for aragorn/beehive or a dedicated run."
                )
            if symmetry_preflight_report is None:
                symmetry_preflight_report = run_symmetry_preflight_for_geometry(geometry)
                effective_symmetry_mode = str(
                    symmetry_preflight_report.get("effective_mode_for_tenax", effective_symmetry_mode)
                )
            if geometry_plot_status == "not_attempted":
                save_geometry_before_sweep(geometry)
            psi, _, dmrg_info = yl.run_cylindrical_dmrg(
                geometry=geometry,
                alpha=args.alpha,
                beta=args.beta,
                coupling_j=args.coupling_j,
                max_bond_dimension=args.max_bond_dimension,
                max_sweeps=args.max_sweeps,
                truncation_cutoff=args.truncation_cutoff,
                svd_min=args.dmrg_svd_min,
                random_seed=args.seed,
                product_state_style=args.initial_state,
                external_field_terms=hamiltonian_external_field_terms,
                symmetry_reductions=symmetry_reduction_settings,
                show_progress=show_progress,
            )
            dmrg_energy = float(dmrg_info["E"])
            dmrg_state_obj = psi
            with profile_stage("observables"):
                if calculate_correlations:
                    dmrg_correlations = yl.collect_correlation_matrices_from_dmrg(psi, show_progress=show_progress)
                    scalar_native = yl.build_spin_orbital_scalar_correlations(dmrg_correlations)
                    dmrg_scalar_correlations = {
                        "S": scalar_native["spin_scalar"],
                        "T": scalar_native["orbital_scalar"],
                        "ST": scalar_native["mixed_scalar"],
                    }
                    if calculate_bond_energies:
                        dmrg_bond_rows = yl.all_bond_energies(
                            geometry,
                            dmrg_correlations,
                            args.alpha,
                            args.beta,
                            args.coupling_j,
                            show_progress=show_progress,
                        )
                    if calculate_structure_factors:
                        dmrg_structure_factor_rows = yl.all_high_symmetry_structure_factors(
                            scalar_native,
                            geometry,
                            show_progress=show_progress,
                        )
            backend_used = backend_label
        finally:
            _end_stage(f"{tenpy_path_label} DMRG", stage_start, show_progress)

    def run_quimb_peps_path() -> None:
        nonlocal geometry, dmrg_info, dmrg_energy, dmrg_state_obj, tenax_mpo
        nonlocal dmrg_scalar_correlations, dmrg_bond_rows, dmrg_structure_factor_rows
        nonlocal dmrg_uniform_observables, dmrg_real_space_patterns
        nonlocal backend_used, symmetry_preflight_report, effective_symmetry_mode
        nonlocal entanglement_warning

        dmrg_scalar_correlations = {}
        dmrg_bond_rows = []
        dmrg_structure_factor_rows = []
        dmrg_uniform_observables = {}
        dmrg_real_space_patterns = {}
        tenax_mpo = None
        with profile_stage("geometry construction"):
            geometry = build_lattice_geometry(
                lattice=lattice_name,
                length_x=args.length_x,
                length_y=args.length_y,
                circumference_x=circumference_x,
                circumference_y=circumference_y,
            )
        symmetry_preflight_report = run_symmetry_preflight_for_geometry(geometry)
        effective_symmetry_mode = str(
            symmetry_preflight_report.get("effective_mode_for_tenax", effective_symmetry_mode)
        )
        if geometry_plot_status == "not_attempted":
            save_geometry_before_sweep(geometry)
        import peps_backend as quimb_peps_backend

        peps_result = quimb_peps_backend.run_quimb_peps_calculation(
            geometry=geometry,
            model_spec=model_spec,
            lattice_name=lattice_name,
            alpha=float(args.alpha),
            beta=float(args.beta),
            coupling_j=float(args.coupling_j),
            jx=float(args.jx),
            jy=float(args.jy),
            jz=float(args.jz),
            external_field_terms=hamiltonian_external_field_terms,
            max_sites=int(args.max_peps_sites),
            max_bond_dimension=int(args.peps_max_bond_dimension),
            max_sweeps=int(args.peps_max_sweeps),
            truncation_cutoff=float(args.truncation_cutoff),
            tau=float(args.peps_tau),
            random_seed=int(args.seed),
            initial_state_style=str(args.initial_state),
            ctm_chi=int(args.peps_ctm_chi),
            entanglement_max_dense_dim=int(args.peps_entanglement_max_dense_dim),
            classifier_thresholds=phase_classifier_thresholds_from_args(args),
            compute_correlations=bool(calculate_correlations),
            compute_bond_energies=bool(calculate_bond_energies),
            compute_structure_factors=bool(calculate_structure_factors),
            compute_uniform_observables=bool(calculate_uniform_observables),
            compute_entanglement=bool(calculate_entanglement),
            entropy_orders=ENTROPY_ORDERS,
            show_progress=show_progress,
            args=args,
            symmetry_reductions=symmetry_reduction_settings,
            use_sz_conserved=bool(args.use_sz_conserved),
            symmetric=False,
            peps_symmetry_mode=args.peps_symmetry_mode,
            peps_strict_symmetry=bool(args.peps_strict_symmetry),
            peps_allow_dense_fallback=bool(args.peps_allow_dense_fallback),
        )
        dmrg_info = dict(peps_result["info"])
        dmrg_info["finite_reference_key"] = "peps"
        dmrg_energy = float(dmrg_info.get("ground_state_energy", dmrg_info.get("E")))
        dmrg_state_obj = None
        dmrg_scalar_correlations = dict(peps_result.get("scalar_correlations", {}))
        dmrg_bond_rows = list(peps_result.get("bond_rows", []))
        dmrg_structure_factor_rows = list(peps_result.get("structure_factor_rows", []))
        dmrg_uniform_observables = dict(peps_result.get("uniform_observables", {}))
        peps_entropy = peps_result.get("entanglement")
        if isinstance(peps_entropy, dict):
            if peps_entropy.get("status") == "completed":
                entropy_profiles["PEPS"] = peps_entropy
            elif calculate_entanglement:
                entanglement_warning = str(
                    peps_entropy.get("warning")
                    or peps_entropy.get("reason")
                    or "PEPS entanglement profile was not produced."
                )
        backend_used = "quimb_peps"

    def run_quimb_ipeps_primary_path() -> None:
        nonlocal geometry, dmrg_info, dmrg_energy, dmrg_state_obj, tenax_mpo
        nonlocal dmrg_scalar_correlations, dmrg_bond_rows, dmrg_structure_factor_rows
        nonlocal dmrg_uniform_observables, dmrg_real_space_patterns
        nonlocal backend_used, symmetry_preflight_report, effective_symmetry_mode

        dmrg_scalar_correlations = {}
        dmrg_bond_rows = []
        dmrg_structure_factor_rows = []
        dmrg_uniform_observables = {}
        dmrg_real_space_patterns = {}
        tenax_mpo = None
        with profile_stage("geometry construction"):
            geometry = build_lattice_geometry(
                lattice=lattice_name,
                length_x=args.length_x,
                length_y=args.length_y,
                circumference_x=circumference_x,
                circumference_y=circumference_y,
            )
        symmetry_preflight_report = run_symmetry_preflight_for_geometry(geometry)
        effective_symmetry_mode = str(
            symmetry_preflight_report.get("effective_mode_for_tenax", effective_symmetry_mode)
        )
        if geometry_plot_status == "not_attempted":
            save_geometry_before_sweep(geometry)
        import peps_backend as quimb_ipeps_backend

        ipeps_info = quimb_ipeps_backend.run_quimb_ipeps_scan(
            geometry=geometry,
            model_spec=model_spec,
            lattice_name=lattice_name,
            alpha=float(args.alpha),
            beta=float(args.beta),
            alpha_values=[float(args.alpha)],
            beta_values=[float(args.beta)],
            coupling_j=float(args.coupling_j),
            jx=float(args.jx),
            jy=float(args.jy),
            jz=float(args.jz),
            external_field_terms=hamiltonian_external_field_terms,
            max_unit_cell_sites=int(args.max_ipeps_unit_cell_sites),
            max_bond_dimension=int(args.ipeps_max_bond_dimension),
            max_iterations=int(args.ipeps_max_iterations),
            truncation_cutoff=float(args.truncation_cutoff),
            random_seed=int(args.seed),
            initial_state_style=str(args.initial_state),
            tau=float(args.ipeps_tau),
            ctm_chi=int(args.ipeps_ctm_chi),
            symmetry_reductions=symmetry_reduction_settings,
            args=args,
            use_sz_conserved=bool(args.use_sz_conserved),
            symmetric=False,
            ipeps_symmetry_mode=args.ipeps_symmetry_mode,
            ipeps_strict_symmetry=bool(args.ipeps_strict_symmetry),
            ipeps_allow_dense_fallback=bool(args.ipeps_allow_dense_fallback),
            unit_cell_kind=args.ipeps_unit_cell_kind,
            use_translation_symmetry=bool(args.ipeps_use_translation_symmetry),
            contraction_method=args.ipeps_contraction_method,
            classifier_thresholds=phase_classifier_thresholds_from_args(args),
            show_progress=show_progress,
        )
        ipeps_info = _normalize_ipeps_result_schema(ipeps_info)
        ipeps_info["requested_backend"] = "quimb"
        ipeps_info["finite_reference_key"] = "ipeps"
        energy_per_site = _finite_float_from_mapping(ipeps_info, "ground_state_energy_per_site", "energy_per_site")
        if energy_per_site is None:
            raise RuntimeError(f"quimb iPEPS did not produce a finite energy_per_site: {ipeps_info}")
        dmrg_energy = float(energy_per_site) * float(max(1, geometry.number_of_sites))
        ipeps_info["E"] = dmrg_energy
        ipeps_info["ground_state_energy"] = dmrg_energy
        dmrg_info = ipeps_info
        dmrg_state_obj = None
        dmrg_bond_rows = list(ipeps_info.get("bond_energies", []))
        dmrg_uniform_observables = dict(ipeps_info.get("local_observables", {}))
        backend_used = "quimb_ipeps"

    backend_request = str(args.backend).strip().lower()
    if backend_request == "tenax":
        run_tenax_dmrg_path()
    elif backend_request == "tenpy":
        run_tenpy_dmrg_path("TeNPy backend", "tenpy")
    elif backend_request == "quimb":
        if str(args.method) == "ipeps":
            run_quimb_ipeps_primary_path()
        else:
            try:
                run_quimb_peps_path()
            except Exception as peps_exc:
                backend_warning = f"quimb PEPS backend skipped; fallback to DMRG: {peps_exc}"
                if show_progress:
                    print(f"[backend] {backend_warning}")
                if tenpy_backend_issue is None:
                    run_tenpy_dmrg_path("TeNPy fallback after quimb PEPS skip", "tenpy_fallback_after_quimb_peps")
                    dmrg_info["requested_backend"] = "quimb"
                    dmrg_info["fallback_from"] = "quimb_peps"
                    dmrg_info["fallback_reason"] = str(peps_exc)
                else:
                    if show_progress:
                        print(f"[backend] TeNPy fallback is not compatible: {tenpy_backend_issue}; trying Tenax.")
                    run_tenax_dmrg_path()
                    backend_used = "tenax_fallback_after_quimb_peps"
                    dmrg_info["requested_backend"] = "quimb"
                    dmrg_info["fallback_from"] = "quimb_peps"
                    dmrg_info["fallback_reason"] = str(peps_exc)
                    dmrg_info["tenpy_fallback_skip_reason"] = str(tenpy_backend_issue)
    elif backend_request == "auto":
        if tenpy_backend_issue is None:
            try:
                run_tenpy_dmrg_path("TeNPy auto-primary backend", "tenpy")
            except Exception as tenpy_exc:
                backend_warning = f"TeNPy primary backend failed; fallback to Tenax: {tenpy_exc}"
                if show_progress:
                    print(f"[backend] {backend_warning}")
                try:
                    run_tenax_dmrg_path()
                    backend_used = "tenax_fallback_after_tenpy"
                    dmrg_info["requested_backend"] = "auto"
                    dmrg_info["fallback_from"] = "tenpy"
                    dmrg_info["fallback_reason"] = str(tenpy_exc)
                except Exception as tenax_exc:
                    raise RuntimeError(
                        "Auto backend failed: TeNPy primary path failed, and Tenax fallback "
                        f"also failed. TeNPy error: {tenpy_exc}; Tenax error: {tenax_exc}"
                    ) from tenax_exc
        else:
            if show_progress:
                print(f"[backend] auto selects Tenax because TeNPy is not compatible: {tenpy_backend_issue}")
            run_tenax_dmrg_path()
            backend_used = "tenax_auto"
            dmrg_info["requested_backend"] = "auto"
            dmrg_info["tenpy_primary_skip_reason"] = str(tenpy_backend_issue)

    with profile_stage("observables"):
        try:
            if calculate_entanglement and dmrg_state_obj is not None:
                if str(backend_used).startswith("tenax"):
                    dmrg_entropy_profile = compute_tenax_finite_mps_entropy_profile(
                        dmrg_state_obj,
                        orders=ENTROPY_ORDERS,
                        show_progress=show_progress,
                    )
                else:
                    dmrg_entropy_profile = compute_tenpy_finite_mps_entropy_profile(
                        dmrg_state_obj,
                        orders=ENTROPY_ORDERS,
                        show_progress=show_progress,
                    )
                entropy_profiles["DMRG"] = dmrg_entropy_profile
            elif not calculate_entanglement:
                entanglement_warning = "DMRG entanglement entropy calculation skipped by calculate_entanglement=false."
        except Exception as exc:
            entanglement_warning = f"Failed to compute DMRG entanglement profile: {exc}"

        if calculate_real_space_patterns and len(dmrg_scalar_correlations) > 0:
            try:
                dmrg_real_space_patterns = build_reference_site_correlation_patterns(
                    geometry,
                    dmrg_scalar_correlations,
                    reference_site_idx=args.reference_site_idx,
                )
            except Exception as exc:
                dmrg_real_space_patterns = {
                    "status": "failed",
                    "warning": f"Failed to extract DMRG reference-site patterns: {exc}",
                }
        elif calculate_real_space_patterns:
            dmrg_real_space_patterns = {
                "status": "skipped",
                "reason": "No DMRG scalar correlations were computed.",
            }

    configure_run_output_names(geometry)
    lattice_label = lattice_display_name(lattice_name)
    model_short_label = model_simplified_name(model_spec)
    size_short_label = geometry_size_filename_label(
        geometry,
        lattice_name,
        args.length_x,
        length_y=args.length_y,
        circumference_x=circumference_x,
        circumference_y=circumference_y,
    )
    size_display_label = geometry_size_display_label(
        geometry,
        lattice_name,
        args.length_x,
        length_y=args.length_y,
        circumference_x=circumference_x,
        circumference_y=circumference_y,
    )
    model_label = (
        f"{model_spec.model_family}, spin={model_spec.spin_rep}, orbital={model_spec.orbital_rep}, axis={model_spec.ising_axis}"
    )
    external_field_summary = external_field_construction_summary(
        treatment=args.external_field_treatment,
        axis=args.external_field_axis,
        field_vector=resolved_field_vector,
        mu_b=args.mu_b,
        field_sign=args.field_sign,
        sigma_factor=args.field_sigma_factor,
        field_terms=hamiltonian_external_field_terms,
    )
    finite_is_peps = str(backend_used).startswith("quimb_peps")
    primary_is_ipeps = str(backend_used).startswith("quimb_ipeps")
    finite_method_label = "iPEPS" if primary_is_ipeps else ("PEPS" if finite_is_peps else "DMRG")
    finite_method_long_label = (
        "quimb infinite iPEPS"
        if primary_is_ipeps
        else ("quimb finite PEPS" if finite_is_peps else "finite DMRG")
    )

    summary: Dict[str, Any] = {
        "model_name": f"{lattice_label} spin-orbital model ({model_label}, {finite_method_long_label})",
        "model_simplified_name": model_short_label,
        "model_size_name": size_short_label,
        "run_output_prefix": run_file_prefix,
        "monitor_data_name": run_summary_filename,
        "plot_title_label": run_plot_title_label,
        "resource_profile": ACTIVE_RESOURCE_PROFILE,
        "active_resource_settings": ACTIVE_RESOURCE_SETTINGS,
        "external_field": external_field_summary,
        "run_status": "running",
        "parameters": vars(args),
        "observable_controls": observable_controls,
        "model_construction_annotations": {
            "shared_workstation_constraints": {
                "active_resource_profile": ACTIVE_RESOURCE_PROFILE,
                "finite_dmrg": {
                    "max_sites": int(args.max_dmrg_sites),
                    "max_bond_dimension": int(args.max_bond_dimension),
                    "max_sweeps": int(args.max_sweeps),
                    "svd_min": float(args.dmrg_svd_min),
                    "truncation_cutoff": float(args.truncation_cutoff),
                    "excited_state_search": {
                        "method": "finite_dmrg_penalty_excited_state",
                        "overlap_tol": float(args.dmrg_excited_overlap_tol),
                        "energy_tol": float(args.dmrg_excited_energy_tol),
                        "variance_tol": float(args.dmrg_excited_variance_tol),
                        "max_attempts_per_penalty_weight": int(args.dmrg_excited_max_attempts),
                        "requires_complete_ed_resolved_ground_mps_manifold": True,
                    },
                    "note": "Default geometry N=8 is chosen so DMRG, ED, and iDMRG can all be compared.",
                },
                "finite_peps": {
                    "max_sites": int(args.max_peps_sites),
                    "max_bond_dimension": int(args.peps_max_bond_dimension),
                    "bond_dimension_cap": int(args.peps_bond_dimension_cap),
                    "max_sweeps": int(args.peps_max_sweeps),
                    "sweep_cap": int(args.peps_sweep_cap),
                    "ctm_chi": int(args.peps_ctm_chi),
                    "ctm_chi_cap": int(args.peps_ctm_chi_cap),
                    "tau": float(args.peps_tau),
                    "entanglement_max_dense_dim": int(args.peps_entanglement_max_dense_dim),
                    "symmetry_mode": str(args.peps_symmetry_mode),
                    "strict_symmetry": bool(args.peps_strict_symmetry),
                    "allow_dense_fallback": bool(args.peps_allow_dense_fallback),
                    "note": (
                        "Finite PEPS controls are independent of finite-DMRG chi/sweeps. "
                        "The current quimb implementation records U(1)_Tz requests but uses dense tensors unless symmetric tensor support is enabled later."
                    ),
                },
                "ed": {
                    "backend": str(args.ed_backend),
                    "max_sites": int(args.max_ed_sites),
                    "max_hilbert_dimension": int(args.max_ed_hilbert_dim),
                    "max_eigenstates": int(args.ed_max_eigenstates),
                    "solver": str(args.ed_solver),
                    "use_sz_conserved": bool(args.use_sz_conserved),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "ed_symmetry_plan": getattr(args, "ed_symmetry_plan", {"status": "pending"}),
                    "ed_symmetry_controls": {
                        "engine": str(args.ed_symmetry_engine),
                        "c3_mode": str(args.ed_c3_mode),
                        "c3_q_blocks": str(args.ed_c3_q_blocks),
                        "z2_mode": str(args.ed_z2_mode),
                        "z2_kind": str(args.ed_z2_kind),
                    },
                    "sz_conserved_eigenstates": int(SZ_CONSERVED_ED_EIGENSTATES),
                    "sparse_tol": float(args.ed_sparse_tol),
                    "sparse_maxiter": (
                        int(args.ed_sparse_maxiter)
                        if int(args.ed_sparse_maxiter) > 0
                        else None
                    ),
                    "ground_manifold_abs_tol": float(args.ed_ground_manifold_abs_tol),
                    "ground_manifold_rel_tol": float(args.ed_ground_manifold_rel_tol),
                    "note": (
                        "ED is skipped automatically above either cap; sparse mode "
                        "solves only the requested lowest eigenstates."
                    ),
                },
                "quspin_ed": {
                    "settings": quspin_ed_settings,
                    "status": (
                        "selected"
                        if str(args.ed_backend) == "quspin"
                        else "available_for_ed_backend_selection"
                    ),
                    "note": (
                        "QuSpin receives the ED-specific symmetry plan. Existing basis builders apply the "
                        "supported subset and record any projector-only symmetries as planned."
                    ),
                },
                "finite_temperature_ed": {
                    "run": bool(args.run_finite_temperature),
                    "max_sites": int(args.thermal_max_sites),
                    "max_hilbert_dimension": int(args.thermal_max_hilbert_dim),
                    "full_spectrum_max_dimension": int(args.thermal_full_spectrum_max_dim),
                    "max_eigenstates": int(args.thermal_max_eigenstates),
                    "temperature_min": float(args.temperature_min),
                    "temperature_max": float(args.temperature_max),
                    "temperature_points": int(args.temperature_points),
                    "temperature_scale": str(args.temperature_scale),
                    "note": (
                        "Full-spectrum finite-T ED is exact only below the full-spectrum cap; "
                        "larger allowed clusters use a low-energy truncated trace."
                    ),
                },
                "phase_scan": {
                    "phase_diagram_enabled": bool(args.phase_diagram),
                    "run": bool(args.run_phase_scan),
                    "scan_only": bool(args.phase_scan_only),
                    "mode": str(args.phase_scan_mode),
                    "quantum_methods": list(args.phase_scan_quantum_methods),
                    "selected_outputs": list(args.phase_scan_methods),
                    "alpha_min": float(args.phase_scan_alpha_min),
                    "alpha_max": float(args.phase_scan_alpha_max),
                    "alpha_points": int(args.phase_scan_alpha_points),
                    "beta_min": float(args.phase_scan_beta_min),
                    "beta_max": float(args.phase_scan_beta_max),
                    "beta_points": int(args.phase_scan_beta_points),
                    "external_scan_mode": str(args.external_scan_mode),
                    "external_scan_field_min": float(args.external_scan_field_min),
                    "external_scan_field_max": float(args.external_scan_field_max),
                    "external_scan_field_points": int(args.external_scan_field_points),
                    "external_scan_ed_bands": int(args.external_scan_ed_bands),
                    "quantum_ed_max_sites": int(args.phase_scan_ed_max_sites),
                    "quantum_ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
                    "quantum_ed_solver": str(args.ed_solver),
                    "quantum_ed_use_sz_block": bool(args.use_sz_block),
                    "dmrg_backend": str(args.backend),
                    "dmrg_effective_symmetry_mode": str(effective_symmetry_mode),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "quantum_ed_sparse_tol": float(args.ed_sparse_tol),
                    "quantum_ed_sparse_maxiter": (
                        int(args.ed_sparse_maxiter)
                        if int(args.ed_sparse_maxiter) > 0
                        else None
                    ),
                    "classical_restarts": int(args.phase_scan_classical_restarts),
                    "classical_sweeps": int(args.phase_scan_classical_sweeps),
                    "classifier_thresholds": {
                        "quantum_weak_order": float(args.phase_scan_quantum_weak_order_threshold),
                        "classical_weak_order": float(args.phase_scan_classical_weak_order_threshold),
                        "quantum_bond_nematicity": float(args.phase_scan_quantum_nematicity_threshold),
                        "classical_bond_nematicity": float(args.phase_scan_classical_nematicity_threshold),
                        "plaquette_flux_target": float(args.phase_scan_plaquette_flux_target),
                        "plaquette_flux_tolerance": float(args.phase_scan_plaquette_flux_tolerance),
                    },
                    "note": (
                        "The phase diagrams are generated from scanned ground-state structure "
                        "patterns, plaquette-flux diagnostics, and bond-energy diagnostics, then classified with recorded thresholds."
                    ),
                },
                "idmrg": {
                    "max_bond_dimension": int(args.idmrg_max_bond_dimension),
                    "max_iterations": int(args.idmrg_max_iterations),
                    "svd_min": float(args.idmrg_svd_min),
                    "truncation_cutoff": float(args.truncation_cutoff),
                    "bulk_kind": str(args.idmrg_bulk_kind),
                    "max_local_dim": int(args.idmrg_max_local_dim),
                    "use_translation_symmetry": bool(args.idmrg_use_translation_symmetry),
                    "translation_symmetry": {
                        "enabled": bool(args.idmrg_use_translation_symmetry),
                        "implemented_as": "infinite repeated MPS unit cell along x",
                    },
                    "note": "iDMRG chi is intentionally independent of finite-DMRG chi to avoid workstation OOM.",
                },
                "ipeps": {
                    "max_unit_cell_sites": int(args.max_ipeps_unit_cell_sites),
                    "max_bond_dimension": int(args.ipeps_max_bond_dimension),
                    "bond_dimension_cap": int(args.ipeps_bond_dimension_cap),
                    "max_iterations": int(args.ipeps_max_iterations),
                    "iteration_cap": int(args.ipeps_iteration_cap),
                    "ctm_chi": int(args.ipeps_ctm_chi),
                    "ctm_chi_cap": int(args.ipeps_ctm_chi_cap),
                    "tau": float(args.ipeps_tau),
                    "symmetry_mode": str(args.ipeps_symmetry_mode),
                    "strict_symmetry": bool(args.ipeps_strict_symmetry),
                    "allow_dense_fallback": bool(args.ipeps_allow_dense_fallback),
                    "unit_cell_kind": str(args.ipeps_unit_cell_kind),
                    "use_translation_symmetry": bool(args.ipeps_use_translation_symmetry),
                    "translation_symmetry": {
                        "enabled": bool(args.ipeps_use_translation_symmetry),
                        "implemented_as": "periodic repeated PEPS unit cell",
                    },
                    "contraction_method": str(args.ipeps_contraction_method),
                    "ctmrg_enabled": str(args.ipeps_contraction_method) == "ctmrg",
                    "unit_cell_candidates": list(IPEPS_UNIT_CELL_CANDIDATES),
                    "note": (
                        "iPEPS controls are independent of iDMRG chi/iterations. "
                        "Internal tensor symmetry is separate from the variational unit-cell ansatz."
                    ),
                },
            },
            "bond_terms": {
                "yao_lee": (
                    "For each gamma bond: -J[alpha S_i.S_j - 2 S_i_gamma S_j_gamma - beta]"
                    "[T_i.T_j - beta], with tilde-T currently treated as standard T."
                ),
                "ising_like": (
                    "For each bond, the chosen ising_axis replaces gamma in the same "
                    "S/T/ST channels."
                ),
                "orbital_rep_0": (
                    "The orbital Hilbert space is removed for spin-only benchmark families."
                ),
            },
            "external_field": external_field_summary,
            "symmetry": {
                "requested_reductions": list(args.symmetry_reductions),
                "applied_reductions": symmetry_reduction_settings,
                "requested_mode": str(args.symmetry_mode),
                "effective_mode_for_tenax": str(effective_symmetry_mode),
                "precheck_enabled": bool(args.symmetry_precheck),
                "strict_precheck": bool(args.strict_symmetry_precheck),
                "allow_dense_fallback": bool(args.symmetry_allow_dense_fallback),
                "precheck": symmetry_preflight_report,
                "u1": {
                    "charge_encoding": _u1_charge_encoding_summary(),
                    "valid_when": (
                        "The Hamiltonian conserves total Sz and orbital tau_z/Tz. "
                        "Examples: ising_like with ising_axis=z; spin-only Heisenberg/XY/XXZ/XYZ "
                        "when transverse Sx/Sy couplings are paired equally."
                    ),
                    "invalid_when": (
                        "The yao_lee Eq. 7 x/y gamma channels contain single-axis flip terms "
                        "and cannot be represented as strict U1 without changing the Hamiltonian."
                    ),
                    "fallback_policy": (
                        "Dense fallback is controlled by symmetry_allow_dense_fallback. "
                        "When disabled, a failed requested symmetry raises instead of silently losing speedup."
                    ),
                },
                "z2": (
                    "Tenax 0.2 AutoMPO is U1-only for symmetric MPO construction here; "
                    "Z2 conservation is reported by the precheck but is not used for block-sparse MPO speedup."
                ),
                "quspin": {
                    "settings": quspin_ed_settings,
                    "auto_policy": (
                        "Use the ED-first symmetry plan, then apply only the currently supported QuSpin subset."
                    ),
                },
            },
            "observables_note": (
                "Bond-energy diagrams report exchange/bond terms only; external Zeeman "
                "one-site terms are recorded separately in external_field."
            ),
        },
        "model_spec": {
            "spin_rep": model_spec.spin_rep,
            "orbital_rep": model_spec.orbital_rep,
            "model_family": model_spec.model_family,
            "ising_axis": model_spec.ising_axis,
            "spin_value": model_spec.spin_value,
            "orbital_value": model_spec.orbital_value,
            "physical_dim": model_spec.physical_dim,
            "orbital_is_trivial": is_trivial_orbital(model_spec),
            "effective_interaction": (
                "yao_lee_eq7_trivial_orbital_limit"
                if (model_spec.model_family == "yao_lee" and is_trivial_orbital(model_spec))
                else model_spec.model_family
            ),
        },
        "backend_used": backend_used,
        "geometry": {
            "lattice": lattice_name,
            "length_x": int(args.length_x),
            "length_y": int(args.length_y),
            "circumference_x": bool(circumference_x),
            "circumference_y": bool(circumference_y),
            "number_of_sites": geometry.number_of_sites,
            "number_of_bonds": len(geometry.bond_list),
            "size_label": size_display_label,
            "mps_path": mps_path_quality(geometry),
        },
        "dmrg": {
            "ground_state_energy": dmrg_energy,
            "energy_per_site": dmrg_energy / geometry.number_of_sites,
            "info": dmrg_info,
            "phase_observables": dmrg_info.get("phase_observables"),
            "entanglement": entropy_profiles.get(finite_method_label) or dmrg_info.get("entanglement"),
            "bond_energies": dmrg_bond_rows,
            "structure_factors": dmrg_structure_factor_rows,
            "real_space_patterns": dmrg_real_space_patterns,
            "uniform_observables": dmrg_uniform_observables,
        },
        "stages": {
            "dmrg": "not_requested" if primary_is_ipeps else "completed",
            "peps": "completed" if finite_is_peps else "not_requested",
            "dmrg_plots": "running",
            "idmrg": (
                "not_requested"
                if (finite_is_peps or primary_is_ipeps)
                else ("pending" if args.run_idmrg else "not_requested")
            ),
            "ipeps": (
                "completed"
                if primary_is_ipeps
                else ("pending" if finite_is_peps and args.run_idmrg else "not_requested")
            ),
            "ed": "pending" if args.run_ed else "not_requested",
            "dmrg_excited_state_search": "pending" if args.run_ed else "not_requested",
            "finite_temperature": "pending" if args.run_finite_temperature else "not_requested",
            "phase_scan": "pending" if args.run_phase_scan else "not_requested",
        },
        "outputs": {
            "run_summary_json": run_summary_filename,
            "monitor_data_json": run_summary_filename,
        },
    }
    if finite_is_peps:
        summary["peps"] = dict(summary["dmrg"])
        summary["peps"]["backend"] = "quimb_peps"
        summary["peps_symmetry_report"] = dmrg_info.get("peps_symmetry_report") or dmrg_info.get("symmetry")
        if summary["peps_symmetry_report"] is not None:
            summary["peps"]["peps_symmetry_report"] = summary["peps_symmetry_report"]
        summary["outputs"]["peps"] = {
            "status": str(dmrg_info.get("status", "completed")),
            "energy_per_site": summary["dmrg"]["energy_per_site"],
            "ground_state_energy_per_site": summary["dmrg"]["energy_per_site"],
            "plaquette_flux": dmrg_info.get("plaquette_flux")
            or (dmrg_info.get("phase_observables") or {}).get("plaquette_flux"),
            "phase_label": dmrg_info.get("phase_label"),
            "peps_symmetry_report": summary.get("peps_symmetry_report"),
        }
    if primary_is_ipeps:
        summary["ipeps"] = dict(summary["dmrg"])
        summary["ipeps"]["backend"] = "quimb_ipeps"
        summary["ipeps_symmetry_report"] = dmrg_info.get("ipeps_symmetry_report") or dmrg_info.get("symmetry")
        if summary["ipeps_symmetry_report"] is not None:
            summary["ipeps"]["ipeps_symmetry_report"] = summary["ipeps_symmetry_report"]
        summary["outputs"]["ipeps"] = {
            "status": str(dmrg_info.get("status", "completed")),
            "energy_per_site": summary["dmrg"]["energy_per_site"],
            "ground_state_energy_per_site": summary["dmrg"]["energy_per_site"],
            "plaquette_flux": dmrg_info.get("plaquette_flux")
            or (dmrg_info.get("observables") or {}).get("plaquette_flux"),
            "phase_label": dmrg_info.get("phase_label"),
            "ipeps_symmetry_report": summary.get("ipeps_symmetry_report"),
        }
    if backend_warning:
        summary["backend_warning"] = backend_warning
    if entanglement_warning:
        summary["entanglement_warning"] = entanglement_warning
    dmrg_all_plaquette_fluxes = _record_all_plaquette_fluxes(summary, "dmrg", dmrg_info)
    if dmrg_all_plaquette_fluxes:
        summary["dmrg"]["all_plaquette_fluxes"] = dmrg_all_plaquette_fluxes
        if finite_is_peps:
            summary["peps"]["all_plaquette_fluxes"] = dmrg_all_plaquette_fluxes
            _record_all_plaquette_fluxes(summary, "peps", dmrg_info)
    _save_summary_checkpoint(args.output_folder, summary)

    # Save finite variational-method plots immediately, one by one.
    _record_output_status(
        summary,
        "geometry_diagram_png",
        output_filename("geometry_diagram.png"),
        geometry_plot_status,
        geometry_plot_error,
    )
    _save_summary_checkpoint(args.output_folder, summary)
    dmrg_pattern_correlations = (
        dmrg_real_space_patterns.get("correlations")
        if isinstance(dmrg_real_space_patterns, dict)
        else None
    )
    dmrg_reference_site_idx = (
        dmrg_real_space_patterns.get("reference_site_idx")
        if isinstance(dmrg_real_space_patterns, dict)
        else None
    )
    has_spin_pattern_overlay = (
        isinstance(dmrg_pattern_correlations, dict)
        and "S" in dmrg_pattern_correlations
        and dmrg_reference_site_idx is not None
    )

    def save_finite_bond_energy_plot(path: str) -> None:
        if has_spin_pattern_overlay:
            save_phase_representative_pattern(
                geometry,
                dmrg_pattern_correlations["S"],
                int(dmrg_reference_site_idx),
                dmrg_bond_rows,
                path,
                plot_title(f"{finite_method_label} Spin Pattern + Resolved Bond Energy"),
                external_field_vector=resolved_field_vector,
            )
            print(
                "[plot] spin-vector overlay: "
                f"{path} uses arrows above bonds "
                "(spin zorder=50, bond zorder=3, white halo enabled)."
            )
            return
        save_bond_energy_diagram(
            geometry,
            dmrg_bond_rows,
            path,
            plot_title(f"{finite_method_label} Bond-Energy Diagram"),
        )

    should_save_finite_bond_plot = bool(args.plot_bond_energies or args.plot_real_space_patterns) and (
        len(dmrg_bond_rows) > 0 or has_spin_pattern_overlay
    )
    if should_save_finite_bond_plot:
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_bond_energy_diagram_png",
            output_filename("dmrg_bond_energy_diagram.png"),
            save_finite_bond_energy_plot,
            overwrite_existing,
            continue_on_plot_error,
        )
    else:
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_bond_energy_diagram_png",
            output_filename("dmrg_bond_energy_diagram.png"),
            "plot_bond_energies/plot_real_space_patterns is false, or no DMRG bond rows/spin pattern were computed",
        )
    if bool(args.plot_structure_factors) and calculate_structure_factors and len(dmrg_structure_factor_rows) > 0:
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_structure_factors_png",
            output_filename("dmrg_structure_factors.png"),
            lambda path: save_structure_factor_plot(dmrg_structure_factor_rows, path, plot_title(f"{finite_method_label} Structure Factors")),
            overwrite_existing,
            continue_on_plot_error,
        )
    else:
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_structure_factors_png",
            output_filename("dmrg_structure_factors.png"),
            "plot_structure_factors or calculate_structure_factors is false, or no DMRG structure rows were computed",
        )
    if bool(args.plot_correlation_heatmaps) and calculate_correlations and len(dmrg_scalar_correlations) > 0:
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_scalar_correlation_heatmaps_png",
            output_filename("dmrg_scalar_correlation_heatmaps.png"),
            lambda path: save_scalar_correlation_heatmaps(dmrg_scalar_correlations, path, f"{finite_method_label} | {run_plot_title_label}"),
            overwrite_existing,
            continue_on_plot_error,
        )
    else:
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_scalar_correlation_heatmaps_png",
            output_filename("dmrg_scalar_correlation_heatmaps.png"),
            "plot_correlation_heatmaps or calculate_correlations is false, or no DMRG scalar correlations were computed",
        )
    overlay_status = _plot_step_status(summary, "dmrg_bond_energy_diagram_png")
    overlay_reason = (
        "spin-vector pattern is drawn in dmrg_bond_energy_diagram.png above resolved bonds "
        "(spin zorder=50, bond zorder=3, white halo enabled)"
    )
    if has_spin_pattern_overlay and overlay_status in ("saved", "skipped_exists"):
        overlay_diagnostics = summary.setdefault("plot_overlay_diagnostics", {})
        if isinstance(overlay_diagnostics, dict):
            overlay_diagnostics["dmrg"] = {
                "combined_plot": output_filename("dmrg_bond_energy_diagram.png"),
                "spin_vectors_on_top": True,
                "spin_vector_zorder": 50,
                "resolved_bond_zorder": 3,
                "white_halo": True,
            }
        _record_combined_plot_alias(
            summary,
            args.output_folder,
            "dmrg_spin_real_space_pattern_png",
            output_filename("dmrg_bond_energy_diagram.png"),
            overlay_reason,
        )
    save_flux_crystal_output(
        summary,
        geometry,
        dmrg_info,
        "flux_crystal_pattern_png",
        "flux_crystal_pattern.png",
        f"{finite_method_label} Plaquette Flux Crystal Pattern",
    )
    summary["stages"]["dmrg_plots"] = "completed"
    _save_summary_checkpoint(args.output_folder, summary)

    # Optional iDMRG workflow (runs after finite DMRG outputs are saved).
    if args.run_idmrg and not primary_is_ipeps:
        if finite_is_peps:
            summary["stages"]["ipeps"] = "running"
            _save_summary_checkpoint(args.output_folder, summary)
            try:
                import peps_backend as quimb_ipeps_backend

                ipeps_info = quimb_ipeps_backend.run_quimb_ipeps_scan(
                    geometry=geometry,
                    model_spec=model_spec,
                    lattice_name=lattice_name,
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    alpha_values=[float(args.alpha)],
                    beta_values=[float(args.beta)],
                    coupling_j=float(args.coupling_j),
                    jx=float(args.jx),
                    jy=float(args.jy),
                    jz=float(args.jz),
                    external_field_terms=hamiltonian_external_field_terms,
                    max_unit_cell_sites=int(args.max_ipeps_unit_cell_sites),
                    max_bond_dimension=int(args.ipeps_max_bond_dimension),
                    max_iterations=int(args.ipeps_max_iterations),
                    truncation_cutoff=float(args.truncation_cutoff),
                    random_seed=int(args.seed),
                    initial_state_style=str(args.initial_state),
                    tau=float(args.ipeps_tau),
                    ctm_chi=int(args.ipeps_ctm_chi),
                    symmetry_reductions=symmetry_reduction_settings,
                    args=args,
                    use_sz_conserved=bool(args.use_sz_conserved),
                    symmetric=False,
                    ipeps_symmetry_mode=args.ipeps_symmetry_mode,
                    ipeps_strict_symmetry=bool(args.ipeps_strict_symmetry),
                    ipeps_allow_dense_fallback=bool(args.ipeps_allow_dense_fallback),
                    unit_cell_kind=args.ipeps_unit_cell_kind,
                    use_translation_symmetry=bool(args.ipeps_use_translation_symmetry),
                    contraction_method=args.ipeps_contraction_method,
                    classifier_thresholds=phase_classifier_thresholds_from_args(args),
                    show_progress=show_progress,
                )
                ipeps_info = _normalize_ipeps_result_schema(ipeps_info)
                ipeps_info["requested_backend"] = "quimb"
                summary["ipeps"] = ipeps_info
                summary["ipeps_symmetry_report"] = ipeps_info.get("ipeps_symmetry_report") or ipeps_info.get("symmetry")
                ipeps_all_plaquette_fluxes = _record_all_plaquette_fluxes(summary, "ipeps", ipeps_info)
                if ipeps_all_plaquette_fluxes:
                    summary["ipeps"]["all_plaquette_fluxes"] = ipeps_all_plaquette_fluxes
                save_flux_crystal_output(
                    summary,
                    geometry,
                    ipeps_info,
                    "ipeps_flux_crystal_pattern_png",
                    "ipeps_flux_crystal_pattern.png",
                    "quimb iPEPS Plaquette Flux Crystal Pattern",
                )
                summary["stages"]["ipeps"] = (
                    "completed"
                    if str(ipeps_info.get("status", "completed")) == "completed"
                    else str(ipeps_info.get("status", "completed_with_warnings"))
                )
            except Exception as exc:
                optional_dependency_missing = isinstance(exc, (ImportError, ModuleNotFoundError))
                if show_progress and optional_dependency_missing:
                    print(f"[backend] skip quimb iPEPS: optional package unavailable :: {exc}")
                summary["ipeps"] = {
                    "status": "skipped" if optional_dependency_missing else "failed",
                    "backend": "quimb_ipeps",
                    "requested_backend": "quimb",
                    "error": str(exc) or exc.__class__.__name__,
                    "energy_per_site": None,
                    "ground_state_energy_per_site": None,
                    "plaquette_flux": {
                        "available": False,
                        "value": None,
                        "W_p": None,
                        "reason": str(exc) or exc.__class__.__name__,
                    },
                }
                summary["stages"]["ipeps"] = "skipped" if optional_dependency_missing else "failed"
                if not continue_on_plot_error and not optional_dependency_missing:
                    raise
            _save_summary_checkpoint(args.output_folder, summary)
        else:
            summary["stages"]["idmrg"] = "running"
            _save_summary_checkpoint(args.output_folder, summary)
            if not bool(args.idmrg_use_translation_symmetry):
                summary["idmrg"] = {
                    "status": "skipped",
                    "reason": "iDMRG translation symmetry was disabled by --no-idmrg-use-translation-symmetry.",
                    "translation_symmetry": {"enabled": False},
                }
                summary["stages"]["idmrg"] = "skipped"
                _save_summary_checkpoint(args.output_folder, summary)
            elif str(backend_used).startswith("tenax") and tenax_mpo is None:
                summary["idmrg"] = {
                    "status": "failed",
                    "error": "Tenax MPO object unavailable after DMRG.",
                }
                summary["stages"]["idmrg"] = "failed"
                _save_summary_checkpoint(args.output_folder, summary)
            else:
                try:
                    if str(backend_used).startswith("tenax"):
                        idmrg_info = run_tenax_idmrg_x_from_finite_mpo(
                            mpo=tenax_mpo,
                            model_spec=model_spec,
                            max_bond_dimension=args.idmrg_max_bond_dimension,
                            max_iterations=args.idmrg_max_iterations,
                            bulk_kind=args.idmrg_bulk_kind,
                            max_local_dim=args.idmrg_max_local_dim,
                            truncation_cutoff=args.truncation_cutoff,
                            svd_min=args.idmrg_svd_min,
                            compute_entanglement=calculate_entanglement,
                            show_progress=show_progress,
                        )
                        tenax_idmrg_energy = _finite_float_from_mapping(
                            idmrg_info,
                            "energy_per_original_site",
                            "ground_state_energy_per_site",
                            "energy_per_site",
                        )
                        if _idmrg_energy_is_suspicious(
                            tenax_idmrg_energy,
                            float(summary["dmrg"]["energy_per_site"]),
                        ):
                            tenax_idmrg_attempt = dict(idmrg_info)
                            yl_idmrg = _load_tenpy_backend_module()
                            idmrg_info = yl_idmrg.run_cylindrical_idmrg(
                                geometry=geometry,
                                alpha=args.alpha,
                                beta=args.beta,
                                coupling_j=args.coupling_j,
                                max_bond_dimension=args.idmrg_max_bond_dimension,
                                max_iterations=args.idmrg_max_iterations,
                                truncation_cutoff=args.truncation_cutoff,
                                svd_min=args.idmrg_svd_min,
                                random_seed=args.seed,
                                product_state_style=args.initial_state,
                                compute_entanglement=calculate_entanglement,
                                external_field_terms=hamiltonian_external_field_terms,
                                symmetry_reductions=symmetry_reduction_settings,
                                show_progress=show_progress,
                            )
                            idmrg_info["backend"] = "tenpy_fallback_after_tenax_idmrg_sanity_check"
                            idmrg_info["tenax_idmrg_attempt"] = tenax_idmrg_attempt
                            idmrg_info["tenax_idmrg_fallback_reason"] = (
                                "Tenax finite-MPO bulk extraction returned an iDMRG energy density "
                                "far outside the finite-DMRG reference scale, so the comparison plot "
                                "uses the TeNPy infinite-MPS path for a sane ground-state benchmark."
                            )
                    else:
                        yl_idmrg = _load_tenpy_backend_module()
                        idmrg_info = yl_idmrg.run_cylindrical_idmrg(
                            geometry=geometry,
                            alpha=args.alpha,
                            beta=args.beta,
                            coupling_j=args.coupling_j,
                            max_bond_dimension=args.idmrg_max_bond_dimension,
                            max_iterations=args.idmrg_max_iterations,
                            truncation_cutoff=args.truncation_cutoff,
                            svd_min=args.idmrg_svd_min,
                            random_seed=args.seed,
                            product_state_style=args.initial_state,
                            compute_entanglement=calculate_entanglement,
                            external_field_terms=hamiltonian_external_field_terms,
                            symmetry_reductions=symmetry_reduction_settings,
                            show_progress=show_progress,
                        )
                    summary["idmrg"] = idmrg_info
                    summary["idmrg"]["translation_symmetry"] = {
                        "enabled": bool(args.idmrg_use_translation_symmetry),
                        "implemented_as": "infinite repeated MPS unit cell along x",
                    }
                    idmrg_all_plaquette_fluxes = _record_all_plaquette_fluxes(summary, "idmrg", idmrg_info)
                    if idmrg_all_plaquette_fluxes:
                        summary["idmrg"]["all_plaquette_fluxes"] = idmrg_all_plaquette_fluxes
                    save_flux_crystal_output(
                        summary,
                        geometry,
                        idmrg_info,
                        "idmrg_flux_crystal_pattern_png",
                        "idmrg_flux_crystal_pattern.png",
                        "iDMRG-x Plaquette Flux Crystal Pattern",
                    )
                    if calculate_entanglement and isinstance(idmrg_info.get("entanglement"), dict):
                        entropy_profiles["iDMRG-x"] = idmrg_info["entanglement"]
                    summary["stages"]["idmrg"] = "completed"
                    _save_summary_checkpoint(args.output_folder, summary)
                except Exception as exc:
                    summary["idmrg"] = {"status": "failed", "error": str(exc)}
                    summary["stages"]["idmrg"] = "failed"
                    if not continue_on_plot_error:
                        raise
                    _save_summary_checkpoint(args.output_folder, summary)

    # Optional ED workflow (runs after all DMRG outputs are already saved).
    local_dim = int(model_spec.physical_dim)
    full_hilbert_dim = int(local_dim ** geometry.number_of_sites)
    requested_ed_backend_name = str(args.ed_backend)
    ed_backend_name = requested_ed_backend_name
    ed_symmetry_plan = (
        getattr(args, "ed_symmetry_plan", {})
        if isinstance(getattr(args, "ed_symmetry_plan", None), dict)
        else {}
    )
    requested_sz_block = bool(ed_symmetry_plan.get("use_sz_block", symmetry_reduction_settings.get("use_sz_block", False)))
    requested_tau_z_block = bool(
        ed_symmetry_plan.get("use_tau_z_block", symmetry_reduction_settings.get("use_tau_z_block", False))
    )
    requested_z2_block = bool(ed_symmetry_plan.get("use_z2_block", symmetry_reduction_settings.get("use_z2_block", False)))
    requested_z2_generator = ed_symmetry_plan.get("z2_generator", symmetry_reduction_settings.get("z2_generator"))
    requested_z2_kind = ed_symmetry_plan.get("z2_kind", requested_z2_generator)
    requested_translation_x_block = bool(ed_symmetry_plan.get("use_translation_x_block", args.use_translation_x_block))
    requested_translation_y_block = bool(ed_symmetry_plan.get("use_translation_y_block", args.use_translation_y_block))
    ed_plan_use_c3_block = bool(ed_symmetry_plan.get("use_c3_block", False))
    ed_actual_field_class = str(ed_symmetry_plan.get("actual_hamiltonian_field_class", "none"))
    effective_ed_engine = str(ed_symmetry_plan.get("effective_engine", ed_symmetry_plan.get("engine", args.ed_symmetry_engine)))
    plan_backend_override_reason = ed_symmetry_plan.get("backend_override_reason")
    ed_backend_override_reason = None
    if effective_ed_engine == "standard_projector" and ed_backend_name == "quspin":
        ed_backend_name = "standard"
        ed_backend_override_reason = (
            str(plan_backend_override_reason)
            if plan_backend_override_reason
            else (
                "ED symmetry engine selected standard_projector; not replacing the in-repo "
                "projector/U1 route "
                "with an incomplete QuSpin site-permutation block."
            )
        )
    elif effective_ed_engine.startswith("quspin") and ed_backend_name != "quspin":
        ed_backend_name = "quspin"
        ed_backend_override_reason = (
            f"ED symmetry engine selected {effective_ed_engine}; using the separate native QuSpin path "
            "for its supported subset."
        )
    quspin_ed_requested = ed_backend_name == "quspin"
    model_requested_reductions = set()
    if isinstance(getattr(args, "model_symmetry_selection", None), dict):
        model_requested_reductions = {
            str(item).strip().lower()
            for item in args.model_symmetry_selection.get("requested_reductions", [])
        }
    quspin_requested_z2_block = bool(requested_z2_block)
    requested_target_sz2 = int(symmetry_reduction_settings.get("target_sz2", args.u1_target_sz2))
    requested_target_tz2 = int(symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2))
    hamiltonian_field_ops = {
        str(op_name)
        for coefficient, op_name in list(hamiltonian_external_field_terms or [])
        if abs(float(coefficient)) > 1e-14
    }
    quspin_use_sz_block = bool(requested_sz_block)
    quspin_sz_reason = None
    if quspin_use_sz_block and model_spec.model_family == "yao_lee":
        quspin_use_sz_block = False
        quspin_sz_reason = (
            "QuSpin full spin basis used because the yao_lee Eq. 7 Hamiltonian breaks total Sz."
        )
    if quspin_use_sz_block and bool(hamiltonian_field_ops.intersection({"Sx", "Sy"})):
        quspin_use_sz_block = False
        quspin_sz_reason = (
            "QuSpin full spin basis used because transverse Sx/Sy Zeeman terms break total Sz."
        )
    quspin_package_available = importlib.util.find_spec("quspin") is not None
    quspin_use_translation_block = False
    quspin_use_translation_x_block = False
    quspin_use_translation_y_block = False
    quspin_translation_x_reason = None
    quspin_translation_y_reason = None
    quspin_translation_reason = {"x": None, "y": None}
    quspin_translation_support_report = None
    requested_translation_block = bool(requested_translation_x_block or requested_translation_y_block)
    if quspin_ed_requested and requested_translation_block:
        if not quspin_package_available:
            reason = "QuSpin package is not installed, so translation blocks cannot be checked."
            quspin_translation_x_reason = reason if requested_translation_x_block else None
            quspin_translation_y_reason = reason if requested_translation_y_block else None
        else:
            try:
                import quspin_backend as quspin_validation_backend

                translation_support = quspin_validation_backend.quspin_translation_block_support(geometry)
                quspin_translation_support_report = translation_support
                x_support = translation_support.get("x", {})
                y_support = translation_support.get("y", {})
                quspin_use_translation_x_block = bool(
                    requested_translation_x_block and x_support.get("supported", False)
                )
                quspin_use_translation_y_block = bool(
                    requested_translation_y_block and y_support.get("supported", False)
                )
                quspin_translation_x_reason = x_support.get("reason") if requested_translation_x_block else None
                quspin_translation_y_reason = y_support.get("reason") if requested_translation_y_block else None
            except Exception as exc:
                reason = str(exc)
                quspin_translation_x_reason = reason if requested_translation_x_block else None
                quspin_translation_y_reason = reason if requested_translation_y_block else None
    quspin_use_translation_block = bool(quspin_use_translation_x_block or quspin_use_translation_y_block)
    if requested_tau_z_block and quspin_use_translation_block:
        if quspin_use_translation_x_block:
            quspin_translation_x_reason = (
                "Dropped for QuSpin native ED: translations must act on spin and orbital together as "
                "one fused physical-site operation; the current tensor-basis path keeps Tz instead."
            )
        if quspin_use_translation_y_block:
            quspin_translation_y_reason = (
                "Dropped for QuSpin native ED: translations must act on spin and orbital together as "
                "one fused physical-site operation; the current tensor-basis path keeps Tz instead."
            )
        quspin_use_translation_x_block = False
        quspin_use_translation_y_block = False
        quspin_use_translation_block = False
    quspin_translation_reason = {
        "x": quspin_translation_x_reason,
        "y": quspin_translation_y_reason,
    }
    quspin_use_reflection_block = False
    quspin_reflection_reason = None
    if quspin_ed_requested and ed_plan_use_c3_block:
        quspin_reflection_reason = (
            "ED symmetry plan accepts combined spin-lattice C3, but QuSpin-native does not use a pure "
            "site-permutation C3 map because the physical Yao-Lee C3 also requires a local spin rotation."
        )
    elif quspin_ed_requested and (bool(args.use_reflection_block) or int(args.reflection_block) != 0):
        quspin_reflection_reason = (
            "QuSpin reflection/C3 blocks are not applied for the bond-directional Yao-Lee Hamiltonian; "
            "they can permute x/y/z bond types unless a gauge map is implemented."
        )
    quspin_use_z2_block = False
    quspin_z2_reason = None
    quspin_z2_kind = None
    quspin_z2_generator = None
    quspin_zero_field_spin_flip_z2 = bool(
        model_spec.model_family == "yao_lee"
        and model_spec.orbital_rep == "1/2"
        and ed_actual_field_class == "none"
        and requested_z2_kind == "spin_flip"
        and not hamiltonian_field_ops
    )
    if quspin_ed_requested and quspin_requested_z2_block:
        if quspin_zero_field_spin_flip_z2:
            quspin_use_z2_block = True
            quspin_z2_kind = "spin_flip"
            quspin_z2_generator = "spin_flip"
            quspin_z2_reason = (
                "Using QuSpin spin_basis_1d spin-flip zblock on the full spin basis; "
                "this does not require total Sz conservation."
            )
        elif requested_z2_kind == "spin_pi_z":
            quspin_z2_reason = (
                "ED symmetry plan selected spin_pi_z parity, but the current QuSpin tensor-basis "
                "path does not implement the spin_pi_z projector yet; keeping the ED run in the Tz sector."
            )
        elif requested_z2_generator not in ("spin_flip", None):
            quspin_z2_reason = (
                f"QuSpin Z2 reduction is disabled because generator {requested_z2_generator!r} "
                "is not the supported spin_flip zblock generator for the current field setting."
            )
        else:
            quspin_z2_reason = (
                "QuSpin spin-flip Z2 is enabled only for zero-field Yao-Lee Hamiltonians in this path."
            )
    spin_orbital_block_dim = _spin_orbital_symmetry_reduced_dimension(
        int(geometry.number_of_sites),
        quspin_use_sz_block,
        requested_target_sz2,
        requested_tau_z_block,
        requested_target_tz2,
    )
    if quspin_use_z2_block:
        spin_orbital_block_dim = max(1, int(spin_orbital_block_dim) // 2)
    quspin_structurally_available = (
        quspin_ed_requested
        and quspin_package_available
        and model_spec.spin_rep == "1/2"
        and model_spec.orbital_rep == "1/2"
        and model_spec.model_family == "yao_lee"
        and model_spec.ising_axis == "z"
        and int(spin_orbital_block_dim) > 0
    )
    quspin_actual_hilbert_dim: int | None = None
    quspin_basis_build_reason = None
    if quspin_structurally_available:
        try:
            import quspin_backend as quspin_basis_backend

            quspin_preflight_basis = quspin_basis_backend.build_quspin_yao_lee_basis(
                int(geometry.number_of_sites),
                geometry=geometry,
                use_sz_block=quspin_use_sz_block,
                target_sz2=requested_target_sz2,
                use_tau_z_block=requested_tau_z_block,
                target_tz2=requested_target_tz2,
                use_z2_block=quspin_use_z2_block,
                z2_generator=quspin_z2_generator,
                z2_target_parity=int(args.z2_target_parity),
                use_translation_block=quspin_use_translation_block,
                use_translation_x_block=quspin_use_translation_x_block,
                use_translation_y_block=quspin_use_translation_y_block,
                momentum_block_1=int(args.momentum_x_block),
                momentum_block_2=int(args.momentum_y_block),
                momentum_x_block=int(args.momentum_x_block),
                momentum_y_block=int(args.momentum_y_block),
                use_reflection_block=False,
                reflection_block=0,
            )
            quspin_actual_hilbert_dim = int(quspin_preflight_basis.Ns)
        except Exception as exc:
            quspin_basis_build_reason = f"Failed to build the requested QuSpin reduced basis: {exc}"
    quspin_ed_available = bool(
        quspin_structurally_available
        and quspin_actual_hilbert_dim is not None
        and int(quspin_actual_hilbert_dim) > 0
    )
    quspin_ed_reason = None
    if quspin_ed_requested and not quspin_package_available:
        quspin_ed_reason = "The Python package 'quspin' is not installed in the active environment."
    elif quspin_ed_requested and model_spec.spin_rep != "1/2":
        quspin_ed_reason = "QuSpin ED currently supports spin_rep=1/2 only."
    elif quspin_ed_requested and model_spec.orbital_rep != "1/2":
        quspin_ed_reason = "QuSpin ED currently supports orbital_rep=1/2 only."
    elif quspin_ed_requested and model_spec.model_family != "yao_lee":
        quspin_ed_reason = "QuSpin ED currently supports model_family=yao_lee only."
    elif quspin_ed_requested and model_spec.ising_axis != "z":
        quspin_ed_reason = "QuSpin ED currently supports ising_axis=z only."
    elif quspin_ed_requested and int(spin_orbital_block_dim) <= 0:
        quspin_ed_reason = "The requested shared U1 target sector is unreachable for this site count."
    elif quspin_ed_requested and requested_tau_z_block and quspin_use_translation_block:
        quspin_ed_reason = "QuSpin kept tau_z and dropped unsupported translation combinations."
    elif quspin_ed_requested and quspin_basis_build_reason is not None:
        quspin_ed_reason = quspin_basis_build_reason
    sz_conserved_requested = bool(requested_sz_block)
    sz_conserved_available = (
        not quspin_ed_requested
        and
        sz_conserved_requested
        and model_spec.model_family != "yao_lee"
        and model_spec.spin_rep == "1/2"
        and model_spec.orbital_rep == "1/2"
        and _sector_dimension_for_spin_half(int(geometry.number_of_sites), requested_target_sz2) > 0
    )
    sz_conserved_reason = None
    if sz_conserved_requested and model_spec.model_family == "yao_lee":
        sz_conserved_reason = "yao_lee now uses the Eq. 7 Hamiltonian, whose Sx/Sy gamma terms do not conserve total Sz."
    elif sz_conserved_requested and model_spec.orbital_rep == "0":
        sz_conserved_reason = "orbital_rep=0 is spin-only; using the legacy full spin ED path."
    elif sz_conserved_requested and model_spec.spin_rep != "1/2":
        sz_conserved_reason = "Sz-conserved bitwise ED currently supports spin_rep=1/2 only."
    elif sz_conserved_requested and model_spec.orbital_rep != "1/2":
        sz_conserved_reason = "Sz-conserved bitwise ED currently supports orbital_rep=1/2 only."
    elif sz_conserved_requested and _sector_dimension_for_spin_half(int(geometry.number_of_sites), requested_target_sz2) <= 0:
        sz_conserved_reason = "The requested total Sz sector is unreachable for this number of spin-1/2 sites."
    standard_u1_requested = bool(
        (not quspin_ed_requested)
        and requested_tau_z_block
    )
    standard_u1_dim = (
        estimate_spin_orbital_u1_dimension(
            int(geometry.number_of_sites),
            use_sz_block=False,
            target_sz2=requested_target_sz2,
            use_tau_z_block=True,
            target_tz2=requested_target_tz2,
        )
        if standard_u1_requested and model_spec.spin_rep == "1/2" and model_spec.orbital_rep == "1/2"
        else 0
    )
    standard_u1_available = bool(
        standard_u1_requested
        and model_spec.spin_rep == "1/2"
        and model_spec.orbital_rep == "1/2"
        and int(standard_u1_dim) > 0
    )
    standard_u1_reason = None
    if standard_u1_requested and model_spec.spin_rep != "1/2":
        standard_u1_reason = "Standard Tz-reduced sparse ED currently supports spin_rep=1/2 only."
    elif standard_u1_requested and model_spec.orbital_rep != "1/2":
        standard_u1_reason = "Standard Tz-reduced sparse ED currently supports orbital_rep=1/2 only."
    elif standard_u1_requested and int(standard_u1_dim) <= 0:
        standard_u1_reason = "The requested total Tz sector is unreachable for this number of orbital-1/2 sites."
    standard_projector_z2_requested = bool(
        standard_u1_available
        and requested_z2_block
        and str(requested_z2_kind) == "spin_pi_z"
    )
    standard_projector_translation_x_requested = bool(standard_u1_available and requested_translation_x_block)
    standard_projector_translation_y_requested = bool(standard_u1_available and requested_translation_y_block)
    standard_projector_c3_requested = bool(standard_u1_available and ed_plan_use_c3_block)
    standard_projector_requested = bool(
        standard_projector_z2_requested
        or standard_projector_translation_x_requested
        or standard_projector_translation_y_requested
        or standard_projector_c3_requested
    )
    standard_projector_reduction_factor = 1
    if standard_projector_z2_requested:
        standard_projector_reduction_factor *= 2
    if standard_projector_translation_x_requested:
        standard_projector_reduction_factor *= max(1, int(getattr(geometry, "length_x", args.length_x) or 1))
    if standard_projector_translation_y_requested:
        standard_projector_reduction_factor *= max(1, int(getattr(geometry, "length_y", args.length_y) or 1))
    if standard_projector_c3_requested:
        standard_projector_reduction_factor *= 3
    standard_projector_dim_estimate = int(
        max(1, int(standard_u1_dim) // max(1, int(standard_projector_reduction_factor)))
    ) if standard_projector_requested else int(standard_u1_dim)
    if quspin_ed_requested:
        hilbert_dim = int(
            quspin_actual_hilbert_dim
            if quspin_actual_hilbert_dim is not None
            else spin_orbital_block_dim
        )
        ed_basis_type = (
            "quspin_tensor_spin_z2_orbital_tz"
            if quspin_use_z2_block and requested_tau_z_block
            else (
                "quspin_tensor_spin_z2_orbital_full"
                if quspin_use_z2_block
                else (
                    "quspin_tensor_"
                    f"spin_{'u1_block' if quspin_use_sz_block else 'full'}_"
                    f"orbital_{'u1_block' if requested_tau_z_block else 'full'}"
                )
            )
        )
    elif sz_conserved_available:
        hilbert_dim = int(estimate_sz_conserved_dimension(int(geometry.number_of_sites), target_sz2=requested_target_sz2))
        ed_basis_type = "bitwise_spin_orbital_total_sz_block"
    elif standard_u1_available:
        hilbert_dim = int(standard_projector_dim_estimate if standard_projector_requested else standard_u1_dim)
        ed_basis_type = (
            "bitwise_spin_orbital_tz_projector_block"
            if standard_projector_requested
            else "bitwise_spin_orbital_total_tz_block"
        )
    else:
        hilbert_dim = full_hilbert_dim
        ed_basis_type = "legacy_full_tensor_product"
    ed_applied_reductions: List[str] = []
    if quspin_ed_available:
        if quspin_use_sz_block:
            ed_applied_reductions.append("sz")
        if requested_tau_z_block:
            ed_applied_reductions.append("tz")
        if quspin_use_z2_block:
            ed_applied_reductions.append("z2")
        if quspin_use_translation_x_block:
            ed_applied_reductions.append("translation_x")
        if quspin_use_translation_y_block:
            ed_applied_reductions.append("translation_y")
    elif sz_conserved_available:
        ed_applied_reductions.append("sz")
    elif standard_u1_available:
        ed_applied_reductions.append("tz")
        if standard_projector_z2_requested:
            ed_applied_reductions.append("z2")
        if standard_projector_translation_x_requested:
            ed_applied_reductions.append("translation_x")
        if standard_projector_translation_y_requested:
            ed_applied_reductions.append("translation_y")
        if standard_projector_c3_requested:
            ed_applied_reductions.append("combined_c3")
    planned_ed_reduction_names = []
    for item in list(ed_symmetry_plan.get("accepted_symmetries", [])):
        item_text = str(item)
        if item_text.startswith("z2:"):
            planned_ed_reduction_names.append("z2")
        elif item_text == "combined_c3":
            planned_ed_reduction_names.append("combined_c3")
        else:
            planned_ed_reduction_names.append(item_text)
    ed_unsupported_reductions = [
        reduction
        for reduction in sorted(set(list(args.symmetry_reductions) + planned_ed_reduction_names))
        if reduction not in ed_applied_reductions and reduction not in ("none", "auto")
    ]
    ed_eligibility: Dict[str, Any] = {
        "requested": bool(args.run_ed),
        "ed_backend": ed_backend_name,
        "requested_backend": requested_ed_backend_name,
        "actual_backend": ed_backend_name,
        "symmetry_engine": effective_ed_engine,
        "requested_symmetry_engine": str(ed_symmetry_plan.get("requested_engine", args.ed_symmetry_engine)),
        "requested_symmetries": list(ed_symmetry_plan.get("requested_symmetries", [])),
        "accepted_symmetries": list(ed_symmetry_plan.get("accepted_symmetries", [])),
        "dropped_symmetries": list(ed_symmetry_plan.get("dropped_symmetries", [])),
        "symmetry_reasons": dict(ed_symmetry_plan.get("reasons", {})),
        "z2_generator_used": ed_symmetry_plan.get("z2_generator_used"),
        "z2_selection_reason": ed_symmetry_plan.get("z2_selection_reason"),
        "quspin_z2_selection_reason": ed_symmetry_plan.get("quspin_z2_selection_reason"),
        "requested_ed_backend": requested_ed_backend_name,
        "effective_ed_backend": ed_backend_name,
        "ed_backend_override_reason": ed_backend_override_reason,
        "backend_override_reason": ed_backend_override_reason,
        "allowed": False,
        "forbidden": True,
        "number_of_sites": int(geometry.number_of_sites),
        "local_dimension": int(local_dim),
        "hilbert_dimension": int(hilbert_dim),
        "effective_hilbert_dimension": int(hilbert_dim),
        "standard_u1_parent_hilbert_dimension": int(standard_u1_dim) if standard_u1_available else None,
        "standard_projector_requested": bool(standard_projector_requested),
        "standard_projector_reduction_factor_estimate": int(standard_projector_reduction_factor),
        "standard_projector_hilbert_dimension_estimate": (
            int(standard_projector_dim_estimate) if standard_projector_requested else None
        ),
        "pre_quspin_hilbert_dimension_estimate": int(spin_orbital_block_dim) if quspin_ed_requested else None,
        "actual_quspin_hilbert_dimension": (
            int(quspin_actual_hilbert_dim) if quspin_actual_hilbert_dim is not None else None
        ),
        "full_hilbert_dimension": int(full_hilbert_dim),
        "basis_type": ed_basis_type,
        "symmetry_reductions": symmetry_reduction_settings,
        "ed_symmetry_plan": ed_symmetry_plan,
        "applied_reductions": ed_applied_reductions,
        "unsupported_or_unapplied_reductions": ed_unsupported_reductions,
        "use_sz_conserved_requested": bool(sz_conserved_requested),
        "use_sz_conserved": "sz" in ed_applied_reductions,
        "use_tau_z_conserved": "tz" in ed_applied_reductions,
        "use_z2_conserved": "z2" in ed_applied_reductions,
        "standard_sz_conserved": bool(sz_conserved_available),
        "standard_u1_conserved": bool(standard_u1_available),
        "standard_u1_reason": standard_u1_reason,
        "quspin_available": bool(quspin_ed_available),
        "quspin_package_available": bool(quspin_package_available),
        "quspin_reason": quspin_ed_reason,
        "quspin_requested_sz_block": bool(requested_sz_block),
        "quspin_use_sz_block": bool(quspin_use_sz_block),
        "quspin_sz_reason": quspin_sz_reason,
        "quspin_requested_translation_block": bool(requested_translation_block),
        "quspin_requested_translation_x_block": bool(requested_translation_x_block),
        "quspin_requested_translation_y_block": bool(requested_translation_y_block),
        "quspin_use_translation_block": bool(quspin_use_translation_block),
        "quspin_use_translation_x_block": bool(quspin_use_translation_x_block),
        "quspin_use_translation_y_block": bool(quspin_use_translation_y_block),
        "quspin_momentum_x_block": int(args.momentum_x_block),
        "quspin_momentum_y_block": int(args.momentum_y_block),
        "quspin_translation_reason": quspin_translation_reason,
        "quspin_translation_x_reason": quspin_translation_x_reason,
        "quspin_translation_y_reason": quspin_translation_y_reason,
        "quspin_translation_support": quspin_translation_support_report,
        "quspin_requested_reflection_block": bool(ed_plan_use_c3_block or args.use_reflection_block),
        "quspin_use_reflection_block": bool(quspin_use_reflection_block),
        "quspin_reflection_reason": quspin_reflection_reason,
        "quspin_requested_z2_block": bool(quspin_requested_z2_block),
        "quspin_requested_z2_kind": requested_z2_kind,
        "quspin_use_z2_block": bool(quspin_use_z2_block),
        "quspin_z2_kind": quspin_z2_kind,
        "quspin_z2_generator": quspin_z2_generator,
        "quspin_z2_reason": quspin_z2_reason,
        "sz_conserved_reason": sz_conserved_reason,
        "max_sites": int(args.max_ed_sites),
        "max_hilbert_dimension": int(args.max_ed_hilbert_dim),
        "solver_requested": str(args.ed_solver),
        "solver": (
            "quspin_eigsh"
            if quspin_ed_requested
            else (
                ("spin_orbital_tz_projector_sparse" if standard_projector_requested else "spin_orbital_u1_sparse")
                if standard_u1_available
                else ("sz_conserved_sparse" if sz_conserved_available else str(args.ed_solver))
            )
        ),
        "max_eigenstates": (
            int(args.ed_max_eigenstates)
            if quspin_ed_requested
            else (
                int(SZ_CONSERVED_ED_EIGENSTATES)
                if (sz_conserved_available or standard_u1_available)
                else int(args.ed_max_eigenstates)
            )
        ),
    }
    if not bool(args.run_ed):
        ed_eligibility["reason"] = "ED is disabled by RUN_ED/--run-ed."
    elif quspin_ed_requested and not quspin_ed_available:
        ed_eligibility["reason"] = str(quspin_ed_reason or "QuSpin ED is not available for this model.")
    elif (
        not quspin_ed_requested
        and sz_conserved_requested
        and not sz_conserved_available
        and not standard_u1_available
        and model_spec.orbital_rep != "0"
    ):
        ed_eligibility["reason"] = str(sz_conserved_reason or "Sz-conserved ED is not available for this model.")
    elif geometry.number_of_sites > int(args.max_ed_sites):
        ed_eligibility["reason"] = f"ED is limited to {int(args.max_ed_sites)} sites or fewer."
    elif hilbert_dim > int(args.max_ed_hilbert_dim):
        ed_eligibility["reason"] = (
            f"ED {ed_basis_type} Hilbert-space dimension {hilbert_dim} exceeds limit {int(args.max_ed_hilbert_dim)} "
            f"(local_dim={local_dim}, sites={geometry.number_of_sites})."
        )
    else:
        ed_eligibility["allowed"] = True
        ed_eligibility["forbidden"] = False
        ed_eligibility["reason"] = "ED is allowed for the current caps."
    if show_progress:
        print(
            "[ED] eligibility: "
            f"requested={bool(args.run_ed)}, "
            f"allowed={bool(ed_eligibility['allowed'])}, "
            f"forbidden={bool(ed_eligibility['forbidden'])}, "
            f"backend={ed_backend_name}, "
            f"requested_backend={requested_ed_backend_name}, "
            f"N={int(geometry.number_of_sites)}, "
            f"local_dim={local_dim}, "
            f"basis={ed_basis_type}, "
            f"effective_hilbert_dim={hilbert_dim}, "
            f"full_hilbert_dim={full_hilbert_dim}, "
            f"max_sites={int(args.max_ed_sites)}, "
            f"max_hilbert_dim={int(args.max_ed_hilbert_dim)}, "
            f"solver={ed_eligibility['solver']}, "
            f"max_eigenstates={int(ed_eligibility['max_eigenstates'])} :: "
            f"{ed_eligibility['reason']}"
        )

    if not args.run_ed:
        summary["ed"] = {
            "status": "not_requested",
            "reason": str(ed_eligibility["reason"]),
            "eligibility": ed_eligibility,
        }
        summary["stages"]["ed"] = "not_requested"
        _save_summary_checkpoint(args.output_folder, summary)
    else:
        summary["stages"]["ed"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        if not bool(ed_eligibility["allowed"]) and geometry.number_of_sites > int(args.max_ed_sites):
            summary["ed"] = {
                "status": "skipped",
                "reason": str(ed_eligibility["reason"]),
                "eligibility": ed_eligibility,
            }
            summary["stages"]["ed"] = "skipped"
        elif not bool(ed_eligibility["allowed"]):
            summary["ed"] = {
                "status": "skipped",
                "reason": str(ed_eligibility["reason"]),
                "eligibility": ed_eligibility,
            }
            summary["stages"]["ed"] = "skipped"
        else:
            try:
                ed_state_for_entropy = None
                ed_correlations: Dict[str, np.ndarray] = {}
                ed_scalar_correlations: Dict[str, np.ndarray] = {}
                ed_real_space_patterns: Dict[str, Any] = {}
                ed_bond_rows: List[Dict[str, Any]] = []
                ed_structure_factor_rows: List[Dict[str, Any]] = []
                ed_plaquette_flux: Dict[str, Any] | None = None
                if quspin_ed_requested:
                    import quspin_backend as quspin_ed_backend

                    ed_spectrum, ed_vectors = quspin_ed_backend.run_small_cluster_exact_spectrum(
                        geometry=geometry,
                        model_spec=model_spec,
                        alpha=args.alpha,
                        beta=args.beta,
                        coupling_j=args.coupling_j,
                        eigenstate_count=max(1, int(args.ed_max_eigenstates)),
                        check_ground_state_degeneracy=bool(args.check_ground_state_degeneracy),
                        jx=args.jx,
                        jy=args.jy,
                        jz=args.jz,
                        external_field_terms=hamiltonian_external_field_terms,
                        show_progress=show_progress,
                        ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                        ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                        solver=args.ed_solver,
                        sparse_tol=float(args.ed_sparse_tol),
                        sparse_maxiter=(
                            int(args.ed_sparse_maxiter)
                            if int(args.ed_sparse_maxiter) > 0
                            else None
                        ),
                        use_sz_block=quspin_use_sz_block,
                        target_sz2=requested_target_sz2,
                        use_tau_z_block=requested_tau_z_block,
                        target_tz2=requested_target_tz2,
                        use_z2_block=quspin_use_z2_block,
                        z2_generator=quspin_z2_generator,
                        z2_target_parity=int(args.z2_target_parity),
                        use_translation_block=quspin_use_translation_block,
                        use_translation_x_block=quspin_use_translation_x_block,
                        use_translation_y_block=quspin_use_translation_y_block,
                        momentum_block_1=int(args.momentum_x_block),
                        momentum_block_2=int(args.momentum_y_block),
                        momentum_x_block=int(args.momentum_x_block),
                        momentum_y_block=int(args.momentum_y_block),
                        use_reflection_block=quspin_use_reflection_block,
                        reflection_block=0,
                        check_symm=bool(args.quspin_check_symmetries),
                        check_herm=bool(args.quspin_check_hermiticity),
                        check_pcon=bool(args.quspin_check_particle_conservation),
                    )
                    quspin_basis_use_sz_block = bool(ed_spectrum.get("use_sz_block", quspin_use_sz_block))
                    quspin_basis_target_sz2 = int(ed_spectrum.get("target_sz2", requested_target_sz2))
                    quspin_basis_use_z2_block = bool(ed_spectrum.get("use_z2_block", quspin_use_z2_block))
                    quspin_basis = quspin_ed_backend.build_quspin_yao_lee_basis(
                        int(geometry.number_of_sites),
                        geometry=geometry,
                        use_sz_block=quspin_basis_use_sz_block,
                        target_sz2=quspin_basis_target_sz2,
                        use_tau_z_block=requested_tau_z_block,
                        target_tz2=requested_target_tz2,
                        use_z2_block=quspin_basis_use_z2_block,
                        z2_generator=quspin_z2_generator,
                        z2_target_parity=int(args.z2_target_parity),
                        use_translation_block=quspin_use_translation_block,
                        use_translation_x_block=quspin_use_translation_x_block,
                        use_translation_y_block=quspin_use_translation_y_block,
                        momentum_block_1=int(args.momentum_x_block),
                        momentum_block_2=int(args.momentum_y_block),
                        momentum_x_block=int(args.momentum_x_block),
                        momentum_y_block=int(args.momentum_y_block),
                        use_reflection_block=quspin_use_reflection_block,
                        reflection_block=0,
                    )
                    ed_energy = float(ed_spectrum["ground_state_energy"])
                    ed_state = ed_vectors[:, 0]
                    ed_plaquette_flux = ed_spectrum.get("plaquette_flux") if isinstance(ed_spectrum, dict) else None
                    if ed_plaquette_flux is None:
                        try:
                            ed_plaquette_flux = quspin_ed_backend.compute_plaquette_flux(
                                quspin_basis,
                                ed_state,
                                geometry,
                                plaquette_center_idx=None,
                            )
                        except Exception as exc:
                            ed_plaquette_flux = {"available": False, "warning": str(exc)}
                    if calculate_correlations:
                        ed_scalar_correlations = quspin_ed_backend.build_spin_orbital_scalar_correlations(
                            quspin_basis,
                            ed_state,
                            int(geometry.number_of_sites),
                        )
                        if calculate_bond_energies:
                            ed_bond_rows = quspin_ed_backend.all_bond_energies(
                                geometry,
                                ed_scalar_correlations,
                                args.alpha,
                                args.beta,
                                args.coupling_j,
                            )
                        if calculate_structure_factors:
                            ed_structure_factor_rows = quspin_ed_backend.all_high_symmetry_structure_factors(
                                ed_scalar_correlations,
                                geometry,
                            )
                elif standard_u1_available:
                    if standard_projector_requested:
                        ed_spectrum, ed_vectors, ed_basis_list, ed_basis_map = run_spin_orbital_projected_exact_spectrum(
                            geometry=geometry,
                            model_spec=model_spec,
                            alpha=args.alpha,
                            beta=args.beta,
                            coupling_j=args.coupling_j,
                            eigenstate_count=int(SZ_CONSERVED_ED_EIGENSTATES),
                            check_ground_state_degeneracy=bool(args.check_ground_state_degeneracy),
                            jx=args.jx,
                            jy=args.jy,
                            jz=args.jz,
                            external_field_terms=hamiltonian_external_field_terms,
                            show_progress=show_progress,
                            ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                            ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                            sparse_tol=float(args.ed_sparse_tol),
                            sparse_maxiter=(
                                int(args.ed_sparse_maxiter)
                                if int(args.ed_sparse_maxiter) > 0
                                else None
                            ),
                            target_tz2=requested_target_tz2,
                            use_spin_pi_z=standard_projector_z2_requested,
                            z2_target_parity=int(ed_symmetry_plan.get("z2_target_parity", args.z2_target_parity)),
                            use_translation_x=standard_projector_translation_x_requested,
                            use_translation_y=standard_projector_translation_y_requested,
                            momentum_x=int(ed_symmetry_plan.get("momentum_x_block", args.momentum_x_block)),
                            momentum_y=int(ed_symmetry_plan.get("momentum_y_block", args.momentum_y_block)),
                            use_combined_c3=standard_projector_c3_requested,
                            c3_q_blocks=str(ed_symmetry_plan.get("c3_q_blocks", args.ed_c3_q_blocks)),
                        )
                    else:
                        ed_spectrum, ed_vectors, ed_basis_list, ed_basis_map = run_spin_orbital_u1_exact_spectrum(
                            geometry=geometry,
                            model_spec=model_spec,
                            alpha=args.alpha,
                            beta=args.beta,
                            coupling_j=args.coupling_j,
                            eigenstate_count=int(SZ_CONSERVED_ED_EIGENSTATES),
                            check_ground_state_degeneracy=bool(args.check_ground_state_degeneracy),
                            jx=args.jx,
                            jy=args.jy,
                            jz=args.jz,
                            external_field_terms=hamiltonian_external_field_terms,
                            show_progress=show_progress,
                            ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                            ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                            sparse_tol=float(args.ed_sparse_tol),
                            sparse_maxiter=(
                                int(args.ed_sparse_maxiter)
                                if int(args.ed_sparse_maxiter) > 0
                                else None
                            ),
                            use_sz_block=False,
                            target_sz2=requested_target_sz2,
                            use_tau_z_block=True,
                            target_tz2=requested_target_tz2,
                        )
                    ed_energy = float(ed_spectrum["ground_state_energy"])
                    ed_state = ed_vectors[:, 0]
                    try:
                        ed_plaquette_flux = plaquette_flux_from_spin_orbital_u1_ed_state(
                            geometry,
                            ed_state,
                            ed_basis_list,
                            ed_basis_map,
                            plaquette_center_idx=None,
                        )
                    except Exception as exc:
                        ed_plaquette_flux = {"available": False, "warning": str(exc)}
                    if calculate_correlations:
                        ed_correlations = collect_correlation_matrices_from_spin_orbital_u1_ed(
                            geometry,
                            ed_state,
                            ed_basis_list,
                            ed_basis_map,
                            show_progress=show_progress,
                        )
                        ed_scalar_correlations = build_spin_orbital_u1_scalar_correlations(ed_correlations)
                        if calculate_bond_energies:
                            ed_bond_rows = all_bond_energies_sz_conserved(
                                geometry,
                                ed_correlations,
                                args.alpha,
                                args.beta,
                                args.coupling_j,
                                show_progress=show_progress,
                                progress_desc="Tz-ED bond energies",
                            )
                        if calculate_structure_factors:
                            ed_structure_factor_rows = all_high_symmetry_structure_factors(
                                ed_scalar_correlations,
                                geometry,
                                lattice=lattice_name,
                                show_progress=show_progress,
                                progress_desc="Tz-ED structure factors",
                            )
                elif sz_conserved_available:
                    ed_spectrum, ed_vectors, ed_basis_list, ed_basis_map = run_sz_conserved_exact_spectrum(
                        geometry=geometry,
                        alpha=args.alpha,
                        beta=args.beta,
                        coupling_j=args.coupling_j,
                        eigenstate_count=int(SZ_CONSERVED_ED_EIGENSTATES),
                        check_ground_state_degeneracy=bool(args.check_ground_state_degeneracy),
                        external_field_terms=hamiltonian_external_field_terms,
                        show_progress=show_progress,
                        ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                        ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                        sparse_tol=float(args.ed_sparse_tol),
                        sparse_maxiter=(
                            int(args.ed_sparse_maxiter)
                            if int(args.ed_sparse_maxiter) > 0
                            else None
                        ),
                        target_sz2=requested_target_sz2,
                    )
                    ed_energy = float(ed_spectrum["ground_state_energy"])
                    ed_state = ed_vectors[:, 0]
                    try:
                        ed_plaquette_flux = plaquette_flux_from_sz_conserved_ed_state(
                            geometry,
                            ed_state,
                            ed_basis_list,
                            ed_basis_map,
                            plaquette_center_idx=None,
                        )
                    except Exception as exc:
                        ed_plaquette_flux = {"available": False, "warning": str(exc)}
                    if calculate_correlations:
                        ed_correlations = collect_correlation_matrices_from_sz_conserved_ed(
                            geometry,
                            ed_state,
                            ed_basis_list,
                            ed_basis_map,
                            show_progress=show_progress,
                        )
                        ed_scalar_correlations = build_sz_conserved_scalar_correlations(ed_correlations)
                        if calculate_bond_energies:
                            ed_bond_rows = all_bond_energies_sz_conserved(
                                geometry,
                                ed_correlations,
                                args.alpha,
                                args.beta,
                                args.coupling_j,
                                show_progress=show_progress,
                                progress_desc="Sz-ED bond energies",
                            )
                        if calculate_structure_factors:
                            ed_structure_factor_rows = all_high_symmetry_structure_factors(
                                ed_scalar_correlations,
                                geometry,
                                lattice=lattice_name,
                                show_progress=show_progress,
                                progress_desc="Sz-ED structure factors",
                            )
                else:
                    ed_spectrum, ed_vectors = run_small_cluster_exact_spectrum(
                        geometry=geometry,
                        model_spec=model_spec,
                        alpha=args.alpha,
                        beta=args.beta,
                        coupling_j=args.coupling_j,
                        eigenstate_count=max(1, int(args.ed_max_eigenstates)),
                        check_ground_state_degeneracy=bool(args.check_ground_state_degeneracy),
                        jx=args.jx,
                        jy=args.jy,
                        jz=args.jz,
                        external_field_terms=hamiltonian_external_field_terms,
                        show_progress=show_progress,
                        ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                        ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                        solver=args.ed_solver,
                        sparse_tol=float(args.ed_sparse_tol),
                        sparse_maxiter=(
                            int(args.ed_sparse_maxiter)
                            if int(args.ed_sparse_maxiter) > 0
                            else None
                        ),
                    )
                    ed_energy = float(ed_spectrum["ground_state_energy"])
                    ed_state = ed_vectors[:, 0]
                    ed_state_for_entropy = ed_state
                    try:
                        ed_plaquette_flux = plaquette_flux_from_ed_state(
                            geometry,
                            ed_state,
                            model_spec,
                            plaquette_center_idx=None,
                        )
                    except Exception as exc:
                        ed_plaquette_flux = {"available": False, "warning": str(exc)}
                    if calculate_correlations:
                        ed_correlations = collect_correlation_matrices_from_ed(
                            geometry,
                            ed_state,
                            model_spec=model_spec,
                            show_progress=show_progress,
                        )
                        ed_scalar_correlations = build_spin_orbital_scalar_correlations(ed_correlations)
                        if calculate_bond_energies:
                            ed_bond_rows = all_bond_energies(
                                geometry,
                                ed_correlations,
                                model_spec,
                                args.alpha,
                                args.beta,
                                args.coupling_j,
                                show_progress=show_progress,
                                progress_desc="ED bond energies",
                            )
                        if calculate_structure_factors:
                            ed_structure_factor_rows = all_high_symmetry_structure_factors(
                                ed_scalar_correlations,
                                geometry,
                                lattice=lattice_name,
                                show_progress=show_progress,
                                progress_desc="ED structure factors",
                            )

                if calculate_real_space_patterns and len(ed_scalar_correlations) > 0:
                    try:
                        ed_real_space_patterns = build_reference_site_correlation_patterns(
                            geometry,
                            ed_scalar_correlations,
                            reference_site_idx=args.reference_site_idx,
                        )
                    except Exception as exc:
                        ed_real_space_patterns = {
                            "status": "failed",
                            "warning": f"Failed to extract ED reference-site patterns: {exc}",
                        }
                elif calculate_real_space_patterns:
                    ed_real_space_patterns = {
                        "status": "skipped",
                        "reason": "No ED scalar correlations were computed.",
                    }

                ed_entropy_warning: str | None = None
                ed_entropy_profile: Dict[str, Any] | None = None
                try:
                    if calculate_entanglement:
                        if ed_state_for_entropy is None:
                            raise ValueError(
                                "Reduced Sz-sector state remains in the bitwise basis; "
                                "legacy full-vector entanglement post-processing is skipped "
                                "to avoid allocating a dense 4**N object."
                            )
                        ed_entropy_profile = compute_ed_entropy_profile_from_state(
                            state=ed_state_for_entropy,
                            n_sites=geometry.number_of_sites,
                            local_dim=local_dim,
                            orders=ENTROPY_ORDERS,
                            show_progress=show_progress,
                        )
                        entropy_profiles["ED"] = ed_entropy_profile
                    else:
                        ed_entropy_warning = "ED entanglement entropy calculation skipped by calculate_entanglement=false."
                except Exception as exc:
                    ed_entropy_warning = f"Failed to compute ED entanglement profile: {exc}"

                actual_ed_applied_reductions = _ed_actual_applied_reductions_from_spectrum(
                    ed_applied_reductions,
                    ed_spectrum,
                )
                ed_runtime_symmetry_status = (
                    _ed_symmetry_status_text(ed_symmetry_plan, spectrum=ed_spectrum)
                    if isinstance(ed_spectrum, dict)
                    else _ed_symmetry_status_text(ed_symmetry_plan)
                )
                if show_progress and isinstance(ed_spectrum, dict):
                    final_dimension = ed_spectrum.get(
                        "projector_reduced_dimension",
                        ed_spectrum.get("hilbert_dimension", ed_spectrum.get("hilbert_dim")),
                    )
                    final_dimension_text = (
                        f", final_dim={int(final_dimension):,}" if final_dimension is not None else ""
                    )
                    print(
                        "[ed-symmetry] applied: "
                        f"backend={effective_ed_engine}, "
                        f"status={ed_runtime_symmetry_status}"
                        f"{final_dimension_text}"
                    )

                ed_all_plaquette_fluxes = _all_plaquette_fluxes_from_payload(ed_plaquette_flux or {})
                summary["ed"] = {
                    "status": "completed",
                    "ed_backend": ed_backend_name,
                    "requested_backend": requested_ed_backend_name,
                    "actual_backend": ed_backend_name,
                    "symmetry_engine": effective_ed_engine,
                    "requested_symmetries": list(ed_symmetry_plan.get("requested_symmetries", [])),
                    "accepted_symmetries": list(ed_symmetry_plan.get("accepted_symmetries", [])),
                    "dropped_symmetries": list(ed_symmetry_plan.get("dropped_symmetries", [])),
                    "z2_generator_used": ed_symmetry_plan.get("z2_generator_used"),
                    "z2_selection_reason": ed_symmetry_plan.get("z2_selection_reason"),
                    "quspin_z2_selection_reason": ed_symmetry_plan.get("quspin_z2_selection_reason"),
                    "requested_ed_backend": requested_ed_backend_name,
                    "ed_backend_override_reason": ed_backend_override_reason,
                    "backend_override_reason": ed_backend_override_reason,
                    "eligibility": ed_eligibility,
                    "basis_type": ed_basis_type,
                    "symmetry_reductions": symmetry_reduction_settings,
                    "ed_symmetry_plan": ed_symmetry_plan,
                    "ed_symmetry_status": ed_runtime_symmetry_status,
                    "applied_reductions": actual_ed_applied_reductions,
                    "unsupported_or_unapplied_reductions": ed_unsupported_reductions,
                    "use_sz_conserved": "sz" in actual_ed_applied_reductions,
                    "use_tau_z_conserved": "tz" in actual_ed_applied_reductions,
                    "use_z2_conserved": "z2" in actual_ed_applied_reductions,
                    "z2_kind": (
                        ed_spectrum.get("z2_kind", quspin_z2_kind)
                        if "z2" in actual_ed_applied_reductions and isinstance(ed_spectrum, dict)
                        else (quspin_z2_kind if "z2" in actual_ed_applied_reductions else None)
                    ),
                    "use_translation_x_conserved": "translation_x" in actual_ed_applied_reductions,
                    "use_translation_y_conserved": "translation_y" in actual_ed_applied_reductions,
                    "use_c3_conserved": "combined_c3" in actual_ed_applied_reductions,
                    "hamiltonian_formula": (
                        ed_spectrum.get("formula")
                        if isinstance(ed_spectrum, dict) and ed_spectrum.get("formula") is not None
                        else "legacy full tensor-product ED formula from models.model_terms_for_bond"
                    ),
                    "selected_sector_energy": ed_energy if "tz" in actual_ed_applied_reductions else None,
                    "selected_target_tz2": requested_target_tz2 if "tz" in actual_ed_applied_reductions else None,
                    "min_over_tz_sectors": {
                        "computed": False,
                        "reason": (
                            "Full Tz-sector scan is reserved for small-cluster validation; "
                            "production ED reports the selected target sector separately."
                        ),
                    } if "tz" in actual_ed_applied_reductions else None,
                    "ground_state_energy": ed_energy,
                    "energy_per_site": ed_energy / geometry.number_of_sites,
                    "absolute_energy_difference_dmrg_minus_ed": abs(dmrg_energy - ed_energy),
                    "bond_energies": ed_bond_rows,
                    "structure_factors": ed_structure_factor_rows,
                    "real_space_patterns": ed_real_space_patterns,
                    "plaquette_flux": ed_plaquette_flux,
                    "all_plaquette_fluxes": ed_all_plaquette_fluxes,
                }
                if ed_all_plaquette_fluxes:
                    _record_all_plaquette_fluxes(summary, "ed", ed_plaquette_flux or {})
                if ed_spectrum is not None:
                    summary["ed"]["spectrum"] = ed_spectrum
                    for projector_key in ("projector_strategy", "memory_estimate_MB", "drop_reasons"):
                        if ed_spectrum.get(projector_key) is not None:
                            summary["ed"][projector_key] = ed_spectrum.get(projector_key)
                    if ed_spectrum.get("dropped_symmetries") is not None:
                        runtime_drops = list(ed_spectrum.get("dropped_symmetries", []))
                        planned_drops = list(ed_symmetry_plan.get("dropped_symmetries", []))
                        summary["ed"]["dropped_symmetries"] = planned_drops + runtime_drops
                    dimension_value = ed_spectrum.get("hilbert_dimension", ed_spectrum.get("hilbert_dim"))
                    if dimension_value is not None:
                        actual_dim = int(dimension_value)
                        summary["ed"]["hilbert_dimension"] = actual_dim
                        summary["ed"]["effective_hilbert_dimension"] = actual_dim
                        if isinstance(summary["ed"].get("eligibility"), dict):
                            summary["ed"]["eligibility"]["actual_hilbert_dimension"] = actual_dim
                            summary["ed"]["eligibility"]["effective_hilbert_dimension_after_quspin"] = actual_dim
                            if ed_spectrum.get("u1_basis_dimension") is not None:
                                summary["ed"]["eligibility"]["standard_u1_parent_hilbert_dimension"] = int(
                                    ed_spectrum["u1_basis_dimension"]
                                )
                    summary["ed"]["ground_state_degeneracy_check_enabled"] = bool(
                        ed_spectrum.get("ground_state_degeneracy_check_enabled", False)
                    )
                    summary["ed"]["ground_state_degeneracy_status"] = str(
                        ed_spectrum.get("ground_state_degeneracy_status", "not_checked")
                    )
                    if ed_spectrum.get("ground_state_degeneracy") is not None:
                        summary["ed"]["ground_state_degeneracy"] = int(ed_spectrum["ground_state_degeneracy"])
                        summary["ed"]["ground_state_degeneracy_tolerance"] = float(
                            ed_spectrum["ground_state_degeneracy_tolerance"]
                        )
                        summary["ed"]["ground_state_degeneracy_absolute_tolerance"] = float(
                            ed_spectrum.get(
                                "ground_state_degeneracy_absolute_tolerance",
                                args.ed_ground_manifold_abs_tol,
                            )
                        )
                        summary["ed"]["ground_state_degeneracy_relative_tolerance"] = float(
                            ed_spectrum.get(
                                "ground_state_degeneracy_relative_tolerance",
                                args.ed_ground_manifold_rel_tol,
                            )
                        )
                        summary["ed"]["ground_state_degeneracy_is_lower_bound"] = bool(
                            ed_spectrum.get("ground_state_degeneracy_is_lower_bound", False)
                        )
                    if ed_spectrum.get("first_excited_energy") is not None:
                        summary["ed"]["first_excited_energy"] = float(ed_spectrum["first_excited_energy"])
                        summary["ed"]["first_excited_energy_per_site"] = (
                            float(ed_spectrum["first_excited_energy"]) / float(geometry.number_of_sites)
                        )
                    if ed_spectrum.get("spectral_gap") is not None:
                        summary["ed"]["spectral_gap"] = float(ed_spectrum["spectral_gap"])
                if ed_entropy_profile is not None:
                    summary["ed"]["entanglement"] = ed_entropy_profile
                if ed_entropy_warning is not None:
                    summary["ed"]["entanglement_warning"] = ed_entropy_warning
                _save_summary_checkpoint(args.output_folder, summary)

                ed_pattern_correlations = (
                    ed_real_space_patterns.get("correlations")
                    if isinstance(ed_real_space_patterns, dict)
                    else None
                )
                ed_reference_site_idx = (
                    ed_real_space_patterns.get("reference_site_idx")
                    if isinstance(ed_real_space_patterns, dict)
                    else None
                )
                has_ed_spin_pattern_overlay = (
                    isinstance(ed_pattern_correlations, dict)
                    and "S" in ed_pattern_correlations
                    and ed_reference_site_idx is not None
                )

                def save_ed_bond_energy_plot(path: str) -> None:
                    if has_ed_spin_pattern_overlay:
                        save_phase_representative_pattern(
                            geometry,
                            ed_pattern_correlations["S"],
                            int(ed_reference_site_idx),
                            ed_bond_rows,
                            path,
                            plot_title("ED Spin Pattern + Resolved Bond Energy"),
                            external_field_vector=resolved_field_vector,
                        )
                        print(
                            "[plot] spin-vector overlay: "
                            f"{path} uses arrows above bonds "
                            "(spin zorder=50, bond zorder=3, white halo enabled)."
                        )
                        return
                    save_bond_energy_diagram(
                        geometry,
                        ed_bond_rows,
                        path,
                        plot_title("ED Bond-Energy Diagram"),
                    )

                should_save_ed_bond_plot = bool(args.plot_bond_energies or args.plot_real_space_patterns) and (
                    len(ed_bond_rows) > 0 or has_ed_spin_pattern_overlay
                )
                if should_save_ed_bond_plot:
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "ed_bond_energy_diagram_png",
                        output_filename("ed_bond_energy_diagram.png"),
                        save_ed_bond_energy_plot,
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                else:
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "ed_bond_energy_diagram_png",
                        output_filename("ed_bond_energy_diagram.png"),
                        "plot_bond_energies/plot_real_space_patterns is false, or no ED bond rows/spin pattern were computed",
                    )
                if bool(args.plot_structure_factors) and calculate_structure_factors and len(ed_structure_factor_rows) > 0:
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "ed_structure_factors_png",
                        output_filename("ed_structure_factors.png"),
                        lambda path: save_structure_factor_plot(ed_structure_factor_rows, path, plot_title("ED Structure Factors")),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                else:
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "ed_structure_factors_png",
                        output_filename("ed_structure_factors.png"),
                        "plot_structure_factors or calculate_structure_factors is false, or no ED structure rows were computed",
                    )
                if bool(args.plot_correlation_heatmaps) and calculate_correlations and len(ed_scalar_correlations) > 0:
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "ed_scalar_correlation_heatmaps_png",
                        output_filename("ed_scalar_correlation_heatmaps.png"),
                        lambda path: save_scalar_correlation_heatmaps(ed_scalar_correlations, path, f"ED | {run_plot_title_label}"),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                else:
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "ed_scalar_correlation_heatmaps_png",
                        output_filename("ed_scalar_correlation_heatmaps.png"),
                        "plot_correlation_heatmaps or calculate_correlations is false, or no ED scalar correlations were computed",
                    )
                ed_overlay_status = _plot_step_status(summary, "ed_bond_energy_diagram_png")
                ed_overlay_reason = (
                    "spin-vector pattern is drawn in ed_bond_energy_diagram.png above resolved bonds "
                    "(spin zorder=50, bond zorder=3, white halo enabled)"
                )
                if has_ed_spin_pattern_overlay and ed_overlay_status in ("saved", "skipped_exists"):
                    overlay_diagnostics = summary.setdefault("plot_overlay_diagnostics", {})
                    if isinstance(overlay_diagnostics, dict):
                        overlay_diagnostics["ed"] = {
                            "combined_plot": output_filename("ed_bond_energy_diagram.png"),
                            "spin_vectors_on_top": True,
                            "spin_vector_zorder": 50,
                            "resolved_bond_zorder": 3,
                            "white_halo": True,
                        }
                    _record_combined_plot_alias(
                        summary,
                        args.output_folder,
                        "ed_spin_real_space_pattern_png",
                        output_filename("ed_bond_energy_diagram.png"),
                        ed_overlay_reason,
                    )
                save_flux_crystal_output(
                    summary,
                    geometry,
                    summary.get("ed", {}),
                    "ed_flux_crystal_pattern_png",
                    "ed_flux_crystal_pattern.png",
                    "ED Plaquette Flux Crystal Pattern",
                )
                method_structure_comparison = {
                    finite_method_label: dmrg_structure_factor_rows,
                    "ED": ed_structure_factor_rows,
                }
                idmrg_structure_rows = (
                    summary["idmrg"].get("structure_factors")
                    if isinstance(summary.get("idmrg"), dict)
                    else None
                )
                if isinstance(idmrg_structure_rows, list) and len(idmrg_structure_rows) > 0:
                    method_structure_comparison["iDMRG-x"] = idmrg_structure_rows
                if bool(args.plot_structure_factors) and any(len(rows) > 0 for rows in method_structure_comparison.values()):
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "dmrg_vs_ed_vs_idmrg_structure_factors_png",
                        output_filename("dmrg_vs_ed_vs_idmrg_structure_factors.png"),
                        lambda path: save_multi_method_structure_comparison(
                            method_to_rows=method_structure_comparison,
                            filepath=path,
                            title_label=run_plot_title_label,
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                else:
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "dmrg_vs_ed_vs_idmrg_structure_factors_png",
                        output_filename("dmrg_vs_ed_vs_idmrg_structure_factors.png"),
                        "plot_structure_factors is false or no available method has structure rows",
                    )
            except Exception as exc:
                summary["ed"] = {
                    "status": "failed",
                    "error": str(exc),
                    "eligibility": ed_eligibility,
                }
                summary["stages"]["ed"] = "failed"
                if not continue_on_plot_error:
                    raise
            if summary["stages"]["ed"] not in ("failed", "skipped"):
                summary["stages"]["ed"] = "completed"
        _save_summary_checkpoint(args.output_folder, summary)

    structure_comparison_status = _plot_step_status(summary, "dmrg_vs_ed_vs_idmrg_structure_factors_png")
    if structure_comparison_status not in ("saved", "skipped_exists") and bool(args.plot_structure_factors):
        available_structure_comparison: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(dmrg_structure_factor_rows, list) and len(dmrg_structure_factor_rows) > 0:
            available_structure_comparison[finite_method_label] = dmrg_structure_factor_rows
        ed_structure_rows_from_summary = (
            summary["ed"].get("structure_factors")
            if isinstance(summary.get("ed"), dict)
            else None
        )
        if isinstance(ed_structure_rows_from_summary, list) and len(ed_structure_rows_from_summary) > 0:
            available_structure_comparison["ED"] = ed_structure_rows_from_summary
        idmrg_structure_rows_from_summary = (
            summary["idmrg"].get("structure_factors")
            if isinstance(summary.get("idmrg"), dict)
            else None
        )
        if isinstance(idmrg_structure_rows_from_summary, list) and len(idmrg_structure_rows_from_summary) > 0:
            available_structure_comparison["iDMRG-x"] = idmrg_structure_rows_from_summary
        if any(len(rows) > 0 for rows in available_structure_comparison.values()):
            _save_plot_step(
                summary,
                args.output_folder,
                "dmrg_vs_ed_vs_idmrg_structure_factors_png",
                output_filename("dmrg_vs_ed_vs_idmrg_structure_factors.png"),
                lambda path: save_multi_method_structure_comparison(
                    method_to_rows=available_structure_comparison,
                    filepath=path,
                    title_label=run_plot_title_label,
                    title="Available Method Structure Factors",
                ),
                overwrite_existing,
                continue_on_plot_error,
            )
        else:
            _skip_plot_step(
                summary,
                args.output_folder,
                "dmrg_vs_ed_vs_idmrg_structure_factors_png",
                output_filename("dmrg_vs_ed_vs_idmrg_structure_factors.png"),
                "plot_structure_factors is true, but no completed method produced structure rows",
            )
    elif structure_comparison_status is None and not bool(args.plot_structure_factors):
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_vs_ed_vs_idmrg_structure_factors_png",
            output_filename("dmrg_vs_ed_vs_idmrg_structure_factors.png"),
            "plot_structure_factors is false",
        )

    def _missing_dmrg_excited_search(reason: str) -> Dict[str, Any]:
        return {
            "method": "finite_dmrg_penalty_excited_state",
            "first_excited_energy": None,
            "spectral_gap": None,
            "status": "not_found",
            "reason": str(reason),
            "penalty_weight_used": None,
            "candidate_variance": None,
            "candidate_max_overlap": None,
        }

    if args.run_ed:
        dmrg_excited_search: Dict[str, Any] | None = None
        if not bool(args.check_ground_state_degeneracy):
            summary["stages"]["dmrg_excited_state_search"] = "not_requested"
            dmrg_excited_search = _missing_dmrg_excited_search(
                "ground-state degeneracy check disabled"
            )
        elif not (
            isinstance(summary.get("ed"), dict)
            and summary["ed"].get("status") == "completed"
        ):
            summary["stages"]["dmrg_excited_state_search"] = "skipped"
            dmrg_excited_search = _missing_dmrg_excited_search(
                "ED ground-state degeneracy unavailable"
            )
        elif bool(summary["ed"].get("use_sz_conserved", False)):
            summary["stages"]["dmrg_excited_state_search"] = "skipped"
            dmrg_excited_search = _missing_dmrg_excited_search(
                "Sz-conserved bitwise ED is not used as the finite-DMRG excited-state guide."
            )
        elif not str(backend_used).startswith("tenax"):
            summary["stages"]["dmrg_excited_state_search"] = "skipped"
            dmrg_excited_search = _missing_dmrg_excited_search(
                f"{finite_method_label} penalty excited-state search requires the Tenax MPS/MPO backend"
            )
        elif dmrg_state_obj is None or tenax_mpo is None:
            summary["stages"]["dmrg_excited_state_search"] = "skipped"
            dmrg_excited_search = _missing_dmrg_excited_search(
                "Tenax ground MPS or Hamiltonian MPO unavailable"
            )
        elif summary["ed"].get("ground_state_degeneracy") is None:
            summary["stages"]["dmrg_excited_state_search"] = "not_found"
            dmrg_excited_search = _missing_dmrg_excited_search(
                "ED ground-state degeneracy unresolved"
            )
        else:
            summary["stages"]["dmrg_excited_state_search"] = "running"
            _save_summary_checkpoint(args.output_folder, summary)
            ed_gap_hint = summary["ed"].get("spectral_gap")
            dmrg_excited_search = find_dmrg_excited_state(
                H_mpo=tenax_mpo,
                ground_mps_list=[dmrg_state_obj],
                E0=float(summary["dmrg"]["ground_state_energy"]),
                ED_gap_hint=(
                    float(ed_gap_hint)
                    if ed_gap_hint is not None
                    else None
                ),
                overlap_tol=float(args.dmrg_excited_overlap_tol),
                energy_tol=float(args.dmrg_excited_energy_tol),
                variance_tol=float(args.dmrg_excited_variance_tol),
                max_attempts=int(args.dmrg_excited_max_attempts),
                required_ground_degeneracy=int(summary["ed"]["ground_state_degeneracy"]),
            )
            if dmrg_excited_search.get("status") == "found":
                first_excited = float(dmrg_excited_search["first_excited_energy"])
                gap = float(dmrg_excited_search["spectral_gap"])
                dmrg_excited_search["first_excited_energy_per_site"] = (
                    first_excited / float(geometry.number_of_sites)
                )
                summary["dmrg"]["first_excited_energy"] = first_excited
                summary["dmrg"]["first_excited_energy_per_site"] = (
                    first_excited / float(geometry.number_of_sites)
                )
                summary["dmrg"]["spectral_gap"] = gap
                summary["stages"]["dmrg_excited_state_search"] = "completed"
            else:
                summary["stages"]["dmrg_excited_state_search"] = "not_found"

        if dmrg_excited_search is not None:
            summary["dmrg"]["penalty_excited_state_search"] = dmrg_excited_search
            _save_summary_checkpoint(args.output_folder, summary)

    method_energy_comparison = {
        finite_method_label: float(summary["dmrg"]["energy_per_site"]),
    }
    if (
        isinstance(summary.get("ed"), dict)
        and summary["ed"].get("status") == "completed"
        and "energy_per_site" in summary["ed"]
    ):
        method_energy_comparison["ED"] = float(summary["ed"]["energy_per_site"])
    idmrg_energy_per_site = _finite_float_from_mapping(
        summary.get("idmrg"),
        "energy_per_original_site",
        "ground_state_energy_per_site",
        "energy_per_site",
    )
    if (
        isinstance(summary.get("idmrg"), dict)
        and summary["idmrg"].get("status") == "completed"
        and idmrg_energy_per_site is not None
    ):
        method_energy_comparison["iDMRG-x"] = idmrg_energy_per_site
    ipeps_energy_per_site = _finite_float_from_mapping(
        summary.get("ipeps"),
        "ground_state_energy_per_site",
        "energy_per_site",
    )
    if (
        isinstance(summary.get("ipeps"), dict)
        and summary["ipeps"].get("status") == "completed"
        and ipeps_energy_per_site is not None
    ):
        method_energy_comparison["iPEPS"] = ipeps_energy_per_site
    if bool(args.plot_energy_comparison):
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_vs_ed_vs_idmrg_energy_png",
            output_filename("dmrg_vs_ed_vs_idmrg_energy.png"),
            lambda path: save_multi_method_energy_comparison(
                method_to_energy=method_energy_comparison,
                filepath=path,
                title="Available Method Energy Per Site",
                title_label=run_plot_title_label,
            ),
            overwrite_existing or "iDMRG-x" in method_energy_comparison,
            continue_on_plot_error,
        )
    else:
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_vs_ed_vs_idmrg_energy_png",
            output_filename("dmrg_vs_ed_vs_idmrg_energy.png"),
            "plot_energy_comparison is false",
        )

    quimb_comparison_key = "ipeps" if primary_is_ipeps else ("peps" if finite_is_peps else None)
    if (
        quimb_comparison_key is not None
        and isinstance(summary.get(quimb_comparison_key), dict)
    ):
        peps_plot_payload = dict(summary[quimb_comparison_key])
        peps_plot_payload.setdefault("alpha", float(args.alpha))
        peps_plot_payload.setdefault("beta", float(args.beta))
        if isinstance(summary.get("dmrg"), dict):
            peps_plot_payload.setdefault("energy_per_site", summary["dmrg"].get("energy_per_site"))
        ed_plot_payload = (
            dict(summary["ed"])
            if isinstance(summary.get("ed"), dict) and summary["ed"].get("status") == "completed"
            else {}
        )
        if ed_plot_payload:
            ed_plot_payload.setdefault("alpha", float(args.alpha))
            ed_plot_payload.setdefault("beta", float(args.beta))
        comparison_title = (
            "iPEPS Benchmark With Available ED Reference"
            if primary_is_ipeps
            else "Finite PEPS Benchmark With Available ED Reference"
        )
        comparison_filename = "ipeps_vs_ed_comparison.png" if primary_is_ipeps else "peps_vs_ed_comparison.png"
        _save_plot_step(
            summary,
            args.output_folder,
            f"{quimb_comparison_key}_vs_ed_comparison_png",
            output_filename(comparison_filename),
            lambda path, peps_payload=peps_plot_payload, ed_payload=ed_plot_payload, label=finite_method_label, title=comparison_title: plot_peps_vs_ed_comparison(
                peps_results=peps_payload,
                ed_results=ed_payload,
                filepath=path,
                title=title,
                title_label=run_plot_title_label,
                peps_label=label,
                ed_label="ED",
            ),
            True,
            continue_on_plot_error,
        )
    elif quimb_comparison_key is not None:
        _skip_plot_step(
            summary,
            args.output_folder,
            f"{quimb_comparison_key}_vs_ed_comparison_png",
            output_filename("ipeps_vs_ed_comparison.png" if primary_is_ipeps else "peps_vs_ed_comparison.png"),
            "PEPS/iPEPS result was not available, so no comparison plot could be written.",
        )

    dmrg_excited_result = (
        summary["dmrg"].get("penalty_excited_state_search")
        if isinstance(summary.get("dmrg"), dict)
        else None
    )
    dmrg_ed_degeneracy = (
        summary["ed"].get("ground_state_degeneracy")
        if isinstance(summary.get("ed"), dict)
        else None
    )
    dmrg_degeneracy_check_enabled = bool(
        isinstance(summary.get("ed"), dict)
        and summary["ed"].get("ground_state_degeneracy_check_enabled", False)
    )
    dmrg_first_excited = None
    dmrg_first_excited_per_site = None
    dmrg_gap = None
    if (
        isinstance(dmrg_excited_result, dict)
        and dmrg_excited_result.get("status") == "found"
    ):
        dmrg_first_excited = dmrg_excited_result.get("first_excited_energy")
        dmrg_first_excited_per_site = dmrg_excited_result.get("first_excited_energy_per_site")
        if dmrg_first_excited_per_site is None and dmrg_first_excited is not None:
            dmrg_first_excited_per_site = (
                float(dmrg_first_excited) / float(geometry.number_of_sites)
            )
        dmrg_gap = dmrg_excited_result.get("spectral_gap")
    dmrg_degeneracy_status = (
        "not_checked"
        if not dmrg_degeneracy_check_enabled
        else ("ed_guided" if dmrg_ed_degeneracy is not None else "not_resolved")
    )
    method_spectrum_comparison: Dict[str, Dict[str, Any]] = {
        finite_method_label: {
            "status": (
                "completed"
                if isinstance(dmrg_excited_result, dict)
                and dmrg_excited_result.get("status") == "found"
                else "not_found"
            ),
            "ground_state_energy": float(summary["dmrg"]["ground_state_energy"]),
            "ground_state_energy_per_site": float(summary["dmrg"]["energy_per_site"]),
            "ground_state_degeneracy_check_enabled": dmrg_degeneracy_check_enabled,
            "ground_state_degeneracy": dmrg_ed_degeneracy,
            "ground_state_degeneracy_label": (
                f"{dmrg_ed_degeneracy}" if dmrg_ed_degeneracy is not None else "unresolved"
            ),
            "ground_state_degeneracy_status": dmrg_degeneracy_status,
            "ground_state_degeneracy_is_lower_bound": (
                summary["ed"].get("ground_state_degeneracy_is_lower_bound")
                if isinstance(summary.get("ed"), dict)
                else None
            ),
            "ground_state_degeneracy_tolerance": (
                summary["ed"].get("ground_state_degeneracy_tolerance")
                if isinstance(summary.get("ed"), dict)
                else None
            ),
            "ground_state_degeneracy_absolute_tolerance": (
                summary["ed"].get("ground_state_degeneracy_absolute_tolerance")
                if isinstance(summary.get("ed"), dict)
                else None
            ),
            "ground_state_degeneracy_relative_tolerance": (
                summary["ed"].get("ground_state_degeneracy_relative_tolerance")
                if isinstance(summary.get("ed"), dict)
                else None
            ),
            "first_excited_energy": dmrg_first_excited,
            "first_excited_energy_per_site": dmrg_first_excited_per_site,
            "spectral_gap": dmrg_gap,
            "excited_state_search": dmrg_excited_result,
            "note": (
                f"{finite_method_label} first-excited energy and gap are reported only when "
                "the penalty-state search finds a distinct low-variance state "
                "orthogonal to the ED-guided ground manifold."
            ),
        }
    }
    if (
        isinstance(summary.get("ed"), dict)
        and summary["ed"].get("status") == "completed"
    ):
        ed_entry: Dict[str, Any] = {
            "status": "completed",
            "ground_state_energy": float(summary["ed"]["ground_state_energy"]),
            "ground_state_energy_per_site": float(summary["ed"]["energy_per_site"]),
            "ground_state_degeneracy_check_enabled": bool(
                summary["ed"].get("ground_state_degeneracy_check_enabled", False)
            ),
            "ground_state_degeneracy": summary["ed"].get("ground_state_degeneracy"),
            "ground_state_degeneracy_label": (
                f"{summary['ed'].get('ground_state_degeneracy')}"
                if summary["ed"].get("ground_state_degeneracy") is not None
                else (
                    "not checked"
                    if not bool(summary["ed"].get("ground_state_degeneracy_check_enabled", False))
                    else "unresolved"
                )
            ),
            "ground_state_degeneracy_status": (
                "not_checked"
                if not bool(summary["ed"].get("ground_state_degeneracy_check_enabled", False))
                else (
                "lower_bound"
                if bool(summary["ed"].get("ground_state_degeneracy_is_lower_bound", False))
                else (
                    "resolved"
                    if summary["ed"].get("ground_state_degeneracy") is not None
                    else "not_resolved"
                )
                )
            ),
            "ground_state_degeneracy_is_lower_bound": summary["ed"].get("ground_state_degeneracy_is_lower_bound"),
            "ground_state_degeneracy_tolerance": summary["ed"].get("ground_state_degeneracy_tolerance"),
            "ground_state_degeneracy_absolute_tolerance": summary["ed"].get("ground_state_degeneracy_absolute_tolerance"),
            "ground_state_degeneracy_relative_tolerance": summary["ed"].get("ground_state_degeneracy_relative_tolerance"),
            "first_excited_energy": summary["ed"].get("first_excited_energy"),
            "first_excited_energy_per_site": summary["ed"].get("first_excited_energy_per_site"),
            "spectral_gap": summary["ed"].get("spectral_gap"),
        }
        if isinstance(summary["ed"].get("spectrum"), dict):
            ed_entry["spectrum"] = summary["ed"]["spectrum"]
        method_spectrum_comparison["ED"] = ed_entry
    if (
        isinstance(summary.get("idmrg"), dict)
        and summary["idmrg"].get("status") == "completed"
        and idmrg_energy_per_site is not None
    ):
        method_spectrum_comparison["iDMRG-x"] = {
            "status": "ground_state_only",
            "ground_state_energy": None,
            "ground_state_energy_per_site": idmrg_energy_per_site,
            "ground_state_degeneracy": None,
            "ground_state_degeneracy_label": "unresolved",
            "ground_state_degeneracy_status": "not_resolved",
            "first_excited_energy": None,
            "first_excited_energy_per_site": None,
            "spectral_gap": None,
            "note": (
                "This driver uses iDMRG-x ground-state workflows only, so no compatible "
                "iDMRG first-excited energy or spectral gap is reported."
            ),
        }
    summary["low_energy_spectrum_comparison"] = method_spectrum_comparison
    _save_summary_checkpoint(args.output_folder, summary)
    if bool(args.plot_low_energy_spectrum):
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_vs_ed_vs_idmrg_low_energy_spectrum_png",
            output_filename("dmrg_vs_ed_vs_idmrg_low_energy_spectrum.png"),
            lambda path: save_low_energy_spectrum_comparison(
                method_spectra=method_spectrum_comparison,
                filepath=path,
                title_label=run_plot_title_label,
            ),
            overwrite_existing or "iDMRG-x" in method_spectrum_comparison,
            continue_on_plot_error,
        )
    else:
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_vs_ed_vs_idmrg_low_energy_spectrum_png",
            output_filename("dmrg_vs_ed_vs_idmrg_low_energy_spectrum.png"),
            "plot_low_energy_spectrum is false",
        )

    if args.run_finite_temperature:
        summary["stages"]["finite_temperature"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        local_dim = int(model_spec.physical_dim)
        thermal_hilbert_dim = int(local_dim ** geometry.number_of_sites)
        if ed_applied_reductions:
            summary["finite_temperature"] = {
                "status": "skipped",
                "reason": (
                    "Finite-temperature ED still uses the legacy full-Hilbert-space path; "
                    "it is skipped when shared ED symmetry reductions are active to avoid mixing ED Hamiltonians."
                ),
            }
            summary["stages"]["finite_temperature"] = "skipped"
        elif geometry.number_of_sites > int(args.thermal_max_sites):
            summary["finite_temperature"] = {
                "status": "skipped",
                "reason": (
                    f"Finite-temperature ED is limited to {int(args.thermal_max_sites)} sites or fewer."
                ),
            }
            summary["stages"]["finite_temperature"] = "skipped"
        elif thermal_hilbert_dim > int(args.thermal_max_hilbert_dim):
            summary["finite_temperature"] = {
                "status": "skipped",
                "reason": (
                    f"Finite-temperature ED Hilbert-space dimension {thermal_hilbert_dim} "
                    f"exceeds limit {int(args.thermal_max_hilbert_dim)} "
                    f"(local_dim={local_dim}, sites={geometry.number_of_sites})."
                ),
            }
            summary["stages"]["finite_temperature"] = "skipped"
        else:
            try:
                thermal_results = run_finite_temperature_ed(
                    geometry=geometry,
                    model_spec=model_spec,
                    alpha=args.alpha,
                    beta=args.beta,
                    coupling_j=args.coupling_j,
                    lattice=lattice_name,
                    temperature_min=args.temperature_min,
                    temperature_max=args.temperature_max,
                    temperature_points=args.temperature_points,
                    temperature_scale=args.temperature_scale,
                    max_eigenstates=args.thermal_max_eigenstates,
                    full_spectrum_max_dim=args.thermal_full_spectrum_max_dim,
                    jx=args.jx,
                    jy=args.jy,
                    jz=args.jz,
                    external_field_terms=hamiltonian_external_field_terms,
                    show_progress=show_progress,
                    ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                    ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                )
                thermal_results["symmetry_reductions"] = symmetry_reduction_settings
                thermal_results["symmetry_note"] = (
                    "Finite-temperature ED uses the full Hilbert trace unless a future implementation "
                    "explicitly sums over all conserved sectors; a single Tz sector would not represent "
                    "the thermal ensemble."
                )
                zero_temperature_references = thermal_results.setdefault("zero_temperature_references", {})
                if not isinstance(zero_temperature_references, dict):
                    zero_temperature_references = {}
                    thermal_results["zero_temperature_references"] = zero_temperature_references
                zero_temperature_references["DMRG"] = build_zero_temperature_dmrg_reference(
                        geometry=geometry,
                        dmrg_energy=dmrg_energy,
                        scalar_correlations=dmrg_scalar_correlations,
                        bond_rows=dmrg_bond_rows,
                        structure_factor_rows=dmrg_structure_factor_rows,
                        uniform_observables=dmrg_uniform_observables,
                    )
                summary["finite_temperature"] = {
                    "status": "completed",
                    **thermal_results,
                }
                summary["stages"]["finite_temperature"] = "completed"
                _save_summary_checkpoint(args.output_folder, summary)
                if bool(args.plot_finite_temperature):
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "finite_temperature_observables_png",
                        output_filename("finite_temperature_observables.png"),
                        lambda path: save_finite_temperature_observables_plot(
                            thermal_results,
                            path,
                            title_label=run_plot_title_label,
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "finite_temperature_correlations_png",
                        output_filename("finite_temperature_correlations.png"),
                        lambda path: save_finite_temperature_correlations_plot(
                            thermal_results,
                            path,
                            title_label=run_plot_title_label,
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "finite_temperature_structure_factors_png",
                        output_filename("finite_temperature_structure_factors.png"),
                        lambda path: save_finite_temperature_structure_factors_plot(
                            thermal_results,
                            path,
                            title_label=run_plot_title_label,
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                else:
                    for output_key, base_name in (
                        ("finite_temperature_observables_png", "finite_temperature_observables.png"),
                        ("finite_temperature_correlations_png", "finite_temperature_correlations.png"),
                        ("finite_temperature_structure_factors_png", "finite_temperature_structure_factors.png"),
                    ):
                        _skip_plot_step(
                            summary,
                            args.output_folder,
                            output_key,
                            output_filename(base_name),
                            "plot_finite_temperature is false",
                        )
            except Exception as exc:
                summary["finite_temperature"] = {"status": "failed", "error": str(exc)}
                summary["stages"]["finite_temperature"] = "failed"
                if not continue_on_plot_error:
                    raise
        _save_summary_checkpoint(args.output_folder, summary)

    if len(entropy_profiles) > 0 and bool(args.plot_entanglement):
        _save_plot_step(
            summary,
            args.output_folder,
            "entanglement_entropy_profiles_png",
            output_filename("entanglement_entropy_profiles.png"),
            lambda path: save_entropy_profiles_comparison(
                entropy_profiles=entropy_profiles,
                filepath=path,
                orders=ENTROPY_ORDERS,
                title_label=run_plot_title_label,
            ),
            overwrite_existing or "iDMRG-x" in entropy_profiles,
            continue_on_plot_error,
        )
        _save_plot_step(
            summary,
            args.output_folder,
            "entanglement_entropy_method_means_png",
            output_filename("entanglement_entropy_method_means.png"),
            lambda path: save_entropy_method_means_comparison(
                entropy_profiles=entropy_profiles,
                filepath=path,
                orders=ENTROPY_ORDERS,
                title_label=run_plot_title_label,
            ),
            overwrite_existing or "iDMRG-x" in entropy_profiles,
            continue_on_plot_error,
        )
    elif len(entropy_profiles) > 0:
        for output_key, base_name in (
            ("entanglement_entropy_profiles_png", "entanglement_entropy_profiles.png"),
            ("entanglement_entropy_method_means_png", "entanglement_entropy_method_means.png"),
        ):
            _skip_plot_step(
                summary,
                args.output_folder,
                output_key,
                output_filename(base_name),
                "plot_entanglement is false",
            )

    if args.run_phase_scan:
        summary["stages"]["phase_scan"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        try:
            phase_scan_data = run_requested_phase_scan_for_geometry(
                geometry,
                incremental_summary_obj=summary,
            )
            summary["phase_scan"] = phase_scan_data
            summary["stages"]["phase_scan"] = "completed"
            save_phase_scan_outputs(summary, phase_scan_data, geometry)
            if _attach_plot_output_warnings(summary, "phase_scan"):
                summary["stages"]["phase_scan"] = "completed_with_warnings"
        except Exception as exc:
            error_text = str(exc) or exc.__class__.__name__
            summary["phase_scan"] = {"status": "failed", "error": error_text}
            summary["stages"]["phase_scan"] = "failed"
            if not continue_on_plot_error:
                raise
        _save_summary_checkpoint(args.output_folder, summary)

    output_warning_keys = _attach_plot_output_warnings(summary, "phase_scan")
    if (
        output_warning_keys
        or (isinstance(summary.get("ed"), dict) and summary["ed"].get("status") == "failed")
        or (isinstance(summary.get("idmrg"), dict) and summary["idmrg"].get("status") == "failed")
        or (
            isinstance(summary.get("finite_temperature"), dict)
            and summary["finite_temperature"].get("status") == "failed"
        )
        or (
            isinstance(summary.get("phase_scan"), dict)
            and summary["phase_scan"].get("status") in ("failed", "completed_with_warnings")
        )
    ):
        summary["run_status"] = "completed_with_warnings"
    else:
        summary["run_status"] = "completed"
    finalize_summary_with_profiling(summary)
    _save_summary_checkpoint(args.output_folder, summary)
    print(
        "[run] finished: "
        f"status={summary['run_status']}, summary={os.path.join(args.output_folder, run_summary_filename)}"
    )


if __name__ == "__main__":
    main()
