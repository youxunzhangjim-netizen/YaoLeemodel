#!/usr/bin/env python3
"""
CLI orchestration for Yao-Lee model benchmarking with Tenax/TeNPy DMRG and optional ED.

The canonical work is split across sibling modules:
- models.py: model specs, local operators, geometry, and shared physics helpers.
- ed_backend.py: full ED plus bitwise total-Sz-conserved sparse ED.
- tenax_backend.py: Tenax MPO/DMRG/iDMRG execution.
- tenpy_backend.py: TeNPy YaoLeeSite/YaoLeeModel execution.
- analysis.py: phase scans, entropy, diagnostics, and summary helpers.
- plot_outputs.py: plotting helpers.

This file keeps settings, CLI, consistency checks, and run orchestration.
main() binds the split modules before running so fixes stay shared by owner.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Configuration (edit this top block for normal runs)
# ----------------------------------------------------------------------
# Resource profiles.
# Edit ACTIVE_RESOURCE_PROFILE to switch all geometry/DMRG/ED/iDMRG defaults
# together. Keep larger aragorn/beehive choices on the command line or in a new
# profile so local/shared-machine runs stay polite by default.
#
# Geometry tuning uses four independent options everywhere:
#   length_x, length_y: number of unit cells.
#   circumference_x, circumference_y: whether x/y boundaries are closed.
LOCAL_LAPTOP_SETTINGS = {
    "geometry": {
        "length_x": 2,
        "length_y": 2,
        "circumference_x": True,
        "circumference_y": True,
        "lattice_type": "honeycomb",
    },
    "finite_dmrg": {
        "max_sites": 18,
        "max_bond_dimension": 64,
        "max_sweeps": 10,
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
        "max_bond_dimension": 32,
        "max_iterations": 10,
        "max_local_dim": 16,
        "bulk_kind": "auto",
    },
}

SHARED_WORKSTATION_SETTINGS = {
    "geometry": {
        "length_x": 3,
        "length_y": 3,
        "circumference_x": True,
        "circumference_y": True,
        "lattice_type": "honeycomb",
    },
    "finite_dmrg": {
        "max_sites": 32,
        "max_bond_dimension": 128,
        "max_sweeps": 20,
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
        "max_bond_dimension": 32,
        "max_iterations": 40,
        "max_local_dim": 16,
        "bulk_kind": "auto",
    },
}

RESOURCE_PROFILES = {
    "local_laptop": LOCAL_LAPTOP_SETTINGS,
    "shared_workstation": SHARED_WORKSTATION_SETTINGS,
    }


ACTIVE_RESOURCE_PROFILE = "local_laptop"  # local_laptop | shared_workstation

# Change this one line when you want to force the DMRG/tensor-network backend:
#   "tenax" -> JAX/Tenax path
#   "tenpy" -> TeNPy U(1) Yao-Lee template path
#   "auto"  -> try Tenax first, then compatible TeNPy fallback
BACKEND = "tenpy"  # auto | tenax | tenpy

# - yao_lee: spin/orbital model using spin-dot and orbital-compass bonds:
#       J[(1+beta) S_i.S_j + (1-beta) T_i^gamma T_j^gamma + alpha (S_i.S_j)(T_i^gamma T_j^gamma)].
#       This conserves total spin Sz and does not conserve orbital tau_z.
# - ising_like: every bond uses the single ISING_AXIS channel.
# - heisenberg/xy/xxz/xyz: spin-only benchmark models; set ORBITAL_REP=0.
# - orbital_rep=0: removes orbital DOF and reduces yao_lee to spin-only Ising-like bonds.
MODEL_FAMILY = "yao_lee"    # yao_lee | ising_like | heisenberg | xy | xxz | xyz
SPIN_REP = "1/2"            # 1/2 | 3/2
ORBITAL_REP = "1/2"         # 0 | 1/2 ; CLI also accepts legacy alias 1 -> 0
ISING_AXIS = "z"            # x | y | z
ALPHA = 1.0
BETA = 0.5
COUPLING_J = 1.0          # overall exchange scale; zero gives a deliberately empty Hamiltonian
JX = 1.0                    # simple XY/XXZ/XYZ benchmark coupling multiplier
JY = 1.0
JZ = 1.0

# External spin Zeeman field / perturbation.
# eg orbital angular momentum is taken as L=0, so the field couples only to spin:
#   H_Z = FIELD_SIGN * MU_B * SIGMA_FACTOR * sum_i (hx*Sx_i + hy*Sy_i + hz*Sz_i)
# SIGMA_FACTOR=2 maps sigma=2S for spin-1/2. Keep this explicit if exploring spin=3/2.
# treatment:
# - perturbation: recorded and annotated, but not inserted into the MPO/ED Hamiltonian.
# - hamiltonian: inserted as one-site terms; use symmetry_mode=none for hx/hy fields.
EXTERNAL_FIELD_TREATMENT = "off"  # off | perturbation | hamiltonian
EXTERNAL_FIELD_AXIS = "custom"                # 111 | custom
EXTERNAL_FIELD_STRENGTH = 1.0              # used for axis=111 as H/sqrt(3)*(1,1,1)
FIELD_HX = 0.0                             # used for axis=custom
FIELD_HY = 0.0
FIELD_HZ = 1.0
MU_B = 1.0
FIELD_SIGN = 1.0
FIELD_SIGMA_FACTOR = 2.0

# Shared symmetry simplification/block-sparse controls.
# SYMMETRY_REDUCTIONS is additive: combine any of "sz", "tz", and "z2".
# "auto" asks the precheck to enable every conserved/reachable reduction that a
# method can implement. Old CLI values like --symmetry-mode u1_sz are still
# accepted, but normal runs should edit only SYMMETRY_REDUCTIONS.
SYMMETRY_REDUCTIONS = ("sz","z2")  # auto | none | sz | tz | z2 ; e.g. ("sz", "tz", "z2")
U1_TARGET_TOTAL_SZ2 = 0     # equals 2 * total S^z; neutral sector is usually 0
U1_TARGET_TOTAL_TZ2 = 0     # equals 2 * total tau^z/T^z; neutral sector is usually 0
Z2_TARGET_PARITY = 0        # 0=even, 1=odd
STRICT_SYMMETRY_SELECTION_RULES = True
SYMMETRY_PRECHECK = True
STRICT_SYMMETRY_PRECHECK = True
SYMMETRY_ALLOW_DENSE_FALLBACK = True

TRUNCATION_CUTOFF = 1e-8
SEED = 42
INITIAL_STATE_STYLE = "random"  # alternating | random

# Optional comparison workflows.
ED_BACKEND = "quspin"  # standard | quspin ; CLI also accepts ed -> standard
SZ_CONSERVED_ED_EIGENSTATES = 3
CHECK_GROUND_STATE_DEGENERACY = True
ED_GROUND_MANIFOLD_ABS_TOL = 1e-12
ED_GROUND_MANIFOLD_REL_TOL = 1e-12
DMRG_EXCITED_OVERLAP_TOL = 1e-6
DMRG_EXCITED_ENERGY_TOL = 1e-7
DMRG_EXCITED_VARIANCE_TOL = 1e-7
DMRG_EXCITED_MAX_ATTEMPTS = 10

# Spatial symmetry reductions are shared options. A backend uses them only when
# it has an implementation for that block and the geometry supports it.  For a
# honeycomb cylinder, x is usually open while y is periodic; keep the directions
# independent so the valid y momentum block can still be used.
USE_TRANSLATION_X_BLOCK = 0
USE_TRANSLATION_Y_BLOCK = 0
MOMENTUM_X_BLOCK = 0
MOMENTUM_Y_BLOCK = 0
USE_REFLECTION_BLOCK = 0
REFLECTION_BLOCK = 0  # Reflection/C3 is unsafe for bond-directional Yao-Lee unless a gauge map is implemented.
QUSPIN_CHECK_SYMMETRIES = False
QUSPIN_CHECK_HERMITICITY = True
QUSPIN_CHECK_PARTICLE_CONSERVATION = False

# Optional alpha-beta phase diagrams.
# PHASE_DIAGRAM_ENABLED is the single switch for both scan calculation and plot
# output.
# PHASE_SCAN_MODE chooses which physics level to scan:
#   quantum   -> run the quantum methods listed in PHASE_SCAN_METHODS.
#   classical -> run only the classical product-state scan.
#   both      -> run the quantum methods plus the classical product-state scan.
# PHASE_SCAN_METHODS chooses quantum solvers only. Use any comma-separated
# subset of ed, dmrg, idmrg, or use all for every quantum solver.
PHASE_SCAN_ONLY = 0
PHASE_DIAGRAM_ENABLED = 1 or PHASE_SCAN_ONLY
RUN_PHASE_SCAN = PHASE_DIAGRAM_ENABLED or PHASE_SCAN_ONLY

PHASE_SCAN_MODE = "quantum"    # quantum | classical | both
PHASE_SCAN_METHODS = "dmrg"    # ed | dmrg | idmrg | all
# Phase-scan ED uses the same SYMMETRY_REDUCTIONS and target-sector options as
# single-point ED. For spin+orbital N=18, even the reduced dimension
# C(18,9)*2^18 = 12,745,441,280 exceeds the default cap; use
# length_x=2, length_y=3, circumference_x=False, circumference_y=True
# (N=12 for honeycomb) or smaller for quantum ED scans.
PHASE_SCAN_ALPHA_MIN = 0.0
PHASE_SCAN_ALPHA_MAX = 2.25
PHASE_SCAN_ALPHA_POINTS = 17
PHASE_SCAN_BETA_MIN = 0.0
PHASE_SCAN_BETA_MAX = 0.27
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

# Output/runtime behavior.
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "outputs")
OVERWRITE_EXISTING_PLOTS = True
CONTINUE_AFTER_PLOT_ERROR = True
STRICT_PLOT_ERRORS = not CONTINUE_AFTER_PLOT_ERROR
SHOW_PROGRESS = True

# Observable calculation controls.
# Calculation flags decide whether expensive post-processing is performed.
# Plot flags decide whether an already-computed observable is written as PNG.
CALCULATE_CORRELATIONS = True
CALCULATE_BOND_ENERGIES = True
CALCULATE_STRUCTURE_FACTORS = True
CALCULATE_ENTANGLEMENT = True
CALCULATE_UNIFORM_OBSERVABLES = True
CALCULATE_REAL_SPACE_PATTERNS = True
REFERENCE_SITE_IDX = None  # None chooses the site closest to the geometric center.

PLOT_GEOMETRY = True
PLOT_BOND_ENERGIES = True
PLOT_STRUCTURE_FACTORS = True
PLOT_CORRELATION_HEATMAPS = True
PLOT_REAL_SPACE_PATTERNS = True
PLOT_ENTANGLEMENT = True
PLOT_ENERGY_COMPARISON = True
PLOT_LOW_ENERGY_SPECTRUM = True
PLOT_FINITE_TEMPERATURE = True
PLOT_PHASE_SCAN = PHASE_DIAGRAM_ENABLED

# ----------------------------------------------------------------------
# Derived/profile-linked defaults and available choices
# ----------------------------------------------------------------------

# Values below are linked to ACTIVE_RESOURCE_PROFILE or used only as parser
# choice tables. Keep normal run edits in the option block above.
ACTIVE_RESOURCE_SETTINGS = RESOURCE_PROFILES[ACTIVE_RESOURCE_PROFILE]

# Geometry defaults from ACTIVE_RESOURCE_SETTINGS.
LENGTH_X = int(ACTIVE_RESOURCE_SETTINGS["geometry"]["length_x"])
LENGTH_Y = int(ACTIVE_RESOURCE_SETTINGS["geometry"]["length_y"])
CIRCUMFERENCE_X = bool(ACTIVE_RESOURCE_SETTINGS["geometry"]["circumference_x"])
CIRCUMFERENCE_Y = bool(ACTIVE_RESOURCE_SETTINGS["geometry"]["circumference_y"])
LATTICE_TYPE = str(ACTIVE_RESOURCE_SETTINGS["geometry"]["lattice_type"])  # honeycomb | square | triangular

# Resource-limited solver defaults from ACTIVE_RESOURCE_SETTINGS.
MAX_DMRG_SITES = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_sites"])
MAX_BOND_DIMENSION = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_bond_dimension"])
MAX_SWEEPS = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_sweeps"])

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
SYMMETRY_MODE_OPTIONS = ("none", "auto", "u1", "u1_sz", "u1_tz", "z2")
SYMMETRY_REDUCTION_OPTIONS = ("auto", "none", "sz", "tz", "z2", "u1", "u1_sz", "u1_tz")
U1_CHARGE_TZ_STRIDE = 4096
Z2_PARITY_OPTIONS = (0, 1)
IDMRG_BULK_KIND_OPTIONS = ("auto", "pair", "single")
BACKEND_OPTIONS = ("auto", "tenax", "tenpy")
ED_BACKEND_OPTIONS = ("standard", "ed", "quspin")
ED_SOLVER_OPTIONS = ("auto", "sparse", "dense")
EXTERNAL_FIELD_TREATMENT_OPTIONS = ("off", "perturbation", "hamiltonian")
EXTERNAL_FIELD_AXIS_OPTIONS = ("custom", "111")
PHASE_SCAN_QUANTUM_METHOD_OPTIONS = ("ed", "dmrg", "idmrg")
PHASE_SCAN_METHOD_OPTIONS = PHASE_SCAN_QUANTUM_METHOD_OPTIONS + ("all",)
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
    "tenax_dmrg",
    "tenax_idmrg",
    "tenpy_dmrg",
    "tenpy_idmrg",
)
REFLECTION_BLOCK_OPTIONS = (-1, 0, 1)
ENTROPY_ORDERS = (1, 2, 3, 4)

# Runtime implementation symbols are imported from sibling modules by
# _bind_split_module_implementations(). Keep implementation work in:
# models.py, ed_backend.py, tenax_backend.py, tenpy_backend.py, analysis.py,
# and plot_outputs.py.
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
        help="Hamiltonian family. With orbital_rep=0, yao_lee automatically reduces to spin-only Ising-like couplings.",
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
            "Aliases u1/u1_sz/u1_tz are accepted."
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
            "Legacy single symmetry shortcut. Prefer --symmetry-reductions to combine sz/tz/z2."
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
        help="External field direction source: 111 uses H/sqrt(3)*(1,1,1); custom uses hx/hy/hz.",
    )
    parser.add_argument(
        "--external-field-strength",
        "--external_field_strength",
        dest="external_field_strength",
        type=float,
        default=EXTERNAL_FIELD_STRENGTH,
        help="Field magnitude H used when external_field_axis=111.",
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
        help="Multiplier converting spin operators to sigma; use 2 for sigma=2S in spin-1/2.",
    )
    parser.add_argument(
        "--max-bond-dimension",
        "--max_bond_dimension",
        dest="max_bond_dimension",
        type=int,
        default=MAX_BOND_DIMENSION,
        help="Finite-DMRG maximum bond dimension.",
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
    parser.add_argument("--max-sweeps", "--max_sweeps", dest="max_sweeps", type=int, default=MAX_SWEEPS)
    parser.add_argument(
        "--truncation-cutoff",
        "--truncation_cutoff",
        dest="truncation_cutoff",
        type=float,
        default=TRUNCATION_CUTOFF,
        help="TeNPy truncation cutoff; Tenax backend may ignore it.",
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
        dest="idmrg_max_bond_dimension",
        type=int,
        default=IDMRG_MAX_BOND_DIMENSION,
        help="iDMRG maximum bond dimension, independent of finite-DMRG chi.",
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
        "--phase-diagram",
        "--phase_diagram",
        dest="phase_diagram",
        action=argparse.BooleanOptionalAction,
        default=PHASE_DIAGRAM_ENABLED,
        help="Combined switch: enable/disable both phase-scan calculation and phase-diagram plots.",
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
        help="Run only the alpha-beta phase scan and skip the single-point DMRG workflow.",
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
            "Comma-separated quantum phase-scan methods: ed, dmrg, idmrg, or all. "
            "Used when --phase-scan-mode is quantum or both."
        ),
    )
    parser.add_argument(
        "--phase-scan-mode",
        "--phase_scan_mode",
        dest="phase_scan_mode",
        type=str,
        choices=list(PHASE_SCAN_MODE_OPTIONS),
        default=PHASE_SCAN_MODE,
        help="High-level phase scan content: quantum, classical, or both. Legacy solver aliases are accepted.",
    )
    parser.add_argument("--phase-scan-alpha-min", "--phase_scan_alpha_min", dest="phase_scan_alpha_min", type=float, default=PHASE_SCAN_ALPHA_MIN)
    parser.add_argument("--phase-scan-alpha-max", "--phase_scan_alpha_max", dest="phase_scan_alpha_max", type=float, default=PHASE_SCAN_ALPHA_MAX)
    parser.add_argument("--phase-scan-alpha-points", "--phase_scan_alpha_points", dest="phase_scan_alpha_points", type=int, default=PHASE_SCAN_ALPHA_POINTS)
    parser.add_argument("--phase-scan-beta-min", "--phase_scan_beta_min", dest="phase_scan_beta_min", type=float, default=PHASE_SCAN_BETA_MIN)
    parser.add_argument("--phase-scan-beta-max", "--phase_scan_beta_max", dest="phase_scan_beta_max", type=float, default=PHASE_SCAN_BETA_MAX)
    parser.add_argument("--phase-scan-beta-points", "--phase_scan_beta_points", dest="phase_scan_beta_points", type=int, default=PHASE_SCAN_BETA_POINTS)
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
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(BACKEND_OPTIONS),
        default=BACKEND,
        help=(
            "Select the DMRG/tensor-network backend. auto tries Tenax first, then falls back "
            "to the local TeNPy U1 template when compatible. Use --ed-backend for ED."
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
        help="Extract reference-site spin/orbital correlation rows for real-space order-pattern diagrams.",
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
        help="Save reference-site real-space pattern diagrams for S and T correlations.",
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
        "max_dmrg_sites",
        "max_bond_dimension",
        "max_sweeps",
        "run_idmrg",
        "idmrg_max_bond_dimension",
        "idmrg_max_iterations",
        "idmrg_max_local_dim",
        "idmrg_bulk_kind",
        "seed",
        "run_ed",
        "max_ed_sites",
        "max_ed_hilbert_dim",
        "ed_max_eigenstates",
        "ed_backend",
        "ed_solver",
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
    if key == "all":
        return list(PHASE_SCAN_QUANTUM_METHOD_OPTIONS)
    return _split_phase_scan_csv(PHASE_SCAN_METHODS)


def _normalize_phase_scan_quantum_methods(
    methods_value: Any,
    legacy_mode: str | None = None,
) -> List[str]:
    """Normalize quantum phase-scan solver choices to ed/dmrg/idmrg."""
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
    """Backward-compatible wrapper returning concrete ed/dmrg/idmrg/classical methods."""
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


def _normalize_symmetry_reductions(value: Any, legacy_mode: str | None = None) -> tuple[str, ...]:
    default_items = (
        [str(legacy_mode)]
        if legacy_mode is not None and str(legacy_mode).strip() != ""
        else [str(item) for item in SYMMETRY_REDUCTIONS]
    )
    if value is None:
        raw_items: List[str] = list(default_items)
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
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
        if mapped == "u1":
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
    return {
        "source": "shared_symmetry_reductions",
        "requested_reductions": list(reductions),
        "legacy_tenax_mode": _legacy_symmetry_mode_from_reductions(reductions),
        "use_sz_block": bool(use_sz_block),
        "use_tau_z_block": bool(use_tau_z_block),
        "use_z2_block": bool(use_z2_block),
        "target_sz2": int(getattr(args, "u1_target_sz2", U1_TARGET_TOTAL_SZ2)),
        "target_tz2": int(getattr(args, "u1_target_tz2", U1_TARGET_TOTAL_TZ2)),
        "z2_target_parity": int(getattr(args, "z2_target_parity", Z2_TARGET_PARITY)) % 2,
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
    print(f"[output] skip disabled: {filename} :: {reason}")


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
        print(f"[output] skip existing: {filename}")
        return

    try:
        save_callable(filepath)
        _record_output_status(summary, key, filename, "saved")
        _save_summary_checkpoint(output_folder, summary)
        print(f"[output] saved: {filename}")
    except Exception as exc:
        _record_output_status(summary, key, filename, "failed", str(exc))
        _save_summary_checkpoint(output_folder, summary)
        print(f"[output] failed: {filename} :: {exc}")
        if not continue_on_plot_error:
            raise


_SPLIT_MODULE_BINDINGS_ACTIVE = False


def _load_tenpy_backend_module() -> Any:
    """Load the local TeNPy U(1) Yao-Lee backend/template."""
    import importlib

    return importlib.import_module("tenpy_backend")


def _bind_split_module_implementations() -> None:
    """Import the canonical implementation modules behind this CLI."""
    global _SPLIT_MODULE_BINDINGS_ACTIVE
    if _SPLIT_MODULE_BINDINGS_ACTIVE:
        return

    try:
        import analysis as analysis_tools
        import ed_backend as ed_backend_impl
        import models as model_defs
        import plot_outputs
        import tenax_backend as tenax_backend_impl
    except Exception as exc:
        raise RuntimeError(f"Could not import the split implementation modules: {exc}") from exc

    bindings: Dict[str, Any] = {}
    for module, names in (
        (
            analysis_tools,
            (
                "get_tenax_api",
                "_get_tqdm",
                "_make_progress_bar",
                "_start_stage",
                "_end_stage",
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
            ed_backend_impl,
            (
                "estimate_sz_conserved_dimension",
                "build_sz_conserved_basis",
                "build_sparse_hamiltonian_sz_conserved",
                "run_sz_conserved_exact_spectrum",
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
            ),
        ),
    ):
        for name in names:
            if hasattr(module, name):
                bindings[name] = getattr(module, name)

    globals().update(bindings)
    _SPLIT_MODULE_BINDINGS_ACTIVE = True


def main() -> None:
    _bind_split_module_implementations()
    args = parse_command_line()
    args.ed_backend = _normalize_ed_backend(args.ed_backend)
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
    if args.run_phase_scan is None:
        args.run_phase_scan = bool(args.phase_diagram)
    if args.plot_phase_scan is None:
        args.plot_phase_scan = bool(args.run_phase_scan)
    if bool(args.phase_scan_only):
        args.run_phase_scan = True
    ensure_folder_exists(args.output_folder)
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
                "Bond energies, structure factors, and real-space patterns require correlation matrices, "
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
            "mode": str(args.phase_scan_mode),
            "quantum_methods": list(args.phase_scan_quantum_methods),
            "selected_outputs": list(args.phase_scan_methods),
            "note": "The phase_diagram switch ties phase-scan calculation and phase-diagram plotting together by default.",
        },
    }
    lattice_name = str(args.lattice).lower()
    circumference_x = bool(args.circumference_x)
    circumference_y = bool(args.circumference_y)
    args.symmetry_mode = _normalize_symmetry_mode(args.symmetry_mode)
    model_spec = build_model_spec(
        spin_rep=args.spin_rep,
        orbital_rep=args.orbital_rep,
        model_family=args.model_family,
        ising_axis=args.ising_axis,
    )
    # Normalize legacy alias "1" -> "0" in recorded parameters.
    args.orbital_rep = model_spec.orbital_rep
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
    try:
        validate_external_field_symmetry_compatibility(
            hamiltonian_external_field_terms,
            symmetry_mode=args.symmetry_mode,
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
    quspin_ed_settings = {
        "symmetry_reductions": symmetry_reduction_settings,
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
            "check_symmetries": bool(args.quspin_check_symmetries),
            "check_hermiticity": bool(args.quspin_check_hermiticity),
            "check_particle_conservation": bool(args.quspin_check_particle_conservation),
        }
        return symmetry_reduction_settings

    refresh_symmetry_reduction_settings(None)
    symmetry_preflight_report: Dict[str, Any] | None = None
    effective_symmetry_mode = str(args.symmetry_mode)

    def run_symmetry_preflight_for_geometry(geometry_obj: Any) -> Dict[str, Any]:
        if not bool(args.symmetry_precheck):
            disabled_effective = "none" if str(args.symmetry_mode) == "auto" else str(args.symmetry_mode)
            report = {
                "status": "disabled",
                "requested_mode": str(args.symmetry_mode),
                "requested_reductions": list(args.symmetry_reductions),
                "effective_mode_for_tenax": disabled_effective,
            }
            refresh_symmetry_reduction_settings(report)
            report["effective_reductions"] = {
                "sz": bool(args.use_sz_block),
                "tz": bool(args.use_tau_z_block),
                "z2": bool(args.use_z2_block),
            }
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
        report["effective_reductions"] = {
            "sz": bool(args.use_sz_block),
            "tz": bool(args.use_tau_z_block),
            "z2": bool(args.use_z2_block),
        }
        if show_progress:
            u1_info = report.get("u1", {}) if isinstance(report.get("u1"), dict) else {}
            u1_sz_info = report.get("u1_sz", {}) if isinstance(report.get("u1_sz"), dict) else {}
            u1_tz_info = report.get("u1_tz", {}) if isinstance(report.get("u1_tz"), dict) else {}
            z2_info = report.get("z2", {}) if isinstance(report.get("z2"), dict) else {}
            print(
                "[symmetry] precheck: "
                f"requested={list(args.symmetry_reductions)}, "
                f"U1(Sz,tau_z)={bool(u1_info.get('conserved_total_Sz_and_total_Tz', False))}, "
                f"U1_target_reachable={bool((u1_info.get('target_sector') or {}).get('reachable', False))}, "
                f"U1_Sz={bool(u1_sz_info.get('conserved_total_Sz', False))}, "
                f"U1_Tz={bool(u1_tz_info.get('conserved_total_Tz', False))}, "
                f"Z2={bool(z2_info.get('conserved_global_parity', False))}, "
                f"effective_reductions={report['effective_reductions']}, "
                f"recommended_tenax={recommended_mode}, "
                f"effective_tenax={effective_mode}"
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

    if args.backend == "tenpy" and lattice_name != "honeycomb":
        raise ValueError(
            "TeNPy backend currently supports only --lattice honeycomb. "
            "Use --backend tenax for square/triangular lattices."
        )
    tenpy_transverse_field = any(
        str(op_name) in ("Sx", "Sy")
        for _coefficient, op_name in list(hamiltonian_external_field_terms or [])
    )
    tenpy_allowed_symmetry_modes = ("auto", "u1_sz", "none") if tenpy_transverse_field else ("auto", "u1_sz")
    if args.backend == "tenpy" and args.symmetry_mode not in tenpy_allowed_symmetry_modes:
        raise ValueError(
            "The local TeNPy backend is the strict spin-U1 implementation and supports "
            "only --symmetry-mode u1_sz (or auto), except transverse Hamiltonian fields may "
            "use --symmetry-mode none. Use --backend tenax/--symmetry-mode none for dense "
            "non-symmetric experiments."
        )
    if args.backend == "tenpy":
        if not (
            model_spec.spin_rep == "1/2"
            and model_spec.orbital_rep == "1/2"
            and model_spec.model_family == "yao_lee"
            and model_spec.ising_axis == "z"
        ):
            raise ValueError(
                "TeNPy backend currently supports only the legacy default model "
                "(spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z). "
                "Use --backend tenax for other constructions."
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
        return labeled_output_filename(run_file_prefix, base_filename)

    def plot_title(base_title: str) -> str:
        return titled_for_run(base_title, run_plot_title_label)

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
            print(f"[output] skip disabled: {filename} :: plot_geometry is false")
            return
        if os.path.exists(filepath) and not overwrite_existing:
            geometry_plot_status = "skipped_exists"
            print(f"[output] skip existing: {filename}")
            return
        try:
            save_geometry_diagram(geometry_obj, filepath, lattice_name, title_label=run_plot_title_label)
            geometry_plot_status = "saved"
            print(f"[output] saved: {filename}")
        except Exception as exc:
            geometry_plot_status = "failed"
            geometry_plot_error = str(exc)
            print(f"[output] failed: {filename} :: {exc}")
            if not continue_on_plot_error:
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
    ) -> Dict[str, Any]:
        ed_backend_name = str(row.get("ed_backend", args.ed_backend)).strip().lower()
        if ed_backend_name == "ed":
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
                external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
                solver=args.ed_solver,
                sparse_tol=float(args.ed_sparse_tol),
                sparse_maxiter=(int(args.ed_sparse_maxiter) if int(args.ed_sparse_maxiter) > 0 else None),
                use_sz_block=use_sz_block,
                target_sz2=int(symmetry_reduction_settings.get("target_sz2", args.u1_target_sz2)),
                use_tau_z_block=use_tau_z_block,
                target_tz2=int(symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2)),
                use_z2_block=use_z2_block,
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
        elif bool(row.get("use_sz_conserved", False)):
            spectrum, vectors, basis_list, basis_map = run_sz_conserved_exact_spectrum(
                geometry=geometry_obj,
                alpha=float(alpha),
                beta=float(beta),
                coupling_j=args.coupling_j,
                eigenstate_count=max(3, min(int(args.ed_max_eigenstates), 8)),
                check_ground_state_degeneracy=False,
                external_field_terms=hamiltonian_external_field_terms,
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
                external_field_terms=hamiltonian_external_field_terms,
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
    ) -> Dict[str, Any]:
        yl_scan = _load_tenpy_backend_module()
        psi, _mpo, info = yl_scan.run_cylindrical_dmrg(
            geometry=geometry_obj,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=args.coupling_j,
            max_bond_dimension=args.max_bond_dimension,
            max_sweeps=args.max_sweeps,
            truncation_cutoff=args.truncation_cutoff,
            initial_state=None,
            compute_phase_observables=False,
            external_field_terms=hamiltonian_external_field_terms,
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
    ) -> Dict[str, Any]:
        yl_scan = _load_tenpy_backend_module()
        _rows, psi = yl_scan.run_alpha_scan_idmrg_with_adiabatic_state_passing(
            geometry=geometry_obj,
            alpha_values=[float(alpha)],
            beta=float(beta),
            coupling_j=args.coupling_j,
            max_bond_dimension=args.idmrg_max_bond_dimension,
            max_iterations=args.idmrg_max_iterations,
            truncation_cutoff=args.truncation_cutoff,
            initial_state=None,
            classifier_thresholds=phase_classifier_thresholds_from_args(args),
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

    def _recalculate_phase_representative_payload(
        geometry_obj: Any,
        mode_key: str,
        representative: Dict[str, Any],
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        alpha = float(representative["alpha"])
        beta = float(representative["beta"])
        print(
            "[phase] representative recalculation: "
            f"method={mode_key}, phase={representative['phase_label']}, alpha={alpha:.6g}, beta={beta:.6g}"
        )
        if mode_key == "classical_product":
            return _classical_representative_payload(geometry_obj, row, alpha, beta)
        if mode_key == "quantum_ed":
            return _quantum_ed_representative_payload(geometry_obj, row, alpha, beta)
        if mode_key == "tenpy_dmrg":
            return _tenpy_dmrg_representative_payload(geometry_obj, alpha, beta)
        if mode_key == "tenpy_idmrg":
            return _tenpy_idmrg_representative_payload(geometry_obj, alpha, beta)
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
        phase_token = _safe_filename_token(phase_label.lower())
        output_key_prefix = f"{mode_key}_{phase_token}_{int(representative['row_index'])}"
        row_bond_rows = row.get("bond_energies") if isinstance(row.get("bond_energies"), list) else None
        recalculation_supported = mode_key in ("classical_product", "quantum_ed", "tenpy_dmrg", "tenpy_idmrg")
        needs_recalculation = bool(
            recalculation_supported
            and (args.plot_real_space_patterns or args.plot_bond_energies)
        )
        payload: Dict[str, Any] = {}
        if needs_recalculation:
            try:
                payload = _recalculate_phase_representative_payload(
                    geometry_obj,
                    mode_key,
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
                lambda path, values=correlations["S"], ref_idx=int(reference_site_idx), rows_for_plot=bond_rows, title_phase=phase_label, title_alpha=alpha, title_beta=beta: save_phase_representative_pattern(
                    geometry_obj,
                    np.asarray(values, dtype=float),
                    ref_idx,
                    rows_for_plot,
                    path,
                    plot_title(
                        f"{mode_key} {title_phase} spin pattern + resolved bonds "
                        f"(alpha={title_alpha:.6g}, beta={title_beta:.6g})"
                    ),
                ),
                overwrite_existing,
                continue_on_plot_error,
            )
            outputs["representative_pattern_png"] = filename
            outputs["representative_pattern_note"] = (
                "Single combined plot: spin arrows use C_S[j]=<S_ref.S_j>; "
                "bond overlays use resolved spin/orbital channel energies when available."
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
    ) -> None:
        _record_phase_scan_plaquette_fluxes(summary_obj, phase_scan_data)
        phase_scan_filename = output_filename("phase_scan_summary.json")
        phase_scan_filepath = os.path.join(args.output_folder, phase_scan_filename)
        write_json(phase_scan_filepath, phase_scan_data)
        _record_output_status(summary_obj, "phase_scan_summary_json", phase_scan_filename, "saved")
        _save_summary_checkpoint(args.output_folder, summary_obj)
        print(f"[output] saved: {phase_scan_filename}")

        for mode_key, title in (
            ("classical_product", "Classical Product-State Phase Diagram"),
            ("quantum_ed", "Quantum ED Phase Diagram"),
            ("tenax_dmrg", "Tenax Finite-DMRG Phase Diagram"),
            ("tenpy_dmrg", "TeNPy Finite-DMRG Phase Diagram"),
            ("tenax_idmrg", "Tenax iDMRG Phase Diagram"),
            ("tenpy_idmrg", "TeNPy iDMRG Phase Diagram"),
        ):
            mode_data = phase_scan_data.get(mode_key)
            if not isinstance(mode_data, dict):
                continue
            base_name = (
                "classical_phase_diagram.png"
                if mode_key == "classical_product"
                else (
                    "quantum_phase_diagram.png"
                    if mode_key == "quantum_ed"
                    else f"{mode_key}_phase_diagram.png"
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
                lambda path, scan_rows=completed_rows, scan_title=title: save_phase_diagram_plot(
                    scan_rows,
                    path,
                    scan_title,
                    title_label=run_plot_title_label,
                ),
                overwrite_existing,
                continue_on_plot_error,
            )

        for mode_key, title_prefix in (
            ("tenpy_dmrg", "TeNPy finite-DMRG"),
            ("tenpy_idmrg", "TeNPy iDMRG"),
        ):
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

    def run_requested_phase_scan_for_geometry(geometry_obj: Any) -> Dict[str, Any]:
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
                    rows.append(_tenax_phase_scan_row(alpha, beta, alpha_index, beta_index, point_index))
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
            n_sites = int(geometry_obj.number_of_sites)
            if n_sites > int(args.max_dmrg_sites):
                output_key = "tenpy_dmrg" if backend_request == "tenpy" else "tenax_dmrg"
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
            if backend_request in ("tenax", "auto"):
                tenax_data = _run_tenax_dmrg_phase_scan()
                if backend_request == "tenax" or int(tenax_data.get("completed_points", 0)) > 0:
                    return "tenax_dmrg", tenax_data
                if show_progress:
                    print("[phase-scan] Tenax DMRG scan produced no completed points; trying TeNPy fallback.")
            if lattice_name != "honeycomb":
                raise ValueError("TeNPy DMRG phase scans currently support only honeycomb geometry.")
            if not (
                model_spec.spin_rep == "1/2"
                and model_spec.orbital_rep == "1/2"
                and model_spec.model_family == "yao_lee"
                and model_spec.ising_axis == "z"
            ):
                raise ValueError(
                    "TeNPy DMRG phase scans use the U(1) Sz-conserved YaoLeeSite and require "
                    "spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z."
                )
            yl_scan = _load_tenpy_backend_module()
            tenpy_data = yl_scan.run_alpha_beta_dmrg_observable_scan(
                geometry=geometry_obj,
                alpha_values=alpha_values,
                beta_values=beta_values,
                coupling_j=args.coupling_j,
                max_bond_dimension=args.max_bond_dimension,
                max_sweeps=args.max_sweeps,
                truncation_cutoff=args.truncation_cutoff,
                carry_state_between_betas=False,
                classifier_thresholds=classifier_thresholds,
                external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
            )
            tenpy_data["requested_backend"] = str(args.backend)
            tenpy_data["symmetry_reductions"] = symmetry_reduction_settings
            tenpy_data["symmetry_note"] = "TeNPy scan uses the fixed total-Sz U(1) YaoLeeSite backend."
            return "tenpy_dmrg", tenpy_data

        output: Dict[str, Any] = {
            "status": "completed",
            "mode": str(args.phase_scan_mode),
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
                "mode": str(args.phase_scan_mode),
                "quantum_methods": list(args.phase_scan_quantum_methods),
                "selected_outputs": methods,
                "ed_max_sites": int(args.phase_scan_ed_max_sites),
                "ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
                "ed_backend": str(args.ed_backend),
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
                "idmrg_max_bond_dimension": int(args.idmrg_max_bond_dimension),
                "idmrg_max_iterations": int(args.idmrg_max_iterations),
                "truncation_cutoff": float(args.truncation_cutoff),
                "tenpy_symmetry_mode": "u1_sz",
                "classifier_thresholds": classifier_thresholds,
            },
        }
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
            for key in ("quantum_ed", "classical_product"):
                if key in legacy_data:
                    output[key] = legacy_data[key]
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
            for key in ("quantum_ed", "classical_product"):
                if key in legacy_data:
                    output[key] = legacy_data[key]

        if "dmrg" in methods:
            dmrg_key, dmrg_data = _run_selected_dmrg_phase_scan()
            output[dmrg_key] = dmrg_data

        if "idmrg" in methods:
            if str(args.backend).strip().lower() == "tenax":
                output["tenax_idmrg"] = {
                    "status": "skipped",
                    "backend": "tenax",
                    "requested_backend": str(args.backend),
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(len(alpha_values) * len(beta_values)),
                    "reason": (
                        "Tenax iDMRG phase-scan classification is not implemented yet; "
                        "the single-point iDMRG workflow still uses Tenax when the finite DMRG backend is Tenax."
                    ),
                    "symmetry_reductions": symmetry_reduction_settings,
                    "symmetry_mode": str(effective_symmetry_mode),
                }
            else:
                if lattice_name != "honeycomb":
                    raise ValueError("TeNPy iDMRG phase scans currently support only honeycomb geometry.")
                if not (
                    model_spec.spin_rep == "1/2"
                    and model_spec.orbital_rep == "1/2"
                    and model_spec.model_family == "yao_lee"
                    and model_spec.ising_axis == "z"
                ):
                    raise ValueError(
                        "TeNPy iDMRG phase scans use the U(1) Sz-conserved YaoLeeSite and require "
                        "spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z."
                    )
                yl_scan = _load_tenpy_backend_module()
                output["tenpy_idmrg"] = yl_scan.run_alpha_beta_idmrg_observable_scan(
                    geometry=geometry_obj,
                    alpha_values=alpha_values,
                    beta_values=beta_values,
                    coupling_j=args.coupling_j,
                    max_bond_dimension=args.idmrg_max_bond_dimension,
                    max_iterations=args.idmrg_max_iterations,
                    truncation_cutoff=args.truncation_cutoff,
                    carry_state_between_betas=False,
                    classifier_thresholds=classifier_thresholds,
                    external_field_terms=hamiltonian_external_field_terms,
                    show_progress=show_progress,
                )
                output["tenpy_idmrg"]["requested_backend"] = str(args.backend)
                output["tenpy_idmrg"]["symmetry_reductions"] = symmetry_reduction_settings
                output["tenpy_idmrg"]["symmetry_note"] = "TeNPy scan uses the fixed total-Sz U(1) YaoLeeSite backend."
        child_statuses = [
            value.get("status")
            for value in output.values()
            if isinstance(value, dict) and "status" in value
        ]
        if any(status in ("failed", "completed_with_warnings") for status in child_statuses):
            output["status"] = "completed_with_warnings"
        return output

    if args.phase_scan_only:
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
                    "mode": str(args.phase_scan_mode),
                    "quantum_methods": list(args.phase_scan_quantum_methods),
                    "selected_outputs": list(args.phase_scan_methods),
                    "alpha_points": int(args.phase_scan_alpha_points),
                    "beta_points": int(args.phase_scan_beta_points),
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
                        "Phase-scan mode chooses quantum, classical, or both. Quantum methods choose ED, "
                        "finite-DMRG, iDMRG, or all quantum outputs."
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
            phase_scan_data = run_requested_phase_scan_for_geometry(geometry)
            scan_summary["phase_scan"] = phase_scan_data
            scan_summary["stages"]["phase_scan"] = "completed"
            save_phase_scan_outputs(scan_summary, phase_scan_data, geometry)
        except Exception as exc:
            scan_summary["phase_scan"] = {"status": "failed", "error": str(exc)}
            scan_summary["stages"]["phase_scan"] = "failed"
            _save_summary_checkpoint(args.output_folder, scan_summary)
            if not continue_on_plot_error:
                raise
        failed_outputs = [
            key
            for key, item in scan_summary.get("output_status", {}).items()
            if isinstance(item, dict) and item.get("status") == "failed"
        ]
        scan_summary["run_status"] = (
            "completed_with_warnings"
            if failed_outputs
            or (
                isinstance(scan_summary.get("phase_scan"), dict)
                and scan_summary["phase_scan"].get("status") in ("failed", "completed_with_warnings")
            )
            else "completed"
        )
        _save_summary_checkpoint(args.output_folder, scan_summary)
        print(
            "[run] phase-scan finished: "
            f"status={scan_summary['run_status']}, summary={os.path.join(args.output_folder, run_summary_filename)}"
        )
        return

    # Try Tenax first unless user forces tenpy.
    if args.backend in ("auto", "tenax"):
        try:
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
        except Exception as tenax_exc:
            if args.backend == "tenax":
                raise
            if effective_symmetry_mode != "none" and not bool(args.symmetry_allow_dense_fallback):
                raise RuntimeError(
                    f"Tenax failed while using symmetry_mode={effective_symmetry_mode}, and dense fallback is disabled. "
                    f"Original Tenax error: {tenax_exc}"
                ) from tenax_exc
            if effective_symmetry_mode != "none":
                backend_warning = (
                    f"Tenax failed after using symmetry_mode={effective_symmetry_mode}; "
                    "continuing to dense TeNPy fallback if compatible. "
                    f"Original Tenax error: {tenax_exc}"
                )
                if show_progress:
                    print(f"[symmetry] {backend_warning}")
            if lattice_name != "honeycomb":
                raise RuntimeError(
                    f"Tenax backend failed on lattice='{lattice_name}', and TeNPy fallback only supports honeycomb. "
                    f"Original Tenax error: {tenax_exc}"
                ) from tenax_exc
            if show_progress:
                print(f"[backend] Tenax failed; switching to TeNPy fallback. Reason: {tenax_exc}")
            if backend_warning is None:
                backend_warning = f"Tenax backend failed, fallback to TeNPy: {tenax_exc}"

    # TeNPy path via the local U(1)-symmetric Yao-Lee template.
    if backend_used is None:
        tenpy_path_label = "TeNPy backend" if args.backend == "tenpy" else "TeNPy fallback"
        if lattice_name != "honeycomb":
            raise RuntimeError(
                f"{tenpy_path_label} does not support lattice='{lattice_name}'. "
                "Only honeycomb is supported in tenpy_backend.py."
            )
        if not (
            model_spec.spin_rep == "1/2"
            and model_spec.orbital_rep == "1/2"
            and model_spec.model_family == "yao_lee"
            and model_spec.ising_axis == "z"
        ):
            raise RuntimeError(
                f"{tenpy_path_label} only supports the legacy default model "
                "(spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z)."
            )
        yl = _load_tenpy_backend_module()
        stage_start = _start_stage(f"{tenpy_path_label} DMRG", show_progress)
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
            random_seed=args.seed,
            product_state_style=args.initial_state,
            external_field_terms=hamiltonian_external_field_terms,
            show_progress=show_progress,
        )
        dmrg_energy = float(dmrg_info["E"])
        dmrg_state_obj = psi
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
        backend_used = "tenpy" if args.backend == "tenpy" else "tenpy_fallback"
        _end_stage(f"{tenpy_path_label} DMRG", stage_start, show_progress)

    try:
        if calculate_entanglement and dmrg_state_obj is not None:
            if backend_used == "tenax":
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

    summary: Dict[str, Any] = {
        "model_name": f"{lattice_label} spin-orbital model ({model_label})",
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
                "ed": {
                    "backend": str(args.ed_backend),
                    "max_sites": int(args.max_ed_sites),
                    "max_hilbert_dimension": int(args.max_ed_hilbert_dim),
                    "max_eigenstates": int(args.ed_max_eigenstates),
                    "solver": str(args.ed_solver),
                    "use_sz_conserved": bool(args.use_sz_conserved),
                    "symmetry_reductions": symmetry_reduction_settings,
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
                        "QuSpin uses the shared SYMMETRY_REDUCTIONS and U1/Z2 target-sector controls; "
                        "backend-specific flags here are construction checks only."
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
                    "bulk_kind": str(args.idmrg_bulk_kind),
                    "max_local_dim": int(args.idmrg_max_local_dim),
                    "note": "iDMRG chi is intentionally independent of finite-DMRG chi to avoid workstation OOM.",
                },
            },
            "bond_terms": {
                "yao_lee": (
                    "For each gamma bond: J[(1+beta) S_i.S_j + "
                    "(1-beta) T_gamma_i T_gamma_j + alpha (S_i.S_j)(T_gamma_i T_gamma_j)]. "
                    "The spin part is expanded as Sp/Sm/Sz terms for total-Sz U(1)."
                ),
                "ising_like": (
                    "For each bond, the chosen ising_axis replaces gamma in the same "
                    "S/T/ST channels."
                ),
                "orbital_rep_0": (
                    "The orbital Hilbert space is removed; yao_lee falls back to "
                    "spin-only Ising-like S_axis_i S_axis_j terms."
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
                        "Full bond-dependent Yao-Lee x/y channels contain single-axis flip terms "
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
                        "Use QuSpin basis blocks from the same shared symmetry reductions, "
                        "only when the precheck verifies the generated Hamiltonian and target sector."
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
                "spin_only_ising_like_fallback"
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
            "entanglement": entropy_profiles.get("DMRG"),
            "structure_factors": dmrg_structure_factor_rows,
            "real_space_patterns": dmrg_real_space_patterns,
            "uniform_observables": dmrg_uniform_observables,
        },
        "stages": {
            "dmrg": "completed",
            "dmrg_plots": "running",
            "idmrg": "pending" if args.run_idmrg else "not_requested",
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
    if backend_warning:
        summary["backend_warning"] = backend_warning
    if entanglement_warning:
        summary["entanglement_warning"] = entanglement_warning
    dmrg_all_plaquette_fluxes = _record_all_plaquette_fluxes(summary, "dmrg", dmrg_info)
    if dmrg_all_plaquette_fluxes:
        summary["dmrg"]["all_plaquette_fluxes"] = dmrg_all_plaquette_fluxes
    _save_summary_checkpoint(args.output_folder, summary)

    # Save DMRG plots immediately, one by one.
    _record_output_status(
        summary,
        "geometry_diagram_png",
        output_filename("geometry_diagram.png"),
        geometry_plot_status,
        geometry_plot_error,
    )
    _save_summary_checkpoint(args.output_folder, summary)
    if bool(args.plot_bond_energies) and calculate_bond_energies and len(dmrg_bond_rows) > 0:
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_bond_energy_diagram_png",
            output_filename("dmrg_bond_energy_diagram.png"),
            lambda path: save_bond_energy_diagram(geometry, dmrg_bond_rows, path, plot_title("DMRG Bond-Energy Diagram")),
            overwrite_existing,
            continue_on_plot_error,
        )
    else:
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_bond_energy_diagram_png",
            output_filename("dmrg_bond_energy_diagram.png"),
            "plot_bond_energies or calculate_bond_energies is false, or no DMRG bond rows were computed",
        )
    if bool(args.plot_structure_factors) and calculate_structure_factors and len(dmrg_structure_factor_rows) > 0:
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_structure_factors_png",
            output_filename("dmrg_structure_factors.png"),
            lambda path: save_structure_factor_plot(dmrg_structure_factor_rows, path, plot_title("DMRG Structure Factors")),
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
            lambda path: save_scalar_correlation_heatmaps(dmrg_scalar_correlations, path, f"DMRG | {run_plot_title_label}"),
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
    if (
        bool(args.plot_real_space_patterns)
        and calculate_real_space_patterns
        and isinstance(dmrg_pattern_correlations, dict)
        and dmrg_reference_site_idx is not None
    ):
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_spin_real_space_pattern_png",
            output_filename("dmrg_spin_real_space_pattern.png"),
            lambda path: save_real_space_pattern_diagram(
                geometry,
                dmrg_pattern_correlations["S"],
                int(dmrg_reference_site_idx),
                path,
                plot_title("DMRG Reference-Site Spin Pattern"),
                "C_S[j] = <S_ref . S_j>",
            ),
            overwrite_existing,
            continue_on_plot_error,
        )
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_orbital_real_space_pattern_png",
            output_filename("dmrg_orbital_real_space_pattern.png"),
            lambda path: save_real_space_pattern_diagram(
                geometry,
                dmrg_pattern_correlations["T"],
                int(dmrg_reference_site_idx),
                path,
                plot_title("DMRG Reference-Site Orbital Pattern"),
                "C_T[j] = <T_ref . T_j>",
            ),
            overwrite_existing,
            continue_on_plot_error,
        )
        if "ST" in dmrg_pattern_correlations:
            _save_plot_step(
                summary,
                args.output_folder,
                "dmrg_mixed_spin_orbital_real_space_pattern_png",
                output_filename("dmrg_mixed_spin_orbital_real_space_pattern.png"),
                lambda path: save_real_space_pattern_diagram(
                    geometry,
                    dmrg_pattern_correlations["ST"],
                    int(dmrg_reference_site_idx),
                    path,
                    plot_title("DMRG Reference-Site Mixed Spin-Orbital Pattern"),
                    "C_ST[j] mixed spin-orbital scalar",
                ),
                overwrite_existing,
                continue_on_plot_error,
            )
    else:
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_spin_real_space_pattern_png",
            output_filename("dmrg_spin_real_space_pattern.png"),
            "plot_real_space_patterns or calculate_real_space_patterns is false, or no DMRG pattern correlations were computed",
        )
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_orbital_real_space_pattern_png",
            output_filename("dmrg_orbital_real_space_pattern.png"),
            "plot_real_space_patterns or calculate_real_space_patterns is false, or no DMRG pattern correlations were computed",
        )
        _skip_plot_step(
            summary,
            args.output_folder,
            "dmrg_mixed_spin_orbital_real_space_pattern_png",
            output_filename("dmrg_mixed_spin_orbital_real_space_pattern.png"),
            "plot_real_space_patterns or calculate_real_space_patterns is false, or no DMRG mixed pattern correlations were computed",
        )
    save_flux_crystal_output(
        summary,
        geometry,
        dmrg_info,
        "flux_crystal_pattern_png",
        "flux_crystal_pattern.png",
        "DMRG Plaquette Flux Crystal Pattern",
    )
    summary["stages"]["dmrg_plots"] = "completed"
    _save_summary_checkpoint(args.output_folder, summary)

    # Optional iDMRG workflow (runs after finite DMRG outputs are saved).
    if args.run_idmrg:
        summary["stages"]["idmrg"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        if backend_used == "tenax" and tenax_mpo is None:
            summary["idmrg"] = {
                "status": "failed",
                "error": "Tenax MPO object unavailable after DMRG.",
            }
            summary["stages"]["idmrg"] = "failed"
            _save_summary_checkpoint(args.output_folder, summary)
        else:
            try:
                if backend_used == "tenax":
                    idmrg_info = run_tenax_idmrg_x_from_finite_mpo(
                        mpo=tenax_mpo,
                        model_spec=model_spec,
                        max_bond_dimension=args.idmrg_max_bond_dimension,
                        max_iterations=args.idmrg_max_iterations,
                        bulk_kind=args.idmrg_bulk_kind,
                        max_local_dim=args.idmrg_max_local_dim,
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
                            random_seed=args.seed,
                            product_state_style=args.initial_state,
                            compute_entanglement=calculate_entanglement,
                            external_field_terms=hamiltonian_external_field_terms,
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
                        random_seed=args.seed,
                        product_state_style=args.initial_state,
                        compute_entanglement=calculate_entanglement,
                        external_field_terms=hamiltonian_external_field_terms,
                        show_progress=show_progress,
                    )
                summary["idmrg"] = idmrg_info
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
    ed_backend_name = str(args.ed_backend)
    quspin_ed_requested = ed_backend_name == "quspin"
    requested_sz_block = bool(symmetry_reduction_settings.get("use_sz_block", False))
    requested_tau_z_block = bool(symmetry_reduction_settings.get("use_tau_z_block", False))
    requested_z2_block = bool(symmetry_reduction_settings.get("use_z2_block", False))
    requested_target_sz2 = int(symmetry_reduction_settings.get("target_sz2", args.u1_target_sz2))
    requested_target_tz2 = int(symmetry_reduction_settings.get("target_tz2", args.u1_target_tz2))
    hamiltonian_field_ops = {
        str(op_name)
        for coefficient, op_name in list(hamiltonian_external_field_terms or [])
        if abs(float(coefficient)) > 1e-14
    }
    quspin_use_sz_block = bool(requested_sz_block)
    quspin_sz_reason = None
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
    requested_translation_block = bool(args.use_translation_x_block or args.use_translation_y_block)
    if quspin_ed_requested and requested_translation_block:
        if not quspin_package_available:
            reason = "QuSpin package is not installed, so translation blocks cannot be checked."
            quspin_translation_x_reason = reason if bool(args.use_translation_x_block) else None
            quspin_translation_y_reason = reason if bool(args.use_translation_y_block) else None
        else:
            try:
                import quspin_backend as quspin_validation_backend

                translation_support = quspin_validation_backend.quspin_translation_block_support(geometry)
                x_support = translation_support.get("x", {})
                y_support = translation_support.get("y", {})
                quspin_use_translation_x_block = bool(
                    args.use_translation_x_block and x_support.get("supported", False)
                )
                quspin_use_translation_y_block = bool(
                    args.use_translation_y_block and y_support.get("supported", False)
                )
                quspin_translation_x_reason = x_support.get("reason") if bool(args.use_translation_x_block) else None
                quspin_translation_y_reason = y_support.get("reason") if bool(args.use_translation_y_block) else None
            except Exception as exc:
                reason = str(exc)
                quspin_translation_x_reason = reason if bool(args.use_translation_x_block) else None
                quspin_translation_y_reason = reason if bool(args.use_translation_y_block) else None
    quspin_use_translation_block = bool(quspin_use_translation_x_block or quspin_use_translation_y_block)
    quspin_translation_reason = {
        "x": quspin_translation_x_reason,
        "y": quspin_translation_y_reason,
    }
    quspin_use_reflection_block = False
    quspin_reflection_reason = None
    if quspin_ed_requested and (bool(args.use_reflection_block) or int(args.reflection_block) != 0):
        quspin_reflection_reason = (
            "QuSpin reflection/C3 blocks are not applied for the bond-directional Yao-Lee Hamiltonian; "
            "they can permute x/y/z bond types unless a gauge map is implemented."
        )
    quspin_use_z2_block = bool(
        requested_z2_block
        and quspin_use_sz_block
        and requested_target_sz2 == 0
        and not bool(hamiltonian_field_ops.intersection({"Sx", "Sy", "Sz"}))
    )
    quspin_z2_reason = None
    if quspin_ed_requested and requested_z2_block and not quspin_use_z2_block:
        if hamiltonian_field_ops.intersection({"Sx", "Sy", "Sz"}):
            quspin_z2_reason = "External spin-field Hamiltonian terms break the QuSpin spin-flip Z2 block."
        else:
            quspin_z2_reason = "QuSpin spin-flip Z2 requires the total Sz=0 spin block."
    spin_orbital_block_dim = _spin_orbital_symmetry_reduced_dimension(
        int(geometry.number_of_sites),
        quspin_use_sz_block,
        requested_target_sz2,
        requested_tau_z_block,
        requested_target_tz2,
    )
    quspin_structurally_available = (
        quspin_ed_requested
        and quspin_package_available
        and model_spec.spin_rep == "1/2"
        and model_spec.orbital_rep == "1/2"
        and model_spec.model_family == "yao_lee"
        and model_spec.ising_axis == "z"
        and int(spin_orbital_block_dim) > 0
        and not (requested_tau_z_block and (quspin_use_z2_block or quspin_use_translation_block))
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
    elif quspin_ed_requested and requested_tau_z_block and (quspin_use_z2_block or quspin_use_translation_block):
        quspin_ed_reason = "QuSpin does not combine tau_z with Z2 or 2D translation blocks for this model."
    elif quspin_ed_requested and quspin_basis_build_reason is not None:
        quspin_ed_reason = quspin_basis_build_reason
    sz_conserved_requested = bool(requested_sz_block)
    sz_conserved_available = (
        not quspin_ed_requested
        and
        sz_conserved_requested
        and model_spec.spin_rep == "1/2"
        and model_spec.orbital_rep == "1/2"
        and _sector_dimension_for_spin_half(int(geometry.number_of_sites), requested_target_sz2) > 0
    )
    sz_conserved_reason = None
    if sz_conserved_requested and model_spec.orbital_rep == "0":
        sz_conserved_reason = "orbital_rep=0 is spin-only; using the legacy full spin ED path."
    elif sz_conserved_requested and model_spec.spin_rep != "1/2":
        sz_conserved_reason = "Sz-conserved bitwise ED currently supports spin_rep=1/2 only."
    elif sz_conserved_requested and model_spec.orbital_rep != "1/2":
        sz_conserved_reason = "Sz-conserved bitwise ED currently supports orbital_rep=1/2 only."
    elif sz_conserved_requested and _sector_dimension_for_spin_half(int(geometry.number_of_sites), requested_target_sz2) <= 0:
        sz_conserved_reason = "The requested total Sz sector is unreachable for this number of spin-1/2 sites."
    if quspin_ed_requested:
        hilbert_dim = int(
            quspin_actual_hilbert_dim
            if quspin_actual_hilbert_dim is not None
            else spin_orbital_block_dim
        )
        ed_basis_type = (
            "quspin_tensor_"
            f"spin_{'u1_block' if quspin_use_sz_block else 'full'}_"
            f"orbital_{'u1_block' if requested_tau_z_block else 'full'}"
        )
    elif sz_conserved_available:
        hilbert_dim = int(estimate_sz_conserved_dimension(int(geometry.number_of_sites), target_sz2=requested_target_sz2))
        ed_basis_type = "bitwise_spin_orbital_total_sz_block"
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
    ed_unsupported_reductions = [
        reduction
        for reduction in ("sz", "tz", "z2")
        if reduction in set(args.symmetry_reductions) and reduction not in ed_applied_reductions
    ]
    ed_eligibility: Dict[str, Any] = {
        "requested": bool(args.run_ed),
        "ed_backend": ed_backend_name,
        "allowed": False,
        "forbidden": True,
        "number_of_sites": int(geometry.number_of_sites),
        "local_dimension": int(local_dim),
        "hilbert_dimension": int(hilbert_dim),
        "effective_hilbert_dimension": int(hilbert_dim),
        "pre_quspin_hilbert_dimension_estimate": int(spin_orbital_block_dim) if quspin_ed_requested else None,
        "actual_quspin_hilbert_dimension": (
            int(quspin_actual_hilbert_dim) if quspin_actual_hilbert_dim is not None else None
        ),
        "full_hilbert_dimension": int(full_hilbert_dim),
        "basis_type": ed_basis_type,
        "symmetry_reductions": symmetry_reduction_settings,
        "applied_reductions": ed_applied_reductions,
        "unsupported_or_unapplied_reductions": ed_unsupported_reductions,
        "use_sz_conserved_requested": bool(sz_conserved_requested),
        "use_sz_conserved": "sz" in ed_applied_reductions,
        "use_tau_z_conserved": "tz" in ed_applied_reductions,
        "use_z2_conserved": "z2" in ed_applied_reductions,
        "standard_sz_conserved": bool(sz_conserved_available),
        "quspin_available": bool(quspin_ed_available),
        "quspin_package_available": bool(quspin_package_available),
        "quspin_reason": quspin_ed_reason,
        "quspin_requested_sz_block": bool(requested_sz_block),
        "quspin_use_sz_block": bool(quspin_use_sz_block),
        "quspin_sz_reason": quspin_sz_reason,
        "quspin_requested_translation_block": bool(requested_translation_block),
        "quspin_requested_translation_x_block": bool(args.use_translation_x_block),
        "quspin_requested_translation_y_block": bool(args.use_translation_y_block),
        "quspin_use_translation_block": bool(quspin_use_translation_block),
        "quspin_use_translation_x_block": bool(quspin_use_translation_x_block),
        "quspin_use_translation_y_block": bool(quspin_use_translation_y_block),
        "quspin_momentum_x_block": int(args.momentum_x_block),
        "quspin_momentum_y_block": int(args.momentum_y_block),
        "quspin_translation_reason": quspin_translation_reason,
        "quspin_translation_x_reason": quspin_translation_x_reason,
        "quspin_translation_y_reason": quspin_translation_y_reason,
        "quspin_requested_reflection_block": bool(args.use_reflection_block),
        "quspin_use_reflection_block": bool(quspin_use_reflection_block),
        "quspin_reflection_reason": quspin_reflection_reason,
        "quspin_use_z2_block": bool(quspin_use_z2_block),
        "quspin_z2_reason": quspin_z2_reason,
        "sz_conserved_reason": sz_conserved_reason,
        "max_sites": int(args.max_ed_sites),
        "max_hilbert_dimension": int(args.max_ed_hilbert_dim),
        "solver_requested": str(args.ed_solver),
        "solver": (
            "quspin_eigsh"
            if quspin_ed_requested
            else ("sz_conserved_sparse" if sz_conserved_available else str(args.ed_solver))
        ),
        "max_eigenstates": (
            int(args.ed_max_eigenstates)
            if quspin_ed_requested
            else (
                int(SZ_CONSERVED_ED_EIGENSTATES)
                if sz_conserved_available
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

                ed_all_plaquette_fluxes = _all_plaquette_fluxes_from_payload(ed_plaquette_flux or {})
                summary["ed"] = {
                    "status": "completed",
                    "ed_backend": ed_backend_name,
                    "eligibility": ed_eligibility,
                    "basis_type": ed_basis_type,
                    "symmetry_reductions": symmetry_reduction_settings,
                    "applied_reductions": ed_applied_reductions,
                    "unsupported_or_unapplied_reductions": ed_unsupported_reductions,
                    "use_sz_conserved": "sz" in ed_applied_reductions,
                    "use_tau_z_conserved": "tz" in ed_applied_reductions,
                    "use_z2_conserved": "z2" in ed_applied_reductions,
                    "use_translation_x_conserved": "translation_x" in ed_applied_reductions,
                    "use_translation_y_conserved": "translation_y" in ed_applied_reductions,
                    "hamiltonian_formula": (
                        ed_spectrum.get("formula")
                        if isinstance(ed_spectrum, dict) and ed_spectrum.get("formula") is not None
                        else "legacy full tensor-product ED formula from models.model_terms_for_bond"
                    ),
                    "ground_state_energy": ed_energy,
                    "energy_per_site": ed_energy / geometry.number_of_sites,
                    "absolute_energy_difference_dmrg_minus_ed": abs(dmrg_energy - ed_energy),
                    "structure_factors": ed_structure_factor_rows,
                    "real_space_patterns": ed_real_space_patterns,
                    "plaquette_flux": ed_plaquette_flux,
                    "all_plaquette_fluxes": ed_all_plaquette_fluxes,
                }
                if ed_all_plaquette_fluxes:
                    _record_all_plaquette_fluxes(summary, "ed", ed_plaquette_flux or {})
                if ed_spectrum is not None:
                    summary["ed"]["spectrum"] = ed_spectrum
                    if ed_spectrum.get("hilbert_dimension") is not None:
                        actual_dim = int(ed_spectrum["hilbert_dimension"])
                        summary["ed"]["hilbert_dimension"] = actual_dim
                        summary["ed"]["effective_hilbert_dimension"] = actual_dim
                        if isinstance(summary["ed"].get("eligibility"), dict):
                            summary["ed"]["eligibility"]["actual_hilbert_dimension"] = actual_dim
                            summary["ed"]["eligibility"]["effective_hilbert_dimension_after_quspin"] = actual_dim
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

                if bool(args.plot_bond_energies) and calculate_bond_energies and len(ed_bond_rows) > 0:
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "ed_bond_energy_diagram_png",
                        output_filename("ed_bond_energy_diagram.png"),
                        lambda path: save_bond_energy_diagram(geometry, ed_bond_rows, path, plot_title("ED Bond-Energy Diagram")),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                else:
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "ed_bond_energy_diagram_png",
                        output_filename("ed_bond_energy_diagram.png"),
                        "plot_bond_energies or calculate_bond_energies is false, or no ED bond rows were computed",
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
                if (
                    bool(args.plot_real_space_patterns)
                    and calculate_real_space_patterns
                    and isinstance(ed_pattern_correlations, dict)
                    and ed_reference_site_idx is not None
                ):
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "ed_spin_real_space_pattern_png",
                        output_filename("ed_spin_real_space_pattern.png"),
                        lambda path: save_real_space_pattern_diagram(
                            geometry,
                            ed_pattern_correlations["S"],
                            int(ed_reference_site_idx),
                            path,
                            plot_title("ED Reference-Site Spin Pattern"),
                            "C_S[j] = <S_ref . S_j>",
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                    _save_plot_step(
                        summary,
                        args.output_folder,
                        "ed_orbital_real_space_pattern_png",
                        output_filename("ed_orbital_real_space_pattern.png"),
                        lambda path: save_real_space_pattern_diagram(
                            geometry,
                            ed_pattern_correlations["T"],
                            int(ed_reference_site_idx),
                            path,
                            plot_title("ED Reference-Site Orbital Pattern"),
                            "C_T[j] = <T_ref . T_j>",
                        ),
                        overwrite_existing,
                        continue_on_plot_error,
                    )
                    if "ST" in ed_pattern_correlations:
                        _save_plot_step(
                            summary,
                            args.output_folder,
                            "ed_mixed_spin_orbital_real_space_pattern_png",
                            output_filename("ed_mixed_spin_orbital_real_space_pattern.png"),
                            lambda path: save_real_space_pattern_diagram(
                                geometry,
                                ed_pattern_correlations["ST"],
                                int(ed_reference_site_idx),
                                path,
                                plot_title("ED Reference-Site Mixed Spin-Orbital Pattern"),
                                "C_ST[j] mixed spin-orbital scalar",
                            ),
                            overwrite_existing,
                            continue_on_plot_error,
                        )
                else:
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "ed_spin_real_space_pattern_png",
                        output_filename("ed_spin_real_space_pattern.png"),
                        "plot_real_space_patterns or calculate_real_space_patterns is false, or no ED pattern correlations were computed",
                    )
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "ed_orbital_real_space_pattern_png",
                        output_filename("ed_orbital_real_space_pattern.png"),
                        "plot_real_space_patterns or calculate_real_space_patterns is false, or no ED pattern correlations were computed",
                    )
                    _skip_plot_step(
                        summary,
                        args.output_folder,
                        "ed_mixed_spin_orbital_real_space_pattern_png",
                        output_filename("ed_mixed_spin_orbital_real_space_pattern.png"),
                        "plot_real_space_patterns or calculate_real_space_patterns is false, or no ED mixed pattern correlations were computed",
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
                    "DMRG": dmrg_structure_factor_rows,
                    "ED": ed_structure_factor_rows,
                }
                idmrg_structure_rows = (
                    summary["idmrg"].get("structure_factors")
                    if isinstance(summary.get("idmrg"), dict)
                    else None
                )
                if isinstance(idmrg_structure_rows, list) and len(idmrg_structure_rows) > 0:
                    method_structure_comparison["iDMRG-x"] = idmrg_structure_rows
                if bool(args.plot_structure_factors) and all(len(rows) > 0 for rows in method_structure_comparison.values()):
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
                        "plot_structure_factors is false or one compared method has no structure rows",
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
        elif backend_used != "tenax":
            summary["stages"]["dmrg_excited_state_search"] = "skipped"
            dmrg_excited_search = _missing_dmrg_excited_search(
                "finite-DMRG penalty excited-state search requires the Tenax MPS/MPO backend"
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
        "DMRG": float(summary["dmrg"]["energy_per_site"]),
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
    if bool(args.plot_energy_comparison):
        _save_plot_step(
            summary,
            args.output_folder,
            "dmrg_vs_ed_vs_idmrg_energy_png",
            output_filename("dmrg_vs_ed_vs_idmrg_energy.png"),
            lambda path: save_multi_method_energy_comparison(
                method_to_energy=method_energy_comparison,
                filepath=path,
                title="Finite DMRG vs ED vs iDMRG-x Energy Per Site",
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
        "DMRG": {
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
                "Finite-DMRG first-excited energy and gap are reported only when "
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
            phase_scan_data = run_requested_phase_scan_for_geometry(geometry)
            summary["phase_scan"] = phase_scan_data
            summary["stages"]["phase_scan"] = "completed"
            save_phase_scan_outputs(summary, phase_scan_data, geometry)
        except Exception as exc:
            error_text = str(exc) or exc.__class__.__name__
            summary["phase_scan"] = {"status": "failed", "error": error_text}
            summary["stages"]["phase_scan"] = "failed"
            if not continue_on_plot_error:
                raise
        _save_summary_checkpoint(args.output_folder, summary)

    failed_outputs = [
        key
        for key, item in summary.get("output_status", {}).items()
        if isinstance(item, dict) and item.get("status") == "failed"
    ]
    if (
        failed_outputs
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
    _save_summary_checkpoint(args.output_folder, summary)
    print(
        "[run] finished: "
        f"status={summary['run_status']}, summary={os.path.join(args.output_folder, run_summary_filename)}"
    )


if __name__ == "__main__":
    main()
