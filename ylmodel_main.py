#!/usr/bin/env python3
"""
CLI orchestration for Yao-Lee model benchmarking with Tenax DMRG and optional ED.

The canonical work is split across sibling modules:
- analysis.py: Tenax loading, progress, timing, entropy, and scan analysis.
- models.py: model specs, operators, geometry, ED, correlations.
- backend.py: Tenax MPO/DMRG/iDMRG execution.
- plot_outputs.py: PNG output helpers.

This file keeps settings, CLI, consistency checks, and run orchestration.
main() binds the split modules before running so fixes stay shared by owner.
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Callable, Dict, List

import numpy as np


# ----------------------------------------------------------------------
# Configuration (edit this top block for normal runs)
# ----------------------------------------------------------------------

# Available choices.
LATTICE_OPTIONS = ("honeycomb", "square", "triangular")
SPIN_ONLY_MODEL_FAMILIES = ("heisenberg", "xy", "xxz", "xyz")
MODEL_FAMILY_OPTIONS = ("yao_lee", "ising_like") + SPIN_ONLY_MODEL_FAMILIES
SPIN_REP_OPTIONS = ("1/2", "3/2")
ORBITAL_REP_OPTIONS = ("0", "1/2")
AXIS_OPTIONS = ("x", "y", "z")
INITIAL_STATE_OPTIONS = ("alternating", "random")
SYMMETRY_MODE_OPTIONS = ("none", "u1", "z2")
U1_CHARGE_TZ_STRIDE = 4096
Z2_PARITY_OPTIONS = (0, 1)
IDMRG_BULK_KIND_OPTIONS = ("auto", "pair", "single")
BACKEND_OPTIONS = ("auto", "tenax", "tenpy")
EXTERNAL_FIELD_TREATMENT_OPTIONS = ("off", "perturbation", "hamiltonian")
EXTERNAL_FIELD_AXIS_OPTIONS = ("custom", "111")
ENTROPY_ORDERS = (1, 2, 3, 4)

# Resource profiles.
# Edit ACTIVE_RESOURCE_PROFILE to switch all geometry/DMRG/ED/iDMRG defaults
# together. Keep larger aragorn/beehive choices on the command line or in a new
# profile so local/shared-machine runs stay polite by default.
LOCAL_LAPTOP_SETTINGS = {
    "geometry": {
        "length_x": 2,
        "circumference_y": 2,
        "periodic_around_cylinder": True,
        "lattice_type": "honeycomb",
    },
    "finite_dmrg": {
        "max_sites": 8,
        "max_bond_dimension": 64,
        "max_sweeps": 10,
    },
    "ed": {
        "run": True,
        "max_sites": 8,
        "max_hilbert_dim": 250_000,
    },
    "finite_temperature_ed": {
        "run": True,
        "max_sites": 8,
        "max_hilbert_dim": 100_000,
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
        "max_local_dim": 8,
        "bulk_kind": "single",
    },
}

SHARED_WORKSTATION_SETTINGS = {
    "geometry": {
        "length_x": 4,
        "circumference_y": 2,
        "periodic_around_cylinder": True,
        "lattice_type": "honeycomb",
    },
    "finite_dmrg": {
        "max_sites": 16,
        "max_bond_dimension": 128,
        "max_sweeps": 20,
    },
    "ed": {
        "run": True,
        "max_sites": 8,
        "max_hilbert_dim": 250_000,
    },
    "finite_temperature_ed": {
        "run": True,
        "max_sites": 8,
        "max_hilbert_dim": 100_000,
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
        "max_iterations": 10,
        "max_local_dim": 8,
        "bulk_kind": "single",
    },
}

RESOURCE_PROFILES = {
    "local_laptop": LOCAL_LAPTOP_SETTINGS,
    "shared_workstation": SHARED_WORKSTATION_SETTINGS,
}


ACTIVE_RESOURCE_PROFILE = "local_laptop"  # local_laptop | shared_workstation
ACTIVE_RESOURCE_SETTINGS = RESOURCE_PROFILES[ACTIVE_RESOURCE_PROFILE]

# Geometry.
LENGTH_X = int(ACTIVE_RESOURCE_SETTINGS["geometry"]["length_x"])
CIRCUMFERENCE_Y = int(ACTIVE_RESOURCE_SETTINGS["geometry"]["circumference_y"])
PERIODIC_AROUND_CYLINDER = bool(ACTIVE_RESOURCE_SETTINGS["geometry"]["periodic_around_cylinder"])
LATTICE_TYPE = str(ACTIVE_RESOURCE_SETTINGS["geometry"]["lattice_type"])  # honeycomb | square | triangular

# - yao_lee: each gamma bond gets S_gamma S_gamma, T_gamma T_gamma, and ST_gamma ST_gamma terms.
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
# - off: ignored.
# - perturbation: recorded and annotated, but not inserted into the MPO/ED Hamiltonian.
# - hamiltonian: inserted as one-site terms; use symmetry_mode=none for hx/hy fields.
EXTERNAL_FIELD_TREATMENT = "off"  # off | perturbation | hamiltonian
EXTERNAL_FIELD_AXIS = "111"                # 111 | custom
EXTERNAL_FIELD_STRENGTH = 0.0              # used for axis=111 as H/sqrt(3)*(1,1,1)
FIELD_HX = 0.0                             # used for axis=custom
FIELD_HY = 0.0
FIELD_HZ = 0.0
MU_B = 1.0
FIELD_SIGN = 1.0
FIELD_SIGMA_FACTOR = 2.0

# Symmetry simplification/block-sparse controls.
# none: dense tensors, no symmetry constraints.
# u1:   encoded U(1)xU(1) charges using target (2*Sz, 2*Tz).
#       Uses additive integer q = 4096*(2*Sz) + (2*Tz), which matches
#       Tenax AutoMPO's raw charge-difference checks. This is only valid
#       when the Hamiltonian conserves total Sz/Tz.
#       Bond-dependent x/y Yao-Lee terms should use z2 or none instead.
# z2:   parity selection rule. Tenax 0.2 AutoMPO cannot build a true Z2
#       block-sparse MPO because its symmetric AutoMPO path is U1-only; use
#       none for full bond-dependent x/y Yao-Lee runs unless Tenax adds Z2 MPO support.
SYMMETRY_MODE = "z2"      # none | u1 | z2
U1_TARGET_TOTAL_SZ2 = 0     # equals 2 * total S^z
U1_TARGET_TOTAL_TZ2 = 0     # equals 2 * total T^z
Z2_TARGET_PARITY = 0        # 0=even, 1=odd
STRICT_SYMMETRY_SELECTION_RULES = True

# Resource-limited solver defaults from ACTIVE_RESOURCE_SETTINGS.
MAX_DMRG_SITES = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_sites"])
MAX_BOND_DIMENSION = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_bond_dimension"])
MAX_SWEEPS = int(ACTIVE_RESOURCE_SETTINGS["finite_dmrg"]["max_sweeps"])
TRUNCATION_CUTOFF = 1e-8
SEED = 42
INITIAL_STATE_STYLE = "random"  # alternating | random

# Optional comparison workflows.
RUN_ED = bool(ACTIVE_RESOURCE_SETTINGS["ed"]["run"])
MAX_ED_SITES = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_sites"])
MAX_ED_HILBERT_DIM = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_hilbert_dim"])
RUN_FINITE_TEMPERATURE_ED = bool(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["run"])
MAX_THERMAL_ED_SITES = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["max_sites"])
MAX_THERMAL_ED_HILBERT_DIM = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["max_hilbert_dim"])
THERMAL_FULL_SPECTRUM_MAX_DIM = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["full_spectrum_max_dim"])
THERMAL_MAX_EIGENSTATES = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["max_eigenstates"])
TEMPERATURE_MIN = float(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_min"])
TEMPERATURE_MAX = float(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_max"])
TEMPERATURE_POINTS = int(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_points"])
TEMPERATURE_SCALE = str(ACTIVE_RESOURCE_SETTINGS["finite_temperature_ed"]["temperature_scale"])
CHECK_GROUND_STATE_DEGENERACY = True
ED_GROUND_MANIFOLD_ABS_TOL = 1e-12
ED_GROUND_MANIFOLD_REL_TOL = 1e-12
DMRG_EXCITED_OVERLAP_TOL = 1e-6
DMRG_EXCITED_ENERGY_TOL = 1e-7
DMRG_EXCITED_VARIANCE_TOL = 1e-7
DMRG_EXCITED_MAX_ATTEMPTS = 10
RUN_IDMRG = bool(ACTIVE_RESOURCE_SETTINGS["idmrg"]["run"])
IDMRG_MAX_BOND_DIMENSION = int(ACTIVE_RESOURCE_SETTINGS["idmrg"]["max_bond_dimension"])
IDMRG_MAX_ITERATIONS = int(ACTIVE_RESOURCE_SETTINGS["idmrg"]["max_iterations"])
IDMRG_MAX_LOCAL_DIM = int(ACTIVE_RESOURCE_SETTINGS["idmrg"]["max_local_dim"])
IDMRG_BULK_KIND = str(ACTIVE_RESOURCE_SETTINGS["idmrg"]["bulk_kind"])  # auto | pair | single

# Optional alpha-beta phase scans. These are off by default because even small
# exact-diagonalization grids can be expensive. The scan records finite-cluster
# observable diagnostics and uses those diagnostics to assign phase labels.
PHASE_SCAN_MODE_OPTIONS = ("quantum_ed", "classical_product", "both")
RUN_PHASE_SCAN = True
PHASE_SCAN_ONLY = False
PHASE_SCAN_MODE = "both"  # quantum_ed | classical_product | both
PHASE_SCAN_ALPHA_MIN = 0.0
PHASE_SCAN_ALPHA_MAX = 2.25
PHASE_SCAN_ALPHA_POINTS = 17
PHASE_SCAN_BETA_MIN = 0.0
PHASE_SCAN_BETA_MAX = 0.27
PHASE_SCAN_BETA_POINTS = 13
PHASE_SCAN_ED_MAX_SITES = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_sites"])
PHASE_SCAN_ED_MAX_HILBERT_DIM = int(ACTIVE_RESOURCE_SETTINGS["ed"]["max_hilbert_dim"])
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

# Output/runtime behavior.
OUTPUT_FOLDER = "DMRG/outputs"
BACKEND = "auto"  # auto | tenax | tenpy
OVERWRITE_EXISTING_PLOTS = False
CONTINUE_AFTER_PLOT_ERROR = True
STRICT_PLOT_ERRORS = not CONTINUE_AFTER_PLOT_ERROR
SHOW_PROGRESS = True


# Runtime implementation symbols are imported from sibling modules by
# _bind_split_module_implementations(). Keep implementation work in:
# models.py, analysis.py, backend.py, and plot_outputs.py.
# ----------------------------------------------------------------------
# CLI + main
# ----------------------------------------------------------------------

def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tenax/ED benchmarking for the Yao-Lee model.")
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
        "--symmetry-mode",
        "--symmetry_mode",
        dest="symmetry_mode",
        type=str,
        choices=list(SYMMETRY_MODE_OPTIONS),
        default=SYMMETRY_MODE,
        help=(
            "Tensor symmetry mode: none (dense), u1 (strict total Sz/Tz conservation), "
            "or z2 (parity selection rule; Tenax 0.2 AutoMPO cannot build Z2 MPOs, "
            "so use none for full x/y Yao-Lee runs)."
        ),
    )
    parser.add_argument(
        "--u1-target-sz2",
        "--u1_target_sz2",
        dest="u1_target_sz2",
        type=int,
        default=U1_TARGET_TOTAL_SZ2,
        help="Target total 2*Sz sector for symmetry_mode=u1.",
    )
    parser.add_argument(
        "--u1-target-tz2",
        "--u1_target_tz2",
        dest="u1_target_tz2",
        type=int,
        default=U1_TARGET_TOTAL_TZ2,
        help="Target total 2*Tz sector for symmetry_mode=u1.",
    )
    parser.add_argument(
        "--z2-target-parity",
        "--z2_target_parity",
        dest="z2_target_parity",
        type=int,
        choices=list(Z2_PARITY_OPTIONS),
        default=Z2_TARGET_PARITY,
        help="Target global parity sector for symmetry_mode=z2 (0=even, 1=odd).",
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
        help="Number of unit cells along the open cylinder axis.",
    )
    parser.add_argument(
        "--circumference-y",
        "--circumference_y",
        dest="circumference_y",
        type=int,
        default=CIRCUMFERENCE_Y,
        help="Number of unit cells around the cylinder circumference.",
    )
    parser.add_argument(
        "--open-around-cylinder",
        "--open_around_cylinder",
        dest="open_around_cylinder",
        action="store_true",
        help="Use open boundary conditions around the cylinder instead of periodic.",
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
        help="Accepted for compatibility with yaoleemodel.py; Tenax backend may ignore it.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--initial-state",
        "--initial_state",
        dest="initial_state",
        type=str,
        choices=list(INITIAL_STATE_OPTIONS),
        default=INITIAL_STATE_STYLE,
        help="Accepted for compatibility with yaoleemodel.py; Tenax backend currently uses random MPS init.",
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
        help="Run additional Tenax iDMRG-x workflow and compare with finite DMRG/ED.",
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
        "--run-phase-scan",
        "--run_phase_scan",
        dest="run_phase_scan",
        action=argparse.BooleanOptionalAction,
        default=RUN_PHASE_SCAN,
        help="Scan the alpha-beta plane and save observable-based phase diagrams.",
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
        "--phase-scan-mode",
        "--phase_scan_mode",
        dest="phase_scan_mode",
        type=str,
        choices=list(PHASE_SCAN_MODE_OPTIONS),
        default=PHASE_SCAN_MODE,
        help="Phase-scan solver: quantum_ed, classical_product, or both.",
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
        default=PHASE_SCAN_ED_MAX_SITES,
        help="Site cap for quantum ED phase scans.",
    )
    parser.add_argument(
        "--phase-scan-ed-max-hilbert-dim",
        "--phase_scan_ed_max_hilbert_dim",
        dest="phase_scan_ed_max_hilbert_dim",
        type=int,
        default=PHASE_SCAN_ED_MAX_HILBERT_DIM,
        help="Hilbert-space dimension cap for quantum ED phase scans.",
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
    parser.add_argument("--output-folder", "--output_folder", dest="output_folder", type=str, default=OUTPUT_FOLDER)
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(BACKEND_OPTIONS),
        default=BACKEND,
        help="Select backend. auto tries Tenax first, then falls back to yaoleemodel (TeNPy).",
    )
    parser.add_argument(
        "--overwrite-plots",
        action="store_true",
        default=OVERWRITE_EXISTING_PLOTS,
        help="Regenerate PNG outputs even if files already exist.",
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
    parser.set_defaults(open_around_cylinder=(not PERIODIC_AROUND_CYLINDER))
    return parser.parse_args()


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
        "circumference_y",
        "open_around_cylinder",
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
        "symmetry_mode",
        "u1_target_sz2",
        "u1_target_tz2",
        "z2_target_parity",
        "strict_symmetry_selection_rules",
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
        "run_phase_scan",
        "phase_scan_only",
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
    ]
    return {key: parameters.get(key) for key in keys}


def _record_output_status(
    summary: Dict[str, Any],
    key: str,
    filename: str,
    status: str,
    error: str | None = None,
) -> None:
    outputs = summary.setdefault("outputs", {})
    outputs[key] = filename
    output_status = summary.setdefault("output_status", {})
    output_status[key] = {"status": status}
    if error is not None:
        output_status[key]["error"] = error


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


def _bind_split_module_implementations() -> None:
    """Import the canonical implementation modules behind this CLI."""
    global _SPLIT_MODULE_BINDINGS_ACTIVE
    if _SPLIT_MODULE_BINDINGS_ACTIVE:
        return

    try:
        import analysis as analysis_tools
        import backend as tenax_backend
        import models as model_defs
        import plot_outputs
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
                "_m2_values_from_spin_value",
                "_get_z2_symmetry_object",
                "_encode_u1_charge_pair",
                "_u1_charge_encoding_summary",
                "_u1_encoded_phys_charges_for_model",
                "_z2_phys_charges_for_model",
                "_u1_basis_charge_table_for_model",
                "_z2_basis_charge_table_for_model",
                "_u1_encoded_target_charge",
                "_operator_charge_transfer",
                "_validate_symmetry_conserving_terms",
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
            tenax_backend,
            (
                "_build_auto_mpo_from_terms",
                "_empty_tenax_hamiltonian_message",
                "build_tenax_model_mpo",
                "build_tenax_yao_lee_mpo",
                "_extract_dmrg_result",
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
    if bool(args.phase_scan_only):
        args.run_phase_scan = True
    ensure_folder_exists(args.output_folder)
    show_progress = bool(args.progress)
    overwrite_existing = bool(args.overwrite_plots)
    continue_on_plot_error = not bool(args.strict_plot_errors)
    lattice_name = str(args.lattice).lower()
    periodic_y = not args.open_around_cylinder
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
    validate_external_field_symmetry_compatibility(
        hamiltonian_external_field_terms,
        symmetry_mode=args.symmetry_mode,
    )

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
    if args.backend == "tenpy" and args.symmetry_mode != "none":
        raise ValueError(
            "TeNPy backend in this script does not support symmetry_mode != none. "
            "Use --backend tenax for U1/Z2 block-sparse runs."
        )
    if args.backend == "tenpy" and hamiltonian_external_field_terms:
        raise ValueError(
            "TeNPy backend in this script does not support external_field_treatment=hamiltonian. "
            "Use --backend tenax or keep external_field_treatment=perturbation."
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
            circumference_y=args.circumference_y,
            periodic_y=periodic_y,
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
            circumference_y=args.circumference_y,
            periodic_y=periodic_y,
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

    def save_geometry_before_sweep(geometry_obj: Any) -> None:
        nonlocal geometry_plot_status, geometry_plot_error
        configure_run_output_names(geometry_obj)
        filename = output_filename("geometry_diagram.png")
        filepath = os.path.join(args.output_folder, filename)
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

    def save_phase_scan_outputs(summary_obj: Dict[str, Any], phase_scan_data: Dict[str, Any]) -> None:
        phase_scan_filename = output_filename("phase_scan_summary.json")
        phase_scan_filepath = os.path.join(args.output_folder, phase_scan_filename)
        write_json(phase_scan_filepath, phase_scan_data)
        _record_output_status(summary_obj, "phase_scan_summary_json", phase_scan_filename, "saved")
        _save_summary_checkpoint(args.output_folder, summary_obj)
        print(f"[output] saved: {phase_scan_filename}")

        for mode_key, title in (
            ("classical_product", "Classical Product-State Phase Diagram"),
            ("quantum_ed", "Quantum ED Phase Diagram"),
        ):
            mode_data = phase_scan_data.get(mode_key)
            if not isinstance(mode_data, dict):
                continue
            rows = list(mode_data.get("rows", []))
            if len(rows) == 0:
                continue
            base_name = (
                "classical_phase_diagram.png"
                if mode_key == "classical_product"
                else "quantum_phase_diagram.png"
            )
            output_key = f"{mode_key}_phase_diagram_png"
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

    if args.phase_scan_only:
        geometry = build_lattice_geometry(
            lattice=lattice_name,
            length_x=args.length_x,
            circumference_y=args.circumference_y,
            periodic_y=periodic_y,
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
            "model_construction_annotations": {
                "phase_scan": {
                    "mode": str(args.phase_scan_mode),
                    "alpha_points": int(args.phase_scan_alpha_points),
                    "beta_points": int(args.phase_scan_beta_points),
                    "quantum_ed_max_sites": int(args.phase_scan_ed_max_sites),
                    "quantum_ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
                    "classical_restarts": int(args.phase_scan_classical_restarts),
                    "classical_sweeps": int(args.phase_scan_classical_sweeps),
                    "classifier_thresholds": {
                        "quantum_weak_order": float(args.phase_scan_quantum_weak_order_threshold),
                        "classical_weak_order": float(args.phase_scan_classical_weak_order_threshold),
                        "quantum_bond_nematicity": float(args.phase_scan_quantum_nematicity_threshold),
                        "classical_bond_nematicity": float(args.phase_scan_classical_nematicity_threshold),
                    },
                    "note": (
                        "Classical and quantum diagrams are generated by scanning alpha and beta, "
                        "then classifying ground-state structure-factor peaks and bond-energy nematicity."
                    ),
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
            phase_scan_data = run_alpha_beta_phase_scan(
                geometry=geometry,
                model_spec=model_spec,
                lattice_name=lattice_name,
                args=args,
                hamiltonian_external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
            )
            scan_summary["phase_scan"] = phase_scan_data
            scan_summary["stages"]["phase_scan"] = "completed"
            save_phase_scan_outputs(scan_summary, phase_scan_data)
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
        print(json.dumps(to_json_compatible(scan_summary), indent=2, sort_keys=True))
        return

    # Try Tenax first unless user forces tenpy.
    if args.backend in ("auto", "tenax"):
        try:
            geometry = build_lattice_geometry(
                lattice=lattice_name,
                length_x=args.length_x,
                circumference_y=args.circumference_y,
                periodic_y=periodic_y,
            )
            if geometry.number_of_sites > args.max_dmrg_sites:
                raise ValueError(
                    f"Finite DMRG safety cap for profile '{ACTIVE_RESOURCE_PROFILE}' is N <= {args.max_dmrg_sites}, "
                    f"but requested N={geometry.number_of_sites}. Increase --max-dmrg-sites "
                    "only for aragorn/beehive or a dedicated run."
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
                symmetry_mode=args.symmetry_mode,
                u1_target_total_sz2=args.u1_target_sz2,
                u1_target_total_tz2=args.u1_target_tz2,
                z2_target_parity=args.z2_target_parity,
                strict_symmetry_selection_rules=args.strict_symmetry_selection_rules,
                show_progress=show_progress,
            )
            dmrg_energy = float(dmrg_info["E"])
            dmrg_state_obj = tenax_mps
            dmrg_correlations = collect_correlation_matrices_from_tenax(
                tenax_mps,
                geometry,
                model_spec=model_spec,
                show_progress=show_progress,
            )
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
            dmrg_scalar_correlations = build_spin_orbital_scalar_correlations(dmrg_correlations)
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
            dmrg_structure_factor_rows = all_high_symmetry_structure_factors(
                dmrg_scalar_correlations,
                geometry,
                lattice=lattice_name,
                show_progress=show_progress,
                progress_desc="DMRG structure factors",
            )
            backend_used = "tenax"
        except Exception as tenax_exc:
            if args.backend == "tenax":
                raise
            if args.symmetry_mode != "none":
                backend_warning = (
                    f"Tenax failed after requested symmetry_mode={args.symmetry_mode}; "
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
            if hamiltonian_external_field_terms:
                raise RuntimeError(
                    "Tenax failed while external_field_treatment=hamiltonian, and TeNPy fallback "
                    f"does not support inserting Zeeman field terms. Original Tenax error: {tenax_exc}"
                ) from tenax_exc
            if show_progress:
                print(f"[backend] Tenax failed; switching to TeNPy fallback. Reason: {tenax_exc}")
            if backend_warning is None:
                backend_warning = f"Tenax backend failed, fallback to TeNPy: {tenax_exc}"

    # TeNPy fallback via sibling module yaoleemodel.py
    if backend_used is None:
        if lattice_name != "honeycomb":
            raise RuntimeError(
                f"TeNPy fallback does not support lattice='{lattice_name}'. "
                "Only honeycomb is supported in yaoleemodel.py."
            )
        if not (
            model_spec.spin_rep == "1/2"
            and model_spec.orbital_rep == "1/2"
            and model_spec.model_family == "yao_lee"
            and model_spec.ising_axis == "z"
        ):
            raise RuntimeError(
                "TeNPy fallback only supports the legacy default model "
                "(spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, ising_axis=z)."
            )
        yl = _load_tenpy_backend_module()
        stage_start = _start_stage("TeNPy fallback DMRG", show_progress)
        geometry = yl.build_honeycomb_cylinder_geometry(
            length_x=args.length_x,
            circumference_y=args.circumference_y,
            periodic_y=periodic_y,
        )
        if geometry.number_of_sites > args.max_dmrg_sites:
            raise ValueError(
                f"Finite DMRG safety cap for profile '{ACTIVE_RESOURCE_PROFILE}' is N <= {args.max_dmrg_sites}, "
                f"but requested N={geometry.number_of_sites}. Increase --max-dmrg-sites "
                "only for aragorn/beehive or a dedicated run."
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
        )
        dmrg_energy = float(dmrg_info["E"])
        dmrg_state_obj = psi
        dmrg_correlations = yl.collect_correlation_matrices_from_dmrg(psi)
        scalar_native = yl.build_spin_orbital_scalar_correlations(dmrg_correlations)
        dmrg_scalar_correlations = {
            "S": scalar_native["spin_scalar"],
            "T": scalar_native["orbital_scalar"],
            "ST": scalar_native["mixed_scalar"],
        }
        dmrg_bond_rows = yl.all_bond_energies(geometry, dmrg_correlations, args.alpha, args.beta, args.coupling_j)
        dmrg_structure_factor_rows = yl.all_high_symmetry_structure_factors(scalar_native, geometry)
        backend_used = "tenpy_fallback"
        _end_stage("TeNPy fallback DMRG", stage_start, show_progress)

    try:
        if dmrg_state_obj is not None:
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
    except Exception as exc:
        entanglement_warning = f"Failed to compute DMRG entanglement profile: {exc}"

    configure_run_output_names(geometry)
    lattice_label = lattice_display_name(lattice_name)
    model_short_label = model_simplified_name(model_spec)
    size_short_label = geometry_size_filename_label(
        geometry,
        lattice_name,
        args.length_x,
        args.circumference_y,
        periodic_y,
    )
    size_display_label = geometry_size_display_label(
        geometry,
        lattice_name,
        args.length_x,
        args.circumference_y,
        periodic_y,
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
                    "max_sites": int(MAX_ED_SITES),
                    "max_hilbert_dimension": int(MAX_ED_HILBERT_DIM),
                    "ground_manifold_abs_tol": float(args.ed_ground_manifold_abs_tol),
                    "ground_manifold_rel_tol": float(args.ed_ground_manifold_rel_tol),
                    "note": "ED is skipped automatically above either cap.",
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
                    "run": bool(args.run_phase_scan),
                    "scan_only": bool(args.phase_scan_only),
                    "mode": str(args.phase_scan_mode),
                    "alpha_min": float(args.phase_scan_alpha_min),
                    "alpha_max": float(args.phase_scan_alpha_max),
                    "alpha_points": int(args.phase_scan_alpha_points),
                    "beta_min": float(args.phase_scan_beta_min),
                    "beta_max": float(args.phase_scan_beta_max),
                    "beta_points": int(args.phase_scan_beta_points),
                    "quantum_ed_max_sites": int(args.phase_scan_ed_max_sites),
                    "quantum_ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
                    "classical_restarts": int(args.phase_scan_classical_restarts),
                    "classical_sweeps": int(args.phase_scan_classical_sweeps),
                    "classifier_thresholds": {
                        "quantum_weak_order": float(args.phase_scan_quantum_weak_order_threshold),
                        "classical_weak_order": float(args.phase_scan_classical_weak_order_threshold),
                        "quantum_bond_nematicity": float(args.phase_scan_quantum_nematicity_threshold),
                        "classical_bond_nematicity": float(args.phase_scan_classical_nematicity_threshold),
                    },
                    "note": (
                        "The phase diagrams are generated from scanned ground-state structure "
                        "patterns and bond-energy diagnostics, then classified with recorded thresholds."
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
                    "For each gamma bond: J(1+beta) S_gamma_i S_gamma_j + "
                    "J(1-beta) T_gamma_i T_gamma_j + J*alpha ST_gamma_i ST_gamma_j."
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
                "requested_mode": str(args.symmetry_mode),
                "u1": {
                    "charge_encoding": _u1_charge_encoding_summary(),
                    "valid_when": (
                        "The Hamiltonian conserves total Sz and total Tz. "
                        "Examples: ising_like with ising_axis=z; spin-only Heisenberg/XY/XXZ/XYZ "
                        "when transverse Sx/Sy couplings are paired equally."
                    ),
                    "invalid_when": (
                        "Full bond-dependent Yao-Lee x/y channels contain single-axis flip terms "
                        "and cannot be represented as strict U1 without changing the Hamiltonian."
                    ),
                    "fallback_policy": (
                        "If a requested U1 MPO/MPS construction fails in Tenax auto mode, "
                        "the run retries with dense symmetry_mode=none and records the reason in dmrg.info."
                    ),
                },
                "z2": (
                    "Tenax 0.2 AutoMPO is U1-only for symmetric MPO construction here; "
                    "requested z2 runs retry dense and record the fallback."
                ),
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
            "number_of_sites": geometry.number_of_sites,
            "number_of_bonds": len(geometry.bond_list),
            "size_label": size_display_label,
            "mps_path": mps_path_quality(geometry),
        },
        "dmrg": {
            "ground_state_energy": dmrg_energy,
            "energy_per_site": dmrg_energy / geometry.number_of_sites,
            "info": dmrg_info,
            "entanglement": entropy_profiles.get("DMRG"),
            "structure_factors": dmrg_structure_factor_rows,
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
    _save_plot_step(
        summary,
        args.output_folder,
        "dmrg_bond_energy_diagram_png",
        output_filename("dmrg_bond_energy_diagram.png"),
        lambda path: save_bond_energy_diagram(geometry, dmrg_bond_rows, path, plot_title("DMRG Bond-Energy Diagram")),
        overwrite_existing,
        continue_on_plot_error,
    )
    _save_plot_step(
        summary,
        args.output_folder,
        "dmrg_structure_factors_png",
        output_filename("dmrg_structure_factors.png"),
        lambda path: save_structure_factor_plot(dmrg_structure_factor_rows, path, plot_title("DMRG Structure Factors")),
        overwrite_existing,
        continue_on_plot_error,
    )
    _save_plot_step(
        summary,
        args.output_folder,
        "dmrg_scalar_correlation_heatmaps_png",
        output_filename("dmrg_scalar_correlation_heatmaps.png"),
        lambda path: save_scalar_correlation_heatmaps(dmrg_scalar_correlations, path, f"DMRG | {run_plot_title_label}"),
        overwrite_existing,
        continue_on_plot_error,
    )
    summary["stages"]["dmrg_plots"] = "completed"
    _save_summary_checkpoint(args.output_folder, summary)

    # Optional iDMRG workflow (runs after finite DMRG outputs are saved).
    if args.run_idmrg:
        summary["stages"]["idmrg"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        if backend_used != "tenax":
            summary["idmrg"] = {
                "status": "skipped",
                "reason": "iDMRG-x currently requires Tenax backend output.",
            }
            summary["stages"]["idmrg"] = "skipped"
            _save_summary_checkpoint(args.output_folder, summary)
        elif tenax_mpo is None:
            summary["idmrg"] = {
                "status": "failed",
                "error": "Tenax MPO object unavailable after DMRG.",
            }
            summary["stages"]["idmrg"] = "failed"
            _save_summary_checkpoint(args.output_folder, summary)
        else:
            try:
                idmrg_info = run_tenax_idmrg_x_from_finite_mpo(
                    mpo=tenax_mpo,
                    model_spec=model_spec,
                    max_bond_dimension=args.idmrg_max_bond_dimension,
                    max_iterations=args.idmrg_max_iterations,
                    bulk_kind=args.idmrg_bulk_kind,
                    max_local_dim=args.idmrg_max_local_dim,
                    show_progress=show_progress,
                )
                summary["idmrg"] = idmrg_info
                if isinstance(idmrg_info.get("entanglement"), dict):
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
    if args.run_ed:
        summary["stages"]["ed"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        local_dim = int(model_spec.physical_dim)
        hilbert_dim = int(local_dim ** geometry.number_of_sites)
        if geometry.number_of_sites > MAX_ED_SITES:
            summary["ed"] = {
                "status": "skipped",
                "reason": f"ED is limited to {MAX_ED_SITES} sites or fewer.",
            }
            summary["stages"]["ed"] = "skipped"
        elif hilbert_dim > MAX_ED_HILBERT_DIM:
            summary["ed"] = {
                "status": "skipped",
                "reason": (
                    f"ED Hilbert-space dimension {hilbert_dim} exceeds limit {MAX_ED_HILBERT_DIM} "
                    f"(local_dim={local_dim}, sites={geometry.number_of_sites})."
                ),
            }
            summary["stages"]["ed"] = "skipped"
        else:
            try:
                if backend_used == "tenax":
                    ed_spectrum, ed_vectors = run_small_cluster_exact_spectrum(
                        geometry=geometry,
                        model_spec=model_spec,
                        alpha=args.alpha,
                        beta=args.beta,
                        coupling_j=args.coupling_j,
                        eigenstate_count=(
                            max(2, int(args.thermal_max_eigenstates))
                            if bool(args.check_ground_state_degeneracy)
                            else 2
                        ),
                        check_ground_state_degeneracy=bool(args.check_ground_state_degeneracy),
                        external_field_terms=hamiltonian_external_field_terms,
                        show_progress=show_progress,
                        ground_manifold_abs_tol=float(args.ed_ground_manifold_abs_tol),
                        ground_manifold_rel_tol=float(args.ed_ground_manifold_rel_tol),
                    )
                    ed_energy = float(ed_spectrum["ground_state_energy"])
                    ed_state = ed_vectors[:, 0]
                    ed_correlations = collect_correlation_matrices_from_ed(
                        geometry,
                        ed_state,
                        model_spec=model_spec,
                        show_progress=show_progress,
                    )
                    ed_scalar_correlations = build_spin_orbital_scalar_correlations(ed_correlations)
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
                    ed_structure_factor_rows = all_high_symmetry_structure_factors(
                        ed_scalar_correlations,
                        geometry,
                        lattice=lattice_name,
                        show_progress=show_progress,
                        progress_desc="ED structure factors",
                    )
                else:
                    yl = _load_tenpy_backend_module()
                    stage_start = _start_stage("TeNPy fallback ED", show_progress)
                    ed_energy, ed_state = yl.run_small_cluster_exact_diagonalization(
                        geometry=geometry,
                        alpha=args.alpha,
                        beta=args.beta,
                        coupling_j=args.coupling_j,
                    )
                    ed_spectrum = {
                        "solver_mode": "tenpy_legacy_ground_state_only",
                        "eigenstates_returned": 1,
                        "energies": [float(ed_energy)],
                        "ground_state_energy": float(ed_energy),
                        "ground_state_degeneracy_check_enabled": bool(args.check_ground_state_degeneracy),
                        "ground_state_degeneracy": None,
                        "ground_state_degeneracy_tolerance": None,
                        "ground_state_degeneracy_is_lower_bound": None,
                        "ground_state_degeneracy_status": "not_resolved",
                        "first_excited_energy": None,
                        "spectral_gap": None,
                    }
                    ed_correlations = yl.collect_correlation_matrices_from_ed(geometry, ed_state)
                    ed_scalar_native = yl.build_spin_orbital_scalar_correlations(ed_correlations)
                    ed_scalar_correlations = {
                        "S": ed_scalar_native["spin_scalar"],
                        "T": ed_scalar_native["orbital_scalar"],
                        "ST": ed_scalar_native["mixed_scalar"],
                    }
                    ed_bond_rows = yl.all_bond_energies(geometry, ed_correlations, args.alpha, args.beta, args.coupling_j)
                    ed_structure_factor_rows = yl.all_high_symmetry_structure_factors(ed_scalar_native, geometry)
                    _end_stage("TeNPy fallback ED", stage_start, show_progress)

                ed_entropy_warning: str | None = None
                ed_entropy_profile: Dict[str, Any] | None = None
                try:
                    ed_entropy_profile = compute_ed_entropy_profile_from_state(
                        state=ed_state,
                        n_sites=geometry.number_of_sites,
                        local_dim=local_dim,
                        orders=ENTROPY_ORDERS,
                        show_progress=show_progress,
                    )
                    entropy_profiles["ED"] = ed_entropy_profile
                except Exception as exc:
                    ed_entropy_warning = f"Failed to compute ED entanglement profile: {exc}"

                summary["ed"] = {
                    "status": "completed",
                    "ground_state_energy": ed_energy,
                    "energy_per_site": ed_energy / geometry.number_of_sites,
                    "absolute_energy_difference_dmrg_minus_ed": abs(dmrg_energy - ed_energy),
                    "structure_factors": ed_structure_factor_rows,
                }
                if ed_spectrum is not None:
                    summary["ed"]["spectrum"] = ed_spectrum
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

                _save_plot_step(
                    summary,
                    args.output_folder,
                    "ed_bond_energy_diagram_png",
                    output_filename("ed_bond_energy_diagram.png"),
                    lambda path: save_bond_energy_diagram(geometry, ed_bond_rows, path, plot_title("ED Bond-Energy Diagram")),
                    overwrite_existing,
                    continue_on_plot_error,
                )
                _save_plot_step(
                    summary,
                    args.output_folder,
                    "ed_structure_factors_png",
                    output_filename("ed_structure_factors.png"),
                    lambda path: save_structure_factor_plot(ed_structure_factor_rows, path, plot_title("ED Structure Factors")),
                    overwrite_existing,
                    continue_on_plot_error,
                )
                _save_plot_step(
                    summary,
                    args.output_folder,
                    "ed_scalar_correlation_heatmaps_png",
                    output_filename("ed_scalar_correlation_heatmaps.png"),
                    lambda path: save_scalar_correlation_heatmaps(ed_scalar_correlations, path, f"ED | {run_plot_title_label}"),
                    overwrite_existing,
                    continue_on_plot_error,
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
            except Exception as exc:
                summary["ed"] = {"status": "failed", "error": str(exc)}
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
    if (
        isinstance(summary.get("idmrg"), dict)
        and summary["idmrg"].get("status") == "completed"
        and "energy_per_original_site" in summary["idmrg"]
    ):
        method_energy_comparison["iDMRG-x"] = float(summary["idmrg"]["energy_per_original_site"])
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
        overwrite_existing,
        continue_on_plot_error,
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
        and "energy_per_original_site" in summary["idmrg"]
    ):
        method_spectrum_comparison["iDMRG-x"] = {
            "status": "ground_state_only",
            "ground_state_energy_per_site": float(summary["idmrg"]["energy_per_original_site"]),
            "ground_state_degeneracy": None,
            "ground_state_degeneracy_label": "unresolved",
            "ground_state_degeneracy_status": "not_resolved",
            "first_excited_energy": None,
            "first_excited_energy_per_site": None,
            "spectral_gap": None,
            "note": (
                "Tenax exposes an iPEPS/CTM excitation API, but this driver uses finite MPS "
                "and iDMRG-x MPO workflows, so no compatible iDMRG spectral gap is reported."
            ),
        }
    summary["low_energy_spectrum_comparison"] = method_spectrum_comparison
    _save_summary_checkpoint(args.output_folder, summary)
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
        overwrite_existing,
        continue_on_plot_error,
    )

    if args.run_finite_temperature:
        summary["stages"]["finite_temperature"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        local_dim = int(model_spec.physical_dim)
        thermal_hilbert_dim = int(local_dim ** geometry.number_of_sites)
        if geometry.number_of_sites > int(args.thermal_max_sites):
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
            except Exception as exc:
                summary["finite_temperature"] = {"status": "failed", "error": str(exc)}
                summary["stages"]["finite_temperature"] = "failed"
                if not continue_on_plot_error:
                    raise
        _save_summary_checkpoint(args.output_folder, summary)

    if len(entropy_profiles) > 0:
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
            overwrite_existing,
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
            overwrite_existing,
            continue_on_plot_error,
        )

    if args.run_phase_scan:
        summary["stages"]["phase_scan"] = "running"
        _save_summary_checkpoint(args.output_folder, summary)
        try:
            phase_scan_data = run_alpha_beta_phase_scan(
                geometry=geometry,
                model_spec=model_spec,
                lattice_name=lattice_name,
                args=args,
                hamiltonian_external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
            )
            summary["phase_scan"] = phase_scan_data
            summary["stages"]["phase_scan"] = "completed"
            save_phase_scan_outputs(summary, phase_scan_data)
        except Exception as exc:
            summary["phase_scan"] = {"status": "failed", "error": str(exc)}
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
    print(json.dumps(to_json_compatible(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
