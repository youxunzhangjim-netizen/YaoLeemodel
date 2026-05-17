# Yao-Lee DMRG / ED / PEPS Driver

This folder contains a Python workflow for finite and infinite tensor-network
and exact-diagonalization studies of the Yao-Lee spin-orbital model and several
spin-only benchmark Hamiltonians.

The main entry point is:

```bash
python ylmodel_main.py
```

Most everyday runs can be controlled by editing the configuration block near
the top of `ylmodel_main.py`, or by passing the matching CLI flags.

## File Layout

- `ylmodel_main.py`: options, CLI parsing, consistency checks, run orchestration.
- `models.py`: model specs, geometry, local operators, Hamiltonian terms, symmetry helpers.
- `ed_backend.py`: in-repo ED, sparse ED, Tz-sector ED, and projector ED.
- `quspin_backend.py`: optional QuSpin ED route for the supported native subset.
- `tenpy_backend.py`: TeNPy finite DMRG and iDMRG backend.
- `tenax_backend.py`: Tenax finite DMRG and iDMRG backend when available.
- `peps_backend.py`: optional quimb PEPS/iPEPS backend.
- `analysis.py`: phase scans, observables, diagnostics, summary helpers.
- `plot_outputs.py`: plotting helpers.
- `tests/`: validation and regression tests.
- `outputs/`: JSON summaries and PNG plots.

## Quick Start

Run the default configuration:

```bash
python ylmodel_main.py
```

Run finite TeNPy DMRG:

```bash
python ylmodel_main.py --backend tenpy --method dmrg
```

Run ED with the standard in-repo projector route:

```bash
python ylmodel_main.py --ed-backend standard --ed-symmetry-engine standard_projector
```

Run QuSpin for the supported native subset:

```bash
python ylmodel_main.py --ed-backend quspin --ed-symmetry-engine quspin
```

Run a normal alpha-beta phase scan:

```bash
python ylmodel_main.py --run-phase-scan --phase-scan-channels normal
```

Run only an external-field scan:

```bash
python ylmodel_main.py --run-phase-scan --phase-scan-channels external --external-scan-mode e_b
```

## Main Options

These are the main editable options in `ylmodel_main.py`.

### Resource Profile

| Option | Choices |
| --- | --- |
| `ACTIVE_RESOURCE_PROFILE` | `local_laptop`, `shared_workstation` |

The active profile sets default lattice size, DMRG bond dimensions, ED caps,
PEPS/iPEPS caps, and iDMRG iteration limits.

### Backend And Method

| Option | Choices |
| --- | --- |
| `BACKEND` | `auto`, `tenax`, `tenpy`, `quimb` |
| `METHOD` | `auto`, `dmrg`, `idmrg`, `peps`, `ipeps` |

Use `BACKEND="quimb"` with `METHOD="peps"` for finite PEPS, and
`METHOD="ipeps"` for infinite PEPS.

### Model

| Option | Choices |
| --- | --- |
| `MODEL_FAMILY` | `yao_lee`, `ising_like`, `heisenberg`, `xy`, `xxz`, `xyz` |
| `SPIN_REP` | `1/2`, `3/2` |
| `ORBITAL_REP` | `0`, `1/2` |
| `ISING_AXIS` | `x`, `y`, `z` |

For spin-only benchmark models, set `ORBITAL_REP="0"`.

The spin-orbital Yao-Lee Hamiltonian is

```text
H = -J sum_<ij>_gamma
    [ alpha S_i.S_j - 2 S_i^gamma S_j^gamma - beta ]
    [ T_i.T_j - beta ].
```

The code works in the transformed orbital basis where `T_i.T_j` is
Heisenberg-like.

### External Field

| Option | Choices |
| --- | --- |
| `EXTERNAL_FIELD_TREATMENT` | `off`, `perturbation`, `hamiltonian` |
| `EXTERNAL_FIELD_AXIS` | `111`, `001`, `custom` |

`EXTERNAL_FIELD_STRENGTH` is used for `axis=111` and `axis=001`.
`FIELD_HX`, `FIELD_HY`, and `FIELD_HZ` are used for `axis=custom`.

The field is a spin Zeeman term:

```text
H_field = FIELD_SIGN * MU_B * FIELD_SIGMA_FACTOR
          * sum_i (hx Sx_i + hy Sy_i + hz Sz_i).
```

The local spin operators use the normalized convention `S = sigma / 2`.
For a physical `-H n.S` field, use `FIELD_SIGN=-1` and
`FIELD_SIGMA_FACTOR=1`.

Treatment meanings:

- `off`: no field is recorded or inserted.
- `perturbation`: field metadata is recorded, but the Hamiltonian is unchanged.
- `hamiltonian`: field terms are inserted into ED/MPO/PEPS Hamiltonians.

For Yao-Lee, spin fields do not break total orbital `Tz` because they act only
on the spin sector.

### Shared Symmetry Options

| Option | Choices |
| --- | --- |
| `SYMMETRY_REDUCTIONS` | `auto`, `none`, `sz`, `tz`, `z2` |
| `Z2_TARGET_PARITY` | `0`, `1` |

For `MODEL_FAMILY="yao_lee"` and `ORBITAL_REP="1/2"`:

- Total `Sz` is not conserved. It is rejected or dropped.
- Total orbital `Tz` is conserved and is the safe production U(1) symmetry.
- Optional spin-sector Z2 is field-dependent and backend-dependent.
- Time reversal, flux sectors, pure lattice C3, and reflection are diagnostics,
  not default Hilbert-space block labels.

### ED Options

| Option | Choices |
| --- | --- |
| `ED_BACKEND` | `standard`, `ed`, `quspin` |
| `ED_SYMMETRY_ENGINE` | `auto`, `standard_projector`, `projector`, `quspin`, `quspin_native`, `quspin_experimental_c3` |
| `ED_C3_MODE` | `auto`, `off`, `on` |
| `ED_C3_Q_BLOCKS` | `all`, `0`, `1`, `2` |
| `ED_Z2_MODE` | `auto`, `off`, `on` |
| `ED_Z2_KIND` | `auto`, `spin_flip`, `spin_pi_z` |

ED routing is physics-first:

- `standard_projector` supports the in-repo Tz parent basis, spin_pi_z
  projector, fused physical-site translations, and true combined spin-lattice
  C3 when allowed.
- `quspin` / `quspin_native` supports only the validated QuSpin-native subset:
  Tz and zero-field `spin_flip` Z2 in cases without fused translations or C3.
- `quspin_experimental_c3` is an API sandbox. It rejects pure site-permutation
  C3 maps for Yao-Lee because the physical combined C3 also includes a local
  spin rotation.

Combined C3 is allowed only when:

- model is spin-orbital Yao-Lee,
- field class is zero field or normalized `[111]`,
- geometry is a honeycomb torus with `Lx=Ly`,
- both x and y directions are periodic,
- momentum is the Gamma sector, `kx=ky=0`.

If both spin-sector Z2 and combined C3 are requested, the current planner keeps
the C3 route and drops Z2. A full joint group projector is not implemented yet.

### Translation Options

| Option | Choices |
| --- | --- |
| `USE_TRANSLATION_X_BLOCK` | `0`, `1` |
| `USE_TRANSLATION_Y_BLOCK` | `0`, `1` |
| `MOMENTUM_X_BLOCK` | integer block label |
| `MOMENTUM_Y_BLOCK` | integer block label |

For Yao-Lee spin-orbital ED, translations must move the fused physical site
`(S_i, T_i)` together. The standard projector path implements this. The current
QuSpin tensor-basis path does not use separate spin and orbital translation
blocks for production Tz+translation runs.

### Phase Scan Options

| Option | Choices |
| --- | --- |
| `PHASE_SCAN_MODE` | `quantum`, `classical`, `both`, `all` |
| `PHASE_SCAN_METHODS` | `ed`, `dmrg`, `idmrg`, `peps`, `ipeps`, `all` |
| `PHASE_SCAN_CHANNELS` | `auto`, `none`, `normal`, `external`, `both` |
| `EXTERNAL_SCAN_MODE` | `none`, `e_b`, `alpha_b_classical`, `alpha_b_quantum`, `alpha_b_both`, `alpha_b_all` |

Channel meanings:

- `normal`: run the usual alpha-beta phase diagram.
- `external`: run the selected external-field scan.
- `both`: run normal and external scans.
- `none`: run no phase scan.
- `auto`: choose external when an active field scan is configured, otherwise
  normal.

Legacy `PHASE_SCAN_MODE` aliases are also accepted by the CLI for old command
lines, including `ed`, `dmrg`, `idmrg`, `peps`, `ipeps`, `classical_product`,
`tenpy_dmrg`, `tenpy_idmrg`, `quimb_peps`, and `quimb_ipeps`.

External scan modes:

- `none`: no external-field scan.
- `e_b`: fixed alpha/beta, scan field magnitude and plot ED bands plus DMRG
  ground-state evolution when available.
- `alpha_b_classical`: scan alpha and field magnitude with classical labels.
- `alpha_b_quantum`: scan alpha and field magnitude with quantum labels.
- `alpha_b_both`: save classical and quantum alpha-field diagrams.
- `alpha_b_all`: alias for the full alpha-field output family.

### PEPS And iPEPS Options

| Option | Choices |
| --- | --- |
| `PEPS_SYMMETRY_MODE` | `auto`, `none`, `u1_tz`, `u1_tz_z2` |
| `IPEPS_SYMMETRY_MODE` | `auto`, `none`, `u1_tz`, `u1_tz_z2` |
| `IPEPS_UNIT_CELL_KIND` | `auto`, `minimal`, `two_sublattice`, `stripy`, `zigzag`, `plaquette` |
| `IPEPS_CONTRACTION_METHOD` | `auto`, `ctmrg`, `crtg`, `boundary` |

The current quimb PEPS/iPEPS path records Tz symmetry intent and dense fallback
status. It should not claim a block-sparse speedup unless the backend actually
uses symmetric tensors.

## Output Files

All normal outputs go to:

```text
DMRG/outputs/
```

Typical files:

- `*_run_summary.json`: full run metadata and results.
- `*_phase_scan_summary.json`: phase scan data.
- `*_geometry_diagram.png`: lattice geometry, when enabled.
- `*_bond_energy_diagram.png`: resolved bond energies with spin-vector overlay.
- `*_phase_diagram.png`: phase diagram plots.
- `*_structure_factors.png`: structure-factor plots.
- `*_energy_comparison.png`: available method energy comparison.

If `PLOT_GEOMETRY=0`, geometry plotting is intentionally skipped and no geometry
PNG is written.

## Common Commands

Small ED-friendly torus:

```bash
python ylmodel_main.py \
  --length-x 2 --length-y 2 --circumference-x --circumference-y \
  --backend tenpy --method dmrg \
  --ed-backend standard
```

Yao-Lee with safe Tz symmetry:

```bash
python ylmodel_main.py \
  --model-family yao_lee \
  --orbital-rep 1/2 \
  --symmetry-reductions tz
```

Zero-field ED with projector C3:

```bash
python ylmodel_main.py \
  --ed-backend standard \
  --ed-symmetry-engine standard_projector \
  --ed-c3-mode on \
  --momentum-x-block 0 \
  --momentum-y-block 0
```

Pure `[111]` Hamiltonian field:

```bash
python ylmodel_main.py \
  --external-field-treatment hamiltonian \
  --external-field-axis 111 \
  --external-field-strength 1.0
```

Pure `z` Hamiltonian field:

```bash
python ylmodel_main.py \
  --external-field-treatment hamiltonian \
  --external-field-axis 001 \
  --external-field-strength 1.0
```

Custom field:

```bash
python ylmodel_main.py \
  --external-field-treatment hamiltonian \
  --external-field-axis custom \
  --field-hx 1 --field-hy 0 --field-hz 0
```

## Validation

Fast validation:

```bash
python -m py_compile ylmodel_main.py models.py ed_backend.py quspin_backend.py analysis.py plot_outputs.py
python tests/test_projector_ed_symmetry_path.py
python tests/test_external_field_axis_options.py
```

Slow projector ED validation is opt-in:

```bash
YL_RUN_SLOW_PROJECTOR_ED=1 python tests/test_projector_ed_symmetry_path.py
```

Use the slow validation carefully on small clusters. It can be much heavier than
the fast routing and commutator tests.

## Practical Notes

- For spin-orbital Yao-Lee, do not impose total `Sz`.
- Use total orbital `Tz` as the production U(1) symmetry.
- External spin fields preserve `Tz`.
- Pure `[111]` field breaks pure spin-sector Z2 but can preserve combined
  spin-lattice C3 as a diagnostic/projector symmetry when geometry allows it.
- Pure `Hz` preserves `spin_pi_z`, but not combined C3.
- QuSpin native C3 is not used for Yao-Lee because pure site permutations are
  not the physical combined C3.
- Standard projector ED is the reference implementation for fused translations
  and true combined C3.
- DMRG/iDMRG are the practical choices for larger clusters.
- PEPS/iPEPS are optional quimb paths and should be treated as experimental
  until the dense/symmetric tensor status in the run summary says otherwise.
