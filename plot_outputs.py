#!/usr/bin/env python3
"""PNG output helpers for the Yao-Lee driver.

This module owns plotting, diagram rendering, and filesystem output helpers
only. Hamiltonian construction remains in ``models.py``/``ed_backend.py``;
scan analysis remains in ``analysis.py``; Tenax execution remains in
``tenax_backend.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

from analysis import ENTROPY_ORDERS
from models import (
    GeometryData,
    ModelSpec,
    build_lattice_geometry,
    honeycomb_plaquette_flux_operators,
    lattice_display_name,
    phase_scan_method_display_name,
    two_site_operator_terms_for_bond,
)


METHOD_ORDER = ("DMRG", "PEPS", "ED", "iDMRG-x", "iPEPS", "quimb_peps", "quimb_ipeps")
METHOD_COLORS = {
    "DMRG": "#1f77b4",
    "PEPS": "#008b8b",
    "ED": "#ff7f0e",
    "iDMRG-x": "#2ca02c",
    "iPEPS": "#7b3294",
    "quimb_peps": "#008b8b",
    "quimb_ipeps": "#7b3294",
}
METHOD_MARKERS = {
    "DMRG": "o",
    "PEPS": "D",
    "ED": "s",
    "iDMRG-x": "^",
    "iPEPS": "*",
    "quimb_peps": "D",
    "quimb_ipeps": "*",
}
METHOD_LINESTYLES = {
    "DMRG": "-",
    "PEPS": "-",
    "ED": "--",
    "iDMRG-x": ":",
    "iPEPS": "-.",
    "quimb_peps": "-",
    "quimb_ipeps": "-.",
}
CHANNEL_ORDER = ("S", "T", "ST")
CHANNEL_COLORS = {
    "S": "#1f77b4",
    "T": "#9467bd",
    "ST": "#2ca02c",
    "total": "#666666",
}
CHANNEL_LINESTYLES = {
    "S": "solid",
    "T": "dashed",
    "ST": "dotted",
    "total": "solid",
}
CHANNEL_LABELS = {
    "S": "spin S",
    "T": "orbital T",
    "ST": "mixed ST",
    "total": "total",
}
PLOTTED_BOND_CHANNELS = ("S", "T", "ST", "total")
BOND_ENERGY_CMAP = "viridis"
GAMMA_COLORS = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
BACKGROUND_BOND_ZORDER = 1
RESOLVED_BOND_ZORDER = 3
SITE_MARKER_ZORDER = 20
SPIN_VECTOR_HALO_ZORDER = 49
SPIN_VECTOR_ZORDER = 50


def titled_for_run(base_title: str, title_label: str | None = None) -> str:
    if title_label:
        return f"{base_title}\n{title_label}"
    return base_title


def _ordered_available_methods(data: Dict[str, Any]) -> List[str]:
    return [method for method in METHOD_ORDER if method in data]


def _lattice_size_scale(n_sites: int, *, min_scale: float = 0.68, max_scale: float = 1.32) -> float:
    """Return a gentle visual scale that shrinks as lattice size grows."""
    site_count = max(int(n_sites), 1)
    raw_scale = (12.0 / float(site_count)) ** 0.25
    return float(np.clip(raw_scale, min_scale, max_scale))


def _scaled_value(
    n_sites: int,
    base: float,
    *,
    min_value: float,
    max_value: float,
    exponent: float = 0.25,
) -> float:
    site_count = max(int(n_sites), 1)
    raw_scale = (12.0 / float(site_count)) ** float(exponent)
    return float(np.clip(float(base) * raw_scale, float(min_value), float(max_value)))


def _field_direction_text(field_vector: Any) -> str | None:
    if field_vector is None:
        return None
    try:
        vector = np.asarray(field_vector, dtype=float).reshape(3)
    except Exception:
        return None
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 1e-14:
        return None
    direction = vector / magnitude
    return "H/|H|=(" + ", ".join(f"{value:.3g}" for value in direction) + ")"


def _annotate_field_direction(axis: Any, field_vector: Any, *, fontsize: float) -> None:
    label = _field_direction_text(field_vector)
    if label is None:
        return
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#777777",
            "alpha": 0.86,
            "linewidth": 0.6,
        },
        zorder=60,
    )


def _phase_row_method_key(row: Dict[str, Any]) -> str:
    method = row.get("method", row.get("backend", row.get("solver", "")))
    method_text = str(method).strip()
    if method_text == "tenpy" and "idmrg" in str(row.get("scan_type", row.get("engine", ""))).lower():
        return "tenpy_idmrg"
    if method_text == "quimb_peps":
        return "quimb_peps"
    if method_text == "quimb_ipeps":
        return "quimb_ipeps"
    return method_text


def ensure_folder_exists(folder_path: str) -> None:
    os.makedirs(folder_path, exist_ok=True)


def _resolve_target_folder(output_folder: str | None, default_folder: str) -> str:
    if output_folder is None:
        return os.path.abspath(default_folder)
    expanded_folder = os.path.expanduser(str(output_folder))
    if os.path.isabs(expanded_folder):
        candidate = os.path.abspath(expanded_folder)
    else:
        candidate = os.path.abspath(expanded_folder)
    script_name = os.path.basename(SCRIPT_DIR)
    duplicate_root = os.path.abspath(os.path.join(SCRIPT_DIR, script_name))
    if candidate == duplicate_root or candidate.startswith(duplicate_root + os.sep):
        candidate = os.path.abspath(
            os.path.join(os.path.dirname(SCRIPT_DIR), os.path.relpath(candidate, SCRIPT_DIR))
        )
    return candidate


def _geometry_positions(geometry: Any) -> np.ndarray:
    if hasattr(geometry, "positions"):
        return np.asarray(geometry.positions, dtype=float)
    if hasattr(geometry, "coordinates"):
        return np.asarray(geometry.coordinates, dtype=float)
    raise AttributeError("Geometry object must provide positions or coordinates.")


def _bond_i_j_gamma(bond: Any) -> Tuple[int, int, str]:
    if hasattr(bond, "i") and hasattr(bond, "j") and hasattr(bond, "gamma"):
        return int(bond.i), int(bond.j), str(bond.gamma)
    if hasattr(bond, "site_i") and hasattr(bond, "site_j") and hasattr(bond, "bond_type"):
        return int(bond.site_i), int(bond.site_j), str(bond.bond_type)
    raise AttributeError("Bond object must have (i,j,gamma) or (site_i,site_j,bond_type).")


def _operator_channel(op_name: str) -> str:
    if op_name.startswith("ST"):
        return "ST"
    if op_name.startswith("T"):
        return "T"
    if op_name.startswith("S"):
        return "S"
    return op_name


def _ordered_channels(channels: List[str]) -> List[str]:
    seen = set(channels)
    ordered = [channel for channel in CHANNEL_ORDER if channel in seen]
    ordered.extend(sorted(channel for channel in seen if channel not in CHANNEL_ORDER))
    return ordered


def _centered_offsets(count: int, spacing: float) -> List[float]:
    if count <= 1:
        return [0.0]
    center = 0.5 * float(count - 1)
    return [(float(index) - center) * spacing for index in range(count)]


def _offset_segment(p_i: np.ndarray, p_j: np.ndarray, offset: float) -> List[np.ndarray]:
    direction = np.asarray(p_j, dtype=float) - np.asarray(p_i, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        normal = np.asarray([0.0, 1.0])
    else:
        normal = np.asarray([-direction[1], direction[0]]) / length
    return [np.asarray(p_i, dtype=float) + offset * normal, np.asarray(p_j, dtype=float) + offset * normal]


def _position_span(positions: np.ndarray) -> float:
    if positions.size == 0:
        return 1.0
    return max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0)


def _channels_for_geometry_bond(
    gamma: str,
    model_spec: ModelSpec | None,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
) -> List[str]:
    if model_spec is None:
        return []
    channels: List[str] = []
    for coeff, op_i, op_j in two_site_operator_terms_for_bond(
        gamma,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
    ):
        if abs(complex(coeff)) <= 1e-14:
            continue
        channels.append(_operator_channel(str(op_i)))
        channels.append(_operator_channel(str(op_j)))
    return _ordered_channels(channels)


def _bond_row_channel_values(row: Dict[str, Any]) -> Dict[str, float]:
    if isinstance(row.get("channel_energies"), dict):
        return {
            str(channel): float(value)
            for channel, value in row["channel_energies"].items()
        }
    values: Dict[str, float] = {}
    components = row.get("components", [])
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            channel = str(component.get("channel", "total"))
            values[channel] = values.get(channel, 0.0) + float(component.get("energy", 0.0))
    if len(values) == 0:
        values["total"] = float(row["O_ij_gamma"])
    return values


def _plottable_bond_channel_values(row: Dict[str, Any]) -> Dict[str, float]:
    values = _bond_row_channel_values(row)
    return {
        channel: value
        for channel, value in values.items()
        if channel in PLOTTED_BOND_CHANNELS and np.isfinite(float(value))
    }


def save_geometry_diagram(
    geometry: GeometryData,
    filepath: str,
    lattice: str,
    title_label: str | None = None,
    model_spec: ModelSpec | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    coupling_j: float = 1.0,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    external_field_vector: Tuple[float, float, float] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    positions = _geometry_positions(geometry)
    n_sites = int(getattr(geometry, "number_of_sites", positions.shape[0]))
    bond_width = _scaled_value(n_sites, 1.5, min_value=0.75, max_value=2.25)
    guide_bond_width = _scaled_value(n_sites, 0.8, min_value=0.35, max_value=1.2)
    site_size = _scaled_value(n_sites, 20.0, min_value=9.0, max_value=34.0, exponent=0.34)
    gamma_fontsize = _scaled_value(n_sites, 7.0, min_value=4.8, max_value=8.6, exponent=0.18)
    label_fontsize = _scaled_value(n_sites, 10.0, min_value=7.5, max_value=11.5, exponent=0.14)
    title_fontsize = _scaled_value(n_sites, 12.0, min_value=9.0, max_value=13.0, exponent=0.12)
    legend_fontsize = _scaled_value(n_sites, 8.0, min_value=6.2, max_value=9.4, exponent=0.14)
    offset_spacing = 0.025 * _position_span(positions)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for bond in geometry.bond_list:
        site_i, site_j, gamma = _bond_i_j_gamma(bond)
        p_i = positions[site_i]
        p_j = positions[site_j]
        channels = _channels_for_geometry_bond(gamma, model_spec, alpha, beta, coupling_j, jx, jy, jz)
        if len(channels) == 0:
            ax.plot(
                [p_i[0], p_j[0]],
                [p_i[1], p_j[1]],
                color=GAMMA_COLORS.get(gamma, "#666666"),
                linewidth=bond_width,
                alpha=0.9,
            )
            continue
        ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], color="#bbbbbb", linewidth=guide_bond_width, alpha=0.5)
        for channel, offset in zip(channels, _centered_offsets(len(channels), offset_spacing)):
            segment = _offset_segment(p_i, p_j, offset)
            ax.plot(
                [segment[0][0], segment[1][0]],
                [segment[0][1], segment[1][1]],
                color=CHANNEL_COLORS.get(channel, "#666666"),
                linestyle=CHANNEL_LINESTYLES.get(channel, "solid"),
                linewidth=bond_width,
                alpha=0.95,
            )
        midpoint = 0.5 * (p_i + p_j)
        ax.text(
            midpoint[0],
            midpoint[1],
            gamma,
            color=GAMMA_COLORS.get(gamma, "#666666"),
            fontsize=gamma_fontsize,
            ha="center",
            va="center",
            zorder=4,
        )

    if hasattr(geometry, "sublattice_indices"):
        sublattice = np.asarray(geometry.sublattice_indices)
        if np.any(sublattice == 1):
            a_idx = np.where(sublattice == 0)[0]
            b_idx = np.where(sublattice == 1)[0]
            ax.scatter(positions[a_idx, 0], positions[a_idx, 1], s=site_size, c="#111111", label="A")
            ax.scatter(positions[b_idx, 0], positions[b_idx, 1], s=site_size, c="#ff7f0e", label="B")
        else:
            ax.scatter(positions[:, 0], positions[:, 1], s=0.8 * site_size, c="#111111", label="sites")
    else:
        ax.scatter(positions[:, 0], positions[:, 1], s=0.8 * site_size, c="#111111", label="sites")
    ax.set_title(titled_for_run(f"{lattice_display_name(lattice)} Lattice Geometry", title_label), fontsize=title_fontsize)
    ax.set_xlabel("x", fontsize=label_fontsize)
    ax.set_ylabel("y", fontsize=label_fontsize)
    ax.tick_params(labelsize=max(6.0, label_fontsize - 1.5))
    handles, labels = ax.get_legend_handles_labels()
    if model_spec is not None:
        channel_handles = [
            Line2D(
                [0],
                [0],
                color=CHANNEL_COLORS.get(channel, "#666666"),
                linestyle=CHANNEL_LINESTYLES.get(channel, "solid"),
                linewidth=max(bond_width, 1.2),
                label=CHANNEL_LABELS.get(channel, channel),
            )
            for channel in CHANNEL_ORDER
        ]
        handles.extend(channel_handles)
        labels.extend([handle.get_label() for handle in channel_handles])
    ax.legend(handles, labels, loc="upper right", fontsize=legend_fontsize)
    _annotate_field_direction(ax, external_field_vector, fontsize=legend_fontsize)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_bond_energy_diagram(
    geometry: GeometryData,
    bond_rows: List[Dict[str, Any]],
    filepath: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    positions = _geometry_positions(geometry)
    n_sites = int(getattr(geometry, "number_of_sites", positions.shape[0]))
    bond_width = _scaled_value(n_sites, 2.2, min_value=0.9, max_value=3.0)
    site_size = _scaled_value(n_sites, 10.0, min_value=4.5, max_value=18.0, exponent=0.34)
    title_fontsize = _scaled_value(n_sites, 12.0, min_value=8.8, max_value=13.0, exponent=0.12)
    label_fontsize = _scaled_value(n_sites, 10.0, min_value=7.2, max_value=11.4, exponent=0.14)
    legend_fontsize = _scaled_value(n_sites, 8.0, min_value=6.0, max_value=9.3, exponent=0.14)
    colorbar_fontsize = _scaled_value(n_sites, 9.0, min_value=6.5, max_value=10.0, exponent=0.14)
    offset_spacing = 0.018 * _position_span(positions)
    segments_by_channel: Dict[str, List[List[np.ndarray]]] = {}
    values_by_channel: Dict[str, List[float]] = {}
    for row in bond_rows:
        i, j = int(row["i"]), int(row["j"])
        p_i = positions[i]
        p_j = positions[j]
        channel_values = _plottable_bond_channel_values(row)
        channels = _ordered_channels(list(channel_values.keys()))
        for channel, offset in zip(channels, _centered_offsets(len(channels), offset_spacing)):
            segments_by_channel.setdefault(channel, []).append(_offset_segment(p_i, p_j, offset))
            values_by_channel.setdefault(channel, []).append(float(channel_values[channel]))

    all_values = [
        value
        for channel_values in values_by_channel.values()
        for value in channel_values
    ]
    if len(all_values) == 0:
        raise RuntimeError("No bond-energy links available to plot.")
    values_arr = np.asarray(all_values, dtype=float)
    v_min = float(np.min(values_arr))
    v_max = float(np.max(values_arr))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        raise RuntimeError("Bond-energy values contain non-finite entries.")
    if v_min == v_max:
        pad = max(1e-8, 0.08 * max(abs(v_min), 1e-6))
        v_min -= pad
        v_max += pad
    norm = Normalize(vmin=v_min, vmax=v_max)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    collections = []
    for channel in _ordered_channels(list(segments_by_channel.keys())):
        channel_values = np.asarray(values_by_channel[channel], dtype=float)
        collection = LineCollection(
            segments_by_channel[channel],
            cmap=BOND_ENERGY_CMAP,
            norm=norm,
            linewidths=bond_width,
            linestyles=CHANNEL_LINESTYLES.get(channel, "solid"),
            alpha=0.95,
        )
        collection.set_array(channel_values)
        ax.add_collection(collection)
        collections.append(collection)
    ax.scatter(positions[:, 0], positions[:, 1], c="black", s=site_size, zorder=3)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=CHANNEL_LINESTYLES.get(channel, "solid"),
            linewidth=max(1.2, bond_width),
            label=CHANNEL_LABELS.get(channel, channel),
        )
        for channel in _ordered_channels(list(segments_by_channel.keys()))
    ]
    if len(legend_handles) > 0:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(4, max(1, len(legend_handles))),
            fontsize=legend_fontsize,
            framealpha=0.94,
        )
    ax.autoscale()
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("x", fontsize=label_fontsize)
    ax.set_ylabel("y", fontsize=label_fontsize)
    ax.tick_params(labelsize=max(6.0, label_fontsize - 1.5))
    ax.set_aspect("equal", adjustable="datalim")
    cbar = fig.colorbar(collections[0], ax=ax, shrink=0.9)
    cbar.set_label("Channel bond-energy contribution", fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=max(6.0, colorbar_fontsize - 1.0))
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def save_structure_factor_plot(rows: List[Dict[str, Any]], filepath: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["Q_label"] for row in rows]
    s_vals = [float(row["S(Q)"]) for row in rows]
    t_vals = [float(row["T(Q)"]) for row in rows]
    st_vals = [float(row["ST(Q)"]) for row in rows]

    x = np.arange(len(labels), dtype=float)
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.bar(x - w, s_vals, width=w, label="S(Q)")
    ax.bar(x, t_vals, width=w, label="T(Q)")
    ax.bar(x + w, st_vals, width=w, label="ST(Q)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("High-symmetry momentum")
    ax.set_ylabel("Structure factor")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_scalar_correlation_heatmaps(
    scalar_correlations: Dict[str, np.ndarray],
    filepath: str,
    title_prefix: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    keys = ("S", "T", "ST")
    for ax, key in zip(axes, keys):
        matrix = np.real(scalar_correlations[key])
        image = ax.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(f"{title_prefix} {key}_ij")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def _correlation_site_color(value: float, scale: float) -> Tuple[float, float, float, float]:
    if not np.isfinite(value):
        return (0.45, 0.45, 0.45, 0.25)
    if scale <= 1e-14 or abs(value) <= 1e-14:
        return (0.45, 0.45, 0.45, 0.22)
    alpha = min(1.0, max(0.12, abs(float(value)) / float(scale)))
    if value > 0.0:
        return (0.84, 0.15, 0.16, alpha)
    return (0.12, 0.47, 0.71, alpha)


def plot_real_space_pattern(
    geometry: GeometryData,
    correlation_array: np.ndarray,
    reference_site_idx: int,
    ax: Any | None = None,
    title: str | None = None,
    channel_label: str = "correlation",
):
    """Draw the reference-site real-space ordering pattern on the lattice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    positions = _geometry_positions(geometry)
    values = np.real_if_close(np.asarray(correlation_array)).astype(float).reshape(-1)
    n_sites = int(getattr(geometry, "number_of_sites", len(values)))
    if values.size != n_sites:
        raise ValueError(f"correlation_array has length {values.size}, but geometry has {n_sites} sites.")
    reference_site_idx = int(reference_site_idx)
    if reference_site_idx < 0:
        reference_site_idx = n_sites + reference_site_idx
    if reference_site_idx < 0 or reference_site_idx >= n_sites:
        raise IndexError(f"reference_site_idx={reference_site_idx} is outside [0, {n_sites - 1}].")

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    else:
        fig = ax.figure

    for bond in geometry.bond_list:
        site_i, site_j, gamma = _bond_i_j_gamma(bond)
        p_i = positions[site_i]
        p_j = positions[site_j]
        ax.plot(
            [p_i[0], p_j[0]],
            [p_i[1], p_j[1]],
            color=GAMMA_COLORS.get(gamma, "#9a9a9a"),
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )

    other_sites = np.asarray([site for site in range(n_sites) if site != reference_site_idx], dtype=int)
    other_values = values[other_sites] if other_sites.size > 0 else np.asarray([], dtype=float)
    scale = float(np.max(np.abs(other_values))) if other_values.size > 0 else 0.0
    if not np.isfinite(scale) or scale <= 1e-14:
        scale = float(np.max(np.abs(values))) if values.size > 0 else 1.0
    if not np.isfinite(scale) or scale <= 1e-14:
        scale = 1.0

    colors = [_correlation_site_color(float(values[site]), scale) for site in other_sites]
    if other_sites.size > 0:
        ax.scatter(
            positions[other_sites, 0],
            positions[other_sites, 1],
            s=95,
            c=colors,
            edgecolors="#222222",
            linewidths=0.6,
            zorder=3,
        )

    reference_position = positions[reference_site_idx]
    ax.scatter(
        [reference_position[0]],
        [reference_position[1]],
        s=250,
        marker="*",
        c="#ffd92f",
        edgecolors="#111111",
        linewidths=1.0,
        zorder=5,
        label=f"reference site {reference_site_idx}",
    )
    ax.text(
        reference_position[0],
        reference_position[1],
        str(reference_site_idx),
        fontsize=7,
        ha="center",
        va="center",
        color="#111111",
        zorder=6,
    )

    vmax = max(scale, 1e-12)
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap="coolwarm"),
        ax=ax,
        shrink=0.86,
    )
    colorbar.set_label(channel_label)
    legend_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd92f", markeredgecolor="#111111",
               markersize=12, label="reference site"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="#222222",
               markersize=8, label="positive relative alignment"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markeredgecolor="#222222",
               markersize=8, label="negative relative alignment"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    if created_figure:
        fig.tight_layout()
    return fig, ax


def save_real_space_pattern_diagram(
    geometry: GeometryData,
    correlation_array: np.ndarray,
    reference_site_idx: int,
    filepath: str,
    title: str,
    channel_label: str,
) -> None:
    fig, _ = plot_real_space_pattern(
        geometry,
        correlation_array,
        reference_site_idx,
        title=title,
        channel_label=channel_label,
    )
    fig.savefig(filepath)
    import matplotlib.pyplot as plt
    plt.close(fig)


def save_phase_representative_pattern(
    geometry: GeometryData,
    spin_correlation_array: np.ndarray,
    reference_site_idx: int,
    bond_rows: List[Dict[str, Any]] | None,
    filepath: str,
    title: str,
    external_field_vector: Tuple[float, float, float] | None = None,
) -> None:
    """Save one compact phase-representative plot with spin arrows and resolved bonds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects

    positions = _geometry_positions(geometry)
    values = np.real_if_close(np.asarray(spin_correlation_array)).astype(float).reshape(-1)
    n_sites = int(getattr(geometry, "number_of_sites", len(values)))
    if values.size != n_sites:
        raise ValueError(f"spin_correlation_array has length {values.size}, but geometry has {n_sites} sites.")
    reference_site_idx = int(reference_site_idx)
    if reference_site_idx < 0:
        reference_site_idx = n_sites + reference_site_idx
    if reference_site_idx < 0 or reference_site_idx >= n_sites:
        raise IndexError(f"reference_site_idx={reference_site_idx} is outside [0, {n_sites - 1}].")

    size_scale = _lattice_size_scale(n_sites)
    background_bond_width = _scaled_value(n_sites, 0.8, min_value=0.32, max_value=1.05)
    resolved_bond_base_width = _scaled_value(n_sites, 0.85, min_value=0.45, max_value=1.15)
    resolved_bond_dynamic_width = _scaled_value(n_sites, 2.25, min_value=1.15, max_value=2.8)
    site_size = _scaled_value(n_sites, 32.0, min_value=13.0, max_value=52.0, exponent=0.34)
    site_edge_width = _scaled_value(n_sites, 0.7, min_value=0.35, max_value=1.0, exponent=0.18)
    title_fontsize = _scaled_value(n_sites, 12.0, min_value=8.6, max_value=13.0, exponent=0.12)
    label_fontsize = _scaled_value(n_sites, 10.0, min_value=7.2, max_value=11.4, exponent=0.14)
    legend_fontsize = _scaled_value(n_sites, 8.0, min_value=6.0, max_value=9.3, exponent=0.14)
    colorbar_fontsize = _scaled_value(n_sites, 9.0, min_value=6.5, max_value=10.0, exponent=0.14)
    arrow_width = float(np.clip(0.014 * size_scale, 0.009, 0.020))
    arrow_halo_width = float(np.clip(1.75 * arrow_width, 0.016, 0.034))
    arrow_headwidth = float(np.clip(5.4 * size_scale, 4.2, 7.0))
    arrow_headlength = float(np.clip(6.6 * size_scale, 5.0, 8.4))

    fig, ax = plt.subplots(figsize=(8.8, 6.4), dpi=150)

    for bond in geometry.bond_list:
        site_i, site_j, gamma = _bond_i_j_gamma(bond)
        p_i = positions[site_i]
        p_j = positions[site_j]
        ax.plot(
            [p_i[0], p_j[0]],
            [p_i[1], p_j[1]],
            color=GAMMA_COLORS.get(gamma, "#bdbdbd"),
            linewidth=background_bond_width,
            alpha=0.28,
            zorder=BACKGROUND_BOND_ZORDER,
        )

    rows = [row for row in (bond_rows or []) if isinstance(row, dict)]
    offset_spacing = 0.024 * _position_span(positions)
    channel_order = ("S", "T", "ST", "total")
    segments_by_channel: Dict[str, List[List[np.ndarray]]] = {}
    values_by_channel: Dict[str, List[float]] = {}
    for row in rows:
        try:
            i, j = int(row["i"]), int(row["j"])
        except (KeyError, TypeError, ValueError):
            continue
        if i < 0 or j < 0 or i >= positions.shape[0] or j >= positions.shape[0]:
            continue
        channel_values = _bond_row_channel_values(row)
        channels = [
            channel for channel in channel_order
            if channel in channel_values and np.isfinite(float(channel_values[channel]))
        ]
        if len(channels) == 0:
            continue
        for channel, offset in zip(channels, _centered_offsets(len(channels), offset_spacing)):
            segments_by_channel.setdefault(channel, []).append(_offset_segment(positions[i], positions[j], offset))
            values_by_channel.setdefault(channel, []).append(float(channel_values[channel]))

    all_bond_values = [
        value
        for channel_values in values_by_channel.values()
        for value in channel_values
        if np.isfinite(float(value))
    ]
    bond_abs_max = max([abs(value) for value in all_bond_values], default=0.0)
    bond_norm = Normalize(vmin=-bond_abs_max, vmax=bond_abs_max) if bond_abs_max > 1e-14 else Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap(BOND_ENERGY_CMAP)
    first_collection = None
    for channel in [channel for channel in channel_order if channel in segments_by_channel]:
        channel_values = np.asarray(values_by_channel[channel], dtype=float)
        widths = resolved_bond_base_width + resolved_bond_dynamic_width * (
            np.abs(channel_values) / max(bond_abs_max, 1e-12)
        )
        collection = LineCollection(
            segments_by_channel[channel],
            cmap=cmap,
            norm=bond_norm,
            linewidths=widths,
            linestyles=CHANNEL_LINESTYLES.get(channel, "solid"),
            alpha=0.92,
            zorder=RESOLVED_BOND_ZORDER,
        )
        collection.set_array(channel_values)
        ax.add_collection(collection)
        if first_collection is None:
            first_collection = collection

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=site_size,
        c="#f7f7f7",
        edgecolors="#222222",
        linewidths=site_edge_width,
        zorder=SITE_MARKER_ZORDER,
    )

    non_reference_sites = np.asarray([site for site in range(n_sites) if site != reference_site_idx], dtype=int)
    non_reference_values = values[non_reference_sites] if non_reference_sites.size > 0 else np.asarray([], dtype=float)
    spin_scale = (
        float(np.max(np.abs(non_reference_values)))
        if non_reference_values.size > 0
        else float(np.max(np.abs(values)))
    )
    if not np.isfinite(spin_scale) or spin_scale <= 1e-14:
        spin_scale = 1.0
    arrow_base = 0.62 * _position_span(positions) / max(np.sqrt(max(n_sites, 1)), 2.0)

    pattern_values = np.array(values, dtype=float, copy=True)
    if 0 <= reference_site_idx < n_sites and np.isfinite(pattern_values[reference_site_idx]):
        # The selected row only fixes the relative sign pattern. Draw the row site
        # as an ordinary +direction arrow rather than marking it as special.
        pattern_values[reference_site_idx] = spin_scale

    for site in range(n_sites):
        value = float(pattern_values[site])
        if not np.isfinite(value) or abs(value) <= 1e-14:
            continue
        sign = 1.0 if value > 0.0 else -1.0
        magnitude = min(1.0, abs(value) / spin_scale)
        length = arrow_base * (0.45 + 0.75 * magnitude)
        vector = np.asarray([sign * length, 0.0], dtype=float)
        start = positions[site] - 0.5 * vector
        color = "#d62728" if value > 0.0 else "#1f77b4"
        halo = ax.quiver(
            [start[0]],
            [start[1]],
            [vector[0]],
            [vector[1]],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="#ffffff",
            width=arrow_halo_width,
            headwidth=arrow_headwidth + 0.6,
            headlength=arrow_headlength + 0.6,
            headaxislength=arrow_headwidth + 0.6,
            pivot="tail",
            zorder=SPIN_VECTOR_HALO_ZORDER,
            clip_on=False,
        )
        halo.set_path_effects([
            path_effects.Stroke(linewidth=1.2, foreground="#ffffff"),
            path_effects.Normal(),
        ])
        arrow = ax.quiver(
            [start[0]],
            [start[1]],
            [vector[0]],
            [vector[1]],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=arrow_width,
            headwidth=arrow_headwidth,
            headlength=arrow_headlength,
            headaxislength=arrow_headwidth,
            pivot="tail",
            zorder=SPIN_VECTOR_ZORDER,
            clip_on=False,
        )
        arrow.set_path_effects([
            path_effects.Stroke(linewidth=0.5, foreground="#ffffff"),
            path_effects.Normal(),
        ])

    legend_handles = [
        Line2D([0], [0], color="#d62728", linewidth=max(2.0, resolved_bond_base_width + 1.0), marker=">", markevery=[1],
               label="spin pattern: + direction"),
        Line2D([0], [0], color="#1f77b4", linewidth=max(2.0, resolved_bond_base_width + 1.0), marker="<", markevery=[1],
               label="spin pattern: - direction"),
    ]
    for channel in [channel for channel in channel_order if channel in segments_by_channel]:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=CHANNEL_LINESTYLES.get(channel, "solid"),
                linewidth=max(1.4, resolved_bond_base_width + 1.0),
                label=f"{CHANNEL_LABELS.get(channel, channel)} bond",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(4, max(1, len(legend_handles))),
        fontsize=legend_fontsize,
        framealpha=0.94,
    )
    if first_collection is not None:
        colorbar = fig.colorbar(first_collection, ax=ax, shrink=0.84)
        colorbar.set_label("Resolved bond-energy contribution", fontsize=colorbar_fontsize)
        colorbar.ax.tick_params(labelsize=max(6.0, colorbar_fontsize - 1.0))

    _annotate_field_direction(ax, external_field_vector, fontsize=legend_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("x", fontsize=label_fontsize)
    ax.set_ylabel("y", fontsize=label_fontsize)
    ax.tick_params(labelsize=max(6.0, label_fontsize - 1.5))
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _plaquette_flux_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("W_p", value.get("value"))
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def _ordered_polygon_vertices(site_positions: np.ndarray) -> np.ndarray:
    vertices = np.asarray(site_positions, dtype=float)
    if vertices.shape != (6, 2):
        raise ValueError(f"Plaquette vertices must have shape (6, 2); got {vertices.shape}.")
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)]


def save_flux_crystal_pattern(
    geometry: GeometryData,
    all_fluxes_dict: Dict[Any, Any],
    filepath: str,
    title: str,
) -> None:
    """Draw a real-space plaquette-flux pattern for vison-crystal diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon

    positions = _geometry_positions(geometry)
    plaquettes = honeycomb_plaquette_flux_operators(geometry)
    plaquette_by_index = {
        int(plaquette["plaquette_index"]): plaquette
        for plaquette in plaquettes
    }
    if not isinstance(all_fluxes_dict, dict) or len(all_fluxes_dict) == 0:
        raise RuntimeError("No plaquette-flux values were provided.")

    patches: List[Polygon] = []
    values: List[float] = []
    for raw_index, raw_value in all_fluxes_dict.items():
        try:
            plaquette_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        plaquette = plaquette_by_index.get(plaquette_index)
        if plaquette is None:
            continue
        flux_value = _plaquette_flux_value(raw_value)
        if flux_value is None:
            continue
        sites = [int(site) for site in plaquette["sites"]]
        vertices = _ordered_polygon_vertices(positions[sites])
        patches.append(Polygon(vertices, closed=True))
        values.append(float(flux_value))

    if len(patches) == 0:
        raise RuntimeError("No provided plaquette-flux indices matched valid honeycomb plaquettes.")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for bond in geometry.bond_list:
        site_i, site_j, gamma = _bond_i_j_gamma(bond)
        p_i = positions[site_i]
        p_j = positions[site_j]
        ax.plot(
            [p_i[0], p_j[0]],
            [p_i[1], p_j[1]],
            color=GAMMA_COLORS.get(gamma, "#9a9a9a"),
            linewidth=0.8,
            alpha=0.32,
            zorder=1,
        )

    collection = PatchCollection(
        patches,
        cmap="RdBu_r",
        norm=Normalize(vmin=-1.0, vmax=1.0, clip=True),
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.88,
        zorder=2,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=16,
        c="#222222",
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.86)
    colorbar.set_label("Plaquette Flux W_p")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.margins(0.08)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_real_space_correlation_pattern(
    geometry: GeometryData,
    C_S: np.ndarray,
    reference_site_idx: int,
    output_path: str,
) -> None:
    """Save the spin reference-site correlation pattern on the honeycomb lattice."""
    fig, _ = plot_real_space_pattern(
        geometry,
        C_S,
        reference_site_idx,
        title="DMRG Reference-Site Spin Correlation Pattern",
        channel_label="C_S[j] = <S_ref . S_j>",
    )
    fig.savefig(output_path)
    import matplotlib.pyplot as plt
    plt.close(fig)


def save_multi_method_energy_comparison(
    method_to_energy: Dict[str, float],
    filepath: str,
    title: str = "Ground-State Energy Per Site Comparison",
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: List[str] = []
    values: List[float] = []
    for label in _ordered_available_methods(method_to_energy):
        value = float(method_to_energy[label])
        if np.isfinite(value):
            labels.append(label)
            values.append(value)
    if len(labels) == 0:
        raise RuntimeError("No finite method energies available for comparison.")

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    x_values = np.arange(len(labels), dtype=float)
    bar_colors = [METHOD_COLORS.get(label, "#666666") for label in labels]
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = y_max - y_min
    if not np.isfinite(span) or span <= 0.0:
        center = 0.5 * (y_min + y_max)
        pad = max(1e-8, 0.08 * max(abs(center), 1e-6))
        y_min = center - pad
        y_max = center + pad
    else:
        pad = max(1e-8, 0.18 * span)
        y_min -= pad
        y_max += pad
    bars = ax.bar(
        x_values,
        [value - y_min for value in values],
        bottom=y_min,
        width=0.62,
        color=bar_colors,
        alpha=0.9,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.legend(bars, labels, loc="best")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, max(float(len(labels) - 1), 0.0) + 0.55)
    ax.set_title(titled_for_run(title, title_label))
    ax.set_ylabel("Energy per site")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(y_min, y_max)
    y_span = y_max - y_min
    for x_value, value in zip(x_values, values):
        text_y = value + 0.035 * y_span
        va = "bottom"
        if text_y > y_max - 0.02 * y_span:
            text_y = value - 0.035 * y_span
            va = "top"
        ax.text(
            x_value,
            text_y,
            f"{value:.8g}",
            ha="center",
            va=va,
            fontsize=8,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def _comparison_rows(payload: Any, method: str) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = [row for row in payload["rows"] if isinstance(row, dict)]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = []
    output: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("status", "completed")) not in ("completed", "completed_with_warnings"):
            continue
        energy = row.get("ground_state_energy_per_site", row.get("energy_per_site"))
        try:
            energy_value = float(energy)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(energy_value):
            continue
        normalized = dict(row)
        normalized["method"] = str(row.get("method", row.get("backend", method)))
        normalized["energy_per_site"] = energy_value
        output.append(normalized)
    return output


def _comparison_alpha(row: Dict[str, Any], fallback: float | None = None) -> float:
    value = row.get("alpha", fallback)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return parsed if np.isfinite(parsed) else 0.0


def _comparison_correlation_value(row: Dict[str, Any]) -> float | None:
    for container in (row, row.get("diagnostics"), row.get("phase_observables")):
        if not isinstance(container, dict):
            continue
        diagnostics = container.get("diagnostics") if isinstance(container.get("diagnostics"), dict) else container
        for key in ("average_spin_dot", "spin_order_strength", "nearest_neighbor_spin_correlation"):
            value = diagnostics.get(key)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                return parsed
    bond_rows = row.get("bond_energies") or row.get("resolved_bond_observables")
    if isinstance(bond_rows, list):
        values: List[float] = []
        for bond in bond_rows:
            if not isinstance(bond, dict):
                continue
            value = bond.get("spin_dot", bond.get("S"))
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                values.append(parsed)
        if values:
            return float(np.mean(values))
    return None


def plot_peps_vs_ed_comparison(
    peps_results: Any,
    ed_results: Any,
    filepath: str,
    title: str = "PEPS vs ED Benchmark",
    title_label: str | None = None,
    peps_label: str = "PEPS",
    ed_label: str = "ED",
) -> None:
    """Overlay PEPS/iPEPS and ED energy/correlation benchmarks versus alpha."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peps_rows = _comparison_rows(peps_results, peps_label)
    ed_rows = _comparison_rows(ed_results, ed_label)
    if not peps_rows and not ed_rows:
        raise RuntimeError("No completed PEPS/iPEPS or ED rows available for comparison.")

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), dpi=150, sharex=True)
    series = ((peps_label, peps_rows), (ed_label, ed_rows))
    plotted_correlation = False
    for label, rows in series:
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda row: _comparison_alpha(row))
        alphas = np.asarray([_comparison_alpha(row) for row in rows_sorted], dtype=float)
        energies = np.asarray([float(row["energy_per_site"]) for row in rows_sorted], dtype=float)
        marker = METHOD_MARKERS.get(label, METHOD_MARKERS.get(str(label).lower(), "o"))
        color = METHOD_COLORS.get(label, METHOD_COLORS.get(str(label).lower(), "#444444"))
        axes[0].plot(
            alphas,
            energies,
            marker=marker,
            linestyle=METHOD_LINESTYLES.get(label, "-"),
            color=color,
            linewidth=1.4,
            markersize=8 if marker == "*" else 5,
            label=label,
        )
        corr_pairs = [
            (_comparison_alpha(row), _comparison_correlation_value(row))
            for row in rows_sorted
        ]
        corr_pairs = [(alpha, value) for alpha, value in corr_pairs if value is not None]
        if corr_pairs:
            corr_alpha = np.asarray([alpha for alpha, _value in corr_pairs], dtype=float)
            corr_values = np.asarray([float(value) for _alpha, value in corr_pairs], dtype=float)
            axes[1].plot(
                corr_alpha,
                corr_values,
                marker=marker,
                linestyle=METHOD_LINESTYLES.get(label, "-"),
                color=color,
                linewidth=1.4,
                markersize=8 if marker == "*" else 5,
                label=label,
            )
            plotted_correlation = True
    axes[0].set_title(titled_for_run(title, title_label))
    axes[0].set_ylabel("Energy per site")
    axes[1].set_ylabel("Average <S_i . S_j>")
    axes[1].set_xlabel("alpha")
    if not plotted_correlation:
        axes[1].text(
            0.5,
            0.5,
            "No correlation data",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            color="#555555",
        )
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if len(handles) > 0:
            axis.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_peps_vs_ed_comparison(*args: Any, **kwargs: Any) -> None:
    plot_peps_vs_ed_comparison(*args, **kwargs)


def save_entropy_profiles_comparison(
    entropy_profiles: Dict[str, Dict[str, Any]],
    filepath: str,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, sharex=True)
    axes_flat = list(axes.flatten())
    method_order = [
        method for method in _ordered_available_methods(entropy_profiles)
        if method != "iDMRG-x"
    ]

    for axis, order_n in zip(axes_flat, orders):
        key = f"S{order_n}"
        plotted = False
        for method in method_order:
            profile = entropy_profiles.get(method, None)
            if profile is None:
                continue
            values = profile.get("entropies", {}).get(key, [])
            x_values = profile.get("cuts_normalized", [])
            if len(values) == 0 or len(x_values) != len(values):
                continue
            axis.plot(
                x_values,
                values,
                marker=METHOD_MARKERS.get(method, "o"),
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                linewidth=1.8,
                markersize=3.5,
                color=METHOD_COLORS.get(method, None),
                label=method,
            )
            plotted = True
        axis.set_title(f"Renyi Entropy n={order_n}")
        axis.set_xlabel("Normalized cut position")
        axis.set_ylabel("Entropy")
        axis.grid(alpha=0.25)
        if not plotted:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if len(handles) > 0:
        axes_flat[0].legend(loc="best")
    fig.suptitle(titled_for_run("Entanglement Entropy Profiles by Method", title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_entropy_method_means_comparison(
    entropy_profiles: Dict[str, Dict[str, Any]],
    filepath: str,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = [
        method for method in _ordered_available_methods(entropy_profiles)
        if method != "iDMRG-x"
    ]
    if len(method_order) == 0:
        raise RuntimeError("No entropy profiles available for method-mean comparison.")

    x = np.arange(len(orders), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    plotted = False
    for method in method_order:
        profile = entropy_profiles[method]
        summary = profile.get("summary", {})
        means = np.asarray([float(summary.get(f"S{order_n}_mean", np.nan)) for order_n in orders], dtype=float)
        valid = np.isfinite(means)
        if not np.any(valid):
            continue
        ax.plot(
            x[valid],
            means[valid],
            marker=METHOD_MARKERS.get(method, "o"),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            linewidth=1.8,
            markersize=4.5,
            label=method,
            color=METHOD_COLORS.get(method, "#666666"),
        )
        plotted = True
    if not plotted:
        raise RuntimeError("No finite entropy means available for method comparison.")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={order_n}" for order_n in orders])
    ax.set_xlabel("Renyi order")
    ax.set_ylabel("Mean entropy across cuts")
    ax.set_title(titled_for_run("Method Comparison: Mean Entanglement Entropies", title_label))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_dmrg_ed_energy_comparison(
    dmrg_energy: float,
    ed_energy: float,
    filepath: str,
    title_label: str | None = None,
) -> None:
    save_multi_method_energy_comparison(
        method_to_energy={"DMRG": float(dmrg_energy), "ED": float(ed_energy)},
        filepath=filepath,
        title="Ground-State Energy Comparison",
        title_label=title_label,
    )


def save_low_energy_spectrum_comparison(
    method_spectra: Dict[str, Dict[str, Any]],
    filepath: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = _ordered_available_methods(method_spectra)
    if len(method_order) == 0:
        raise RuntimeError("No low-energy spectrum data available for comparison.")

    def degeneracy_label(method: str) -> str:
        entry = method_spectra.get(method, {})
        value = entry.get("ground_state_degeneracy")
        status = str(entry.get("ground_state_degeneracy_status", "")).lower()
        if status == "not_checked" or entry.get("ground_state_degeneracy_check_enabled") is False:
            return "g=off"
        lower_bound = bool(entry.get("ground_state_degeneracy_is_lower_bound", False))
        try:
            degeneracy = int(value)
        except (TypeError, ValueError):
            return "g=?"
        if status == "ed_guided":
            return f"g~{degeneracy}"
        prefix = "g>=" if lower_bound or status == "lower_bound" else "g="
        return f"{prefix}{degeneracy}"

    def value_label(value: float) -> str:
        abs_value = abs(float(value))
        if abs_value > 0.0 and (abs_value < 1e-4 or abs_value >= 1e4):
            return f"{float(value):.3e}"
        return f"{float(value):.8g}"

    def set_focused_ylim(axis: Any, values: List[float]) -> Tuple[float, float]:
        if len(values) == 0:
            axis.set_ylim(-0.5, 0.5)
            return -0.5, 0.5
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        span = y_max - y_min
        if not np.isfinite(span) or span <= 0.0:
            center = 0.5 * (y_min + y_max)
            pad = max(1e-8, 0.08 * max(abs(center), 1e-6))
            y_min = center - pad
            y_max = center + pad
        else:
            pad = max(1e-8, 0.18 * span)
            y_min -= pad
            y_max += pad
        axis.set_ylim(y_min, y_max)
        return y_min, y_max

    finite_spectrum_method_order = [
        method for method in method_order if method != "iDMRG-x"
    ]
    panel_specs = [
        ("ground_state_energy_per_site", "Ground energy/site", method_order),
        ("first_excited_energy_per_site", "First excited energy/site", finite_spectrum_method_order),
        ("spectral_gap", "Gap E1 - E0", finite_spectrum_method_order),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150)
    for axis, (key, label, panel_methods) in zip(axes, panel_specs):
        x_values = np.arange(len(panel_methods), dtype=float)
        x_tick_labels = [f"{method}\n{degeneracy_label(method)}" for method in panel_methods]
        finite_x: List[float] = []
        finite_values: List[float] = []
        missing: List[Tuple[float, str]] = []
        for x_value, method in zip(x_values, panel_methods):
            value = method_spectra.get(method, {}).get(key)
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                missing.append((float(x_value), method))
                continue
            if not np.isfinite(value_float):
                missing.append((float(x_value), method))
                continue
            finite_x.append(float(x_value))
            finite_values.append(value_float)
        axis.set_xticks(x_values)
        axis.set_xticklabels(x_tick_labels, rotation=0)
        axis.set_title(label)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlim(-0.45, max(0.45, len(panel_methods) - 0.55))
        y_min, y_max = set_focused_ylim(axis, finite_values)
        y_span = y_max - y_min
        if finite_values:
            finite_pairs = sorted(zip(finite_x, finite_values), key=lambda pair: pair[0])
            for x_value, value_float in finite_pairs:
                method = panel_methods[int(round(x_value))]
                axis.bar(
                    [x_value],
                    [value_float - y_min],
                    bottom=y_min,
                    width=0.58,
                    color=METHOD_COLORS.get(method, "#666666"),
                    alpha=0.9,
                    edgecolor="#222222",
                    linewidth=0.8,
                    zorder=3,
                )
        for x_value, value_float in zip(finite_x, finite_values):
            vertical_offset = 0.035 * y_span
            text_y = value_float + vertical_offset
            va = "bottom"
            if text_y > y_max - 0.02 * y_span:
                text_y = value_float - vertical_offset
                va = "top"
            axis.text(
                x_value,
                text_y,
                value_label(value_float),
                ha="center",
                va=va,
                fontsize=8,
                color="#333333",
            )
        if missing:
            text_y = y_min + 0.08 * y_span if y_max > y_min else 0.0
            for x_value, _method in missing:
                axis.text(
                    x_value,
                    text_y,
                    "not found",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#666666",
                    rotation=90,
                )
    if "iDMRG-x" in method_order:
        base_title = "Ground-State and Low-Energy Spectrum Comparison"
    elif set(method_order).issubset({"DMRG", "ED"}):
        base_title = "Finite DMRG vs ED Low-Energy Spectrum"
    else:
        base_title = "Low-Energy Spectrum Comparison"
    fig.suptitle(titled_for_run(base_title, title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_energy_b_scan_plot(
    scan_data: Dict[str, Any],
    filepath: str,
    title: str = "Energy vs External Field",
    title_label: str | None = None,
) -> None:
    """Overlay DMRG ground-state energy and ED low-energy bands versus B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row for row in list(scan_data.get("rows", []))
        if isinstance(row, dict) and str(row.get("status", "completed")) == "completed"
    ]
    if len(rows) == 0:
        raise RuntimeError("No completed Energy-B scan rows available to plot.")
    rows = sorted(rows, key=lambda row: float(row.get("field_strength", row.get("B", 0.0))))
    fields = np.asarray([float(row.get("field_strength", row.get("B", 0.0))) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    max_band_count = max(
        [
            len(row.get("ed_energies_per_site", []) or [])
            for row in rows
        ],
        default=0,
    )
    for band_index in range(max_band_count):
        band_values: List[float] = []
        band_fields: List[float] = []
        for field_value, row in zip(fields, rows):
            energies = row.get("ed_energies_per_site", []) or []
            if band_index >= len(energies):
                continue
            value = float(energies[band_index])
            if np.isfinite(value):
                band_fields.append(float(field_value))
                band_values.append(value)
        if band_values:
            ax.plot(
                band_fields,
                band_values,
                color="#5f6b7a",
                alpha=0.50 if band_index > 0 else 0.85,
                linewidth=1.0 if band_index > 0 else 1.5,
                label="ED bands" if band_index == 0 else None,
            )

    dmrg_fields: List[float] = []
    dmrg_values: List[float] = []
    for field_value, row in zip(fields, rows):
        value = row.get("dmrg_energy_per_site")
        if value is None:
            continue
        value_float = float(value)
        if np.isfinite(value_float):
            dmrg_fields.append(float(field_value))
            dmrg_values.append(value_float)
    if dmrg_values:
        ax.plot(
            dmrg_fields,
            dmrg_values,
            color="#7b3294",
            marker="*",
            markersize=9,
            linewidth=1.2,
            label="DMRG ground state",
            zorder=4,
        )

    ax.set_xlabel(r"External field strength $B$")
    ax.set_ylabel("Energy per site")
    ax.set_title(titled_for_run(title, title_label))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_multi_method_structure_comparison(
    method_to_rows: Dict[str, List[Dict[str, Any]]],
    filepath: str,
    title_label: str | None = None,
    title: str = "Finite DMRG vs ED vs iDMRG-x Structure Factors",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = _ordered_available_methods(method_to_rows)
    if len(method_order) == 0:
        raise RuntimeError("No method structure-factor rows available for comparison.")

    labels: List[str] = []
    rows_by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method in method_order:
        row_map: Dict[str, Dict[str, Any]] = {}
        for row in method_to_rows.get(method, []):
            label = str(row["Q_label"])
            row_map[label] = row
            if label not in labels:
                labels.append(label)
        rows_by_method[method] = row_map
    if len(labels) == 0:
        raise RuntimeError("No high-symmetry momentum labels available for structure-factor comparison.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150, sharex=True)
    channels = ("S(Q)", "T(Q)", "ST(Q)")
    for ax, channel in zip(axes, channels):
        plotted = False
        for method in method_order:
            method_labels = [
                label
                for label in labels
                if label in rows_by_method[method] and channel in rows_by_method[method][label]
            ]
            if len(method_labels) == 0:
                continue
            x_values = np.asarray([labels.index(label) for label in method_labels], dtype=float)
            values = np.asarray(
                [float(rows_by_method[method][label][channel]) for label in method_labels],
                dtype=float,
            )
            valid = np.isfinite(values)
            if not np.any(valid):
                continue
            ax.plot(
                x_values[valid],
                values[valid],
                marker=METHOD_MARKERS.get(method, "o"),
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                linewidth=1.8,
                markersize=3.5,
                color=METHOD_COLORS.get(method, None),
                label=method,
            )
            plotted = True
        ax.set_title(channel)
        ax.set_xticks(np.arange(len(labels), dtype=float))
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(alpha=0.25)
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        handles, legend_labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(handles, legend_labels, loc="best")
    axes[0].set_ylabel("Value")
    fig.suptitle(titled_for_run(title, title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_dmrg_ed_structure_comparison(
    dmrg_rows: List[Dict[str, Any]],
    ed_rows: List[Dict[str, Any]],
    filepath: str,
    title_label: str | None = None,
) -> None:
    save_multi_method_structure_comparison(
        method_to_rows={"DMRG": dmrg_rows, "ED": ed_rows},
        filepath=filepath,
        title_label=title_label,
        title="DMRG vs ED Structure Factors",
    )


def _finite_temperature_xscale(axis: Any, rows: List[Dict[str, Any]], thermal_data: Dict[str, Any]) -> None:
    grid = thermal_data.get("temperature_grid", {})
    if isinstance(grid, dict) and str(grid.get("scale", "")).lower() == "log":
        temperatures = [float(row["T"]) for row in rows if float(row["T"]) > 0.0]
        if len(temperatures) > 0:
            axis.set_xscale("log")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _thermal_references(thermal_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    references = thermal_data.get("zero_temperature_references", {})
    return references if isinstance(references, dict) else {}


def _reference_section(
    thermal_data: Dict[str, Any],
    method: str,
    section: str,
) -> Dict[str, Any]:
    method_ref = _thermal_references(thermal_data).get(method, {})
    if not isinstance(method_ref, dict):
        return {}
    section_data = method_ref.get(section, {})
    return section_data if isinstance(section_data, dict) else {}


def _add_reference_hline(
    axis: Any,
    value: Any,
    method: str,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = ":",
) -> bool:
    number = _finite_number(value)
    if number is None:
        return False
    axis.axhline(
        number,
        color=color or METHOD_COLORS.get(method, "#444444"),
        linestyle=linestyle,
        linewidth=1.7,
        alpha=0.9,
        label=label or f"{method} T=0",
    )
    return True


def save_finite_temperature_observables_plot(
    thermal_data: Dict[str, Any],
    filepath: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(thermal_data.get("observables", []))
    if len(rows) == 0:
        raise RuntimeError("No finite-temperature observable rows available to plot.")

    temperatures = np.asarray([float(row["T"]) for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150, sharex=True)
    axes_flat = list(axes.flatten())
    plot_specs = [
        ("energy_per_site", "Energy per site"),
        ("specific_heat_per_site", "Specific heat per site"),
        ("entropy_per_site", "Thermal entropy per site"),
    ]
    ed_gs_observables = _reference_section(thermal_data, "ED-GS", "observables")
    dmrg_observables = _reference_section(thermal_data, "DMRG", "observables")
    for axis, (key, label) in zip(axes_flat[:3], plot_specs):
        values = [float(row[key]) for row in rows]
        axis.plot(
            temperatures,
            values,
            marker=METHOD_MARKERS.get("ED", "o"),
            linewidth=1.8,
            markersize=3.5,
            color=METHOD_COLORS.get("ED"),
            label="ED finite T",
        )
        legend_needed = _add_reference_hline(
            axis,
            ed_gs_observables.get(key),
            "ED",
            label="ED GS T=0",
            color=METHOD_COLORS.get("ED"),
            linestyle="--",
        )
        if _add_reference_hline(axis, dmrg_observables.get(key), "DMRG"):
            legend_needed = True
        if legend_needed:
            axis.legend(loc="best")
        axis.set_title(label)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        _finite_temperature_xscale(axis, rows, thermal_data)

    axes_flat[3].plot(
        temperatures,
        [float(row["spin_z_per_site"]) for row in rows],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        color="#9467bd",
        label="ED Sz/site",
    )
    axes_flat[3].plot(
        temperatures,
        [float(row["orbital_z_per_site"]) for row in rows],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=3.5,
        color="#8c564b",
        label="ED Tz/site",
    )
    _add_reference_hline(
        axes_flat[3],
        ed_gs_observables.get("spin_z_per_site"),
        "ED",
        label="ED GS Sz/site T=0",
        color="#9467bd",
        linestyle="--",
    )
    _add_reference_hline(
        axes_flat[3],
        ed_gs_observables.get("orbital_z_per_site"),
        "ED",
        label="ED GS Tz/site T=0",
        color="#8c564b",
        linestyle="--",
    )
    _add_reference_hline(
        axes_flat[3],
        dmrg_observables.get("spin_z_per_site"),
        "DMRG",
        label="DMRG Sz/site T=0",
        color="#9467bd",
    )
    _add_reference_hline(
        axes_flat[3],
        dmrg_observables.get("orbital_z_per_site"),
        "DMRG",
        label="DMRG Tz/site T=0",
        color="#8c564b",
    )
    axes_flat[3].set_title("Uniform z moments")
    axes_flat[3].set_ylabel("Moment per site")
    axes_flat[3].grid(alpha=0.25)
    axes_flat[3].legend(loc="best")
    _finite_temperature_xscale(axes_flat[3], rows, thermal_data)

    for axis in axes_flat:
        axis.set_xlabel("Temperature T")
    fig.suptitle(titled_for_run("Finite-Temperature Observables: ED with T=0 References", title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_finite_temperature_correlations_plot(
    thermal_data: Dict[str, Any],
    filepath: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(thermal_data.get("correlations", []))
    if len(rows) == 0:
        raise RuntimeError("No finite-temperature correlation rows available to plot.")

    temperatures = np.asarray([float(row["T"]) for row in rows], dtype=float)
    ed_gs_correlations = _reference_section(thermal_data, "ED-GS", "correlations")
    dmrg_correlations = _reference_section(thermal_data, "DMRG", "correlations")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for key, label, marker, color in (
        ("nearest_neighbor_S", "S dot S", "o", "#1f77b4"),
        ("nearest_neighbor_T", "T dot T", "s", "#2ca02c"),
        ("nearest_neighbor_ST", "ST dot ST", "^", "#d62728"),
    ):
        axes[0].plot(
            temperatures,
            [float(row[key]) for row in rows],
            marker=marker,
            linewidth=1.8,
            markersize=3.5,
            color=color,
            label=f"ED {label}",
        )
        _add_reference_hline(
            axes[0],
            ed_gs_correlations.get(key),
            "ED",
            label=f"ED GS {label} T=0",
            color=color,
            linestyle="--",
        )
        _add_reference_hline(
            axes[0],
            dmrg_correlations.get(key),
            "DMRG",
            label=f"DMRG {label} T=0",
            color=color,
        )
    axes[0].set_title("Nearest-neighbor scalar correlations")
    axes[0].set_ylabel("Bond average")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    _finite_temperature_xscale(axes[0], rows, thermal_data)

    axes[1].plot(
        temperatures,
        [float(row["bond_energy_per_site"]) for row in rows],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        color="#d62728",
        label="ED finite T",
    )
    _add_reference_hline(
        axes[1],
        ed_gs_correlations.get("bond_energy_per_site"),
        "ED",
        label="ED GS T=0",
        color=METHOD_COLORS.get("ED"),
        linestyle="--",
    )
    _add_reference_hline(
        axes[1],
        dmrg_correlations.get("bond_energy_per_site"),
        "DMRG",
        label="DMRG T=0",
        color=METHOD_COLORS.get("DMRG"),
    )
    axes[1].set_title("Exchange bond energy")
    axes[1].set_ylabel("Bond energy per site")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    _finite_temperature_xscale(axes[1], rows, thermal_data)

    for axis in axes:
        axis.set_xlabel("Temperature T")
    fig.suptitle(titled_for_run("Finite-Temperature Correlations: ED with T=0 References", title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_finite_temperature_structure_factors_plot(
    thermal_data: Dict[str, Any],
    filepath: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(thermal_data.get("structure_factors", []))
    if len(rows) == 0:
        raise RuntimeError("No finite-temperature structure-factor rows available to plot.")

    labels: List[str] = []
    for row in rows:
        label = str(row["Q_label"])
        if label not in labels:
            labels.append(label)

    channels = ("S(Q)", "T(Q)", "ST(Q)")
    ed_gs_rows_raw = _thermal_references(thermal_data).get("ED-GS", {}).get("structure_factors", [])
    ed_gs_rows = ed_gs_rows_raw if isinstance(ed_gs_rows_raw, list) else []
    ed_gs_by_label = {
        str(row.get("Q_label")): row
        for row in ed_gs_rows
        if isinstance(row, dict) and "Q_label" in row
    }
    dmrg_rows_raw = _thermal_references(thermal_data).get("DMRG", {}).get("structure_factors", [])
    dmrg_rows = dmrg_rows_raw if isinstance(dmrg_rows_raw, list) else []
    dmrg_by_label = {
        str(row.get("Q_label")): row
        for row in dmrg_rows
        if isinstance(row, dict) and "Q_label" in row
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150, sharey=False)
    for axis, channel in zip(axes, channels):
        added_ed_gs_reference = False
        added_dmrg_reference = False
        for label in labels:
            label_rows = [row for row in rows if str(row["Q_label"]) == label]
            temperatures = np.asarray([float(row["T"]) for row in label_rows], dtype=float)
            values = [float(row[channel]) for row in label_rows]
            line = axis.plot(temperatures, values, marker="o", linewidth=1.5, markersize=3.0, label=label)[0]
            ed_gs_value = _finite_number(ed_gs_by_label.get(label, {}).get(channel))
            if ed_gs_value is not None:
                axis.axhline(
                    ed_gs_value,
                    color=line.get_color(),
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                )
                added_ed_gs_reference = True
            dmrg_value = _finite_number(dmrg_by_label.get(label, {}).get(channel))
            if dmrg_value is not None:
                axis.axhline(
                    dmrg_value,
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.85,
                )
                added_dmrg_reference = True
        if added_ed_gs_reference:
            axis.plot([], [], color="#444444", linestyle="--", linewidth=1.4, label="ED GS T=0")
        if added_dmrg_reference:
            axis.plot([], [], color="#444444", linestyle=":", linewidth=1.4, label="DMRG T=0")
        axis.set_title(channel)
        axis.set_xlabel("Temperature T")
        axis.set_ylabel("Structure factor")
        axis.grid(alpha=0.25)
        _finite_temperature_xscale(axis, rows, thermal_data)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(titled_for_run("Finite-Temperature Structure Factors: ED with T=0 References", title_label))
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def _grid_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot build grid edges from an empty coordinate array.")
    if values.size == 1:
        delta = max(0.5 * abs(values[0]), 0.5)
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], mids, [last]])


def _nested_phase_observable_value(row: Dict[str, Any], path: Tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return float("nan")
        value = value[key]
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if np.isfinite(output) else float("nan")


def save_phase_observable_heatmap(
    rows: List[Dict[str, Any]],
    filepath: str,
    observable_path: Tuple[str, ...],
    title: str,
    colorbar_label: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good_rows = [
        row for row in rows
        if str(row.get("status", "completed")) == "completed"
    ]
    if len(good_rows) == 0:
        raise RuntimeError("No completed observable-scan rows available to plot.")

    alphas = np.asarray(sorted({float(row["alpha"]) for row in good_rows}), dtype=float)
    betas = np.asarray(sorted({float(row["beta"]) for row in good_rows}), dtype=float)
    value_grid = np.full((len(betas), len(alphas)), np.nan, dtype=float)
    alpha_index = {float(value): idx for idx, value in enumerate(alphas)}
    beta_index = {float(value): idx for idx, value in enumerate(betas)}

    for row in good_rows:
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        value_grid[beta_index[beta], alpha_index[alpha]] = _nested_phase_observable_value(
            row,
            observable_path,
        )

    if not np.any(np.isfinite(value_grid)):
        raise RuntimeError(f"No finite values available for observable path {'.'.join(observable_path)}.")

    alpha_edges = _grid_edges(alphas)
    beta_edges = _grid_edges(betas)
    masked_grid = np.ma.masked_invalid(value_grid)

    fig, ax = plt.subplots(figsize=(7.1, 5.2), dpi=150)
    mesh = ax.pcolormesh(
        alpha_edges,
        beta_edges,
        masked_grid,
        cmap="viridis",
        shading="flat",
        edgecolors=(0.0, 0.0, 0.0, 0.10),
        linewidth=0.25,
    )
    ax.scatter(
        [float(row["alpha"]) for row in good_rows],
        [float(row["beta"]) for row in good_rows],
        c="black",
        s=7,
        marker=".",
        linewidths=0,
        zorder=3,
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(titled_for_run(title, title_label))
    ax.set_xlim(float(alpha_edges[0]), float(alpha_edges[-1]))
    ax.set_ylim(float(beta_edges[0]), float(beta_edges[-1]))
    ax.grid(color="black", alpha=0.12, linewidth=0.4)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_phase_diagram_plot(
    rows: List[Dict[str, Any]],
    filepath: str,
    title: str,
    title_label: str | None = None,
    x_label: str = r"$\alpha$",
    y_label: str = r"$\beta$",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    good_rows = [
        row for row in rows
        if str(row.get("status", "completed")) == "completed" and "phase_label" in row
    ]
    if len(good_rows) == 0:
        raise RuntimeError("No completed phase-scan rows available to plot.")

    phase_order = [
        "Spin-Orbital Liquid",
        "Spin liquid",
        "NP1",
        "NP2",
        "NP3",
        "Stripy S / AFO",
        "AFM / AFO",
        "Weak/undetermined",
    ]
    phase_colors = {
        "Spin-Orbital Liquid": "#d7191c",
        "Spin liquid": "#d7191c",
        "NP1": "#f3dfb8",
        "NP2": "#ead1d9",
        "NP3": "#f0edb8",
        "Stripy S / AFO": "#dcece7",
        "AFM / AFO": "#d8d1ec",
        "Weak/undetermined": "#eeeeee",
    }
    for row in good_rows:
        label = str(row["phase_label"])
        if label not in phase_order:
            phase_order.append(label)
            phase_colors[label] = "#cccccc"

    alphas = np.asarray(sorted({float(row["alpha"]) for row in good_rows}), dtype=float)
    betas = np.asarray(sorted({float(row["beta"]) for row in good_rows}), dtype=float)
    code_grid = np.full((len(betas), len(alphas)), np.nan, dtype=float)
    alpha_index = {float(value): idx for idx, value in enumerate(alphas)}
    beta_index = {float(value): idx for idx, value in enumerate(betas)}
    phase_to_code = {phase: idx for idx, phase in enumerate(phase_order)}

    for row in good_rows:
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        phase = str(row["phase_label"])
        code_grid[beta_index[beta], alpha_index[alpha]] = float(phase_to_code[phase])

    cmap = ListedColormap([phase_colors[phase] for phase in phase_order])
    norm = BoundaryNorm(np.arange(len(phase_order) + 1) - 0.5, cmap.N)
    masked_grid = np.ma.masked_invalid(code_grid)
    alpha_edges = _grid_edges(alphas)
    beta_edges = _grid_edges(betas)

    fig, ax = plt.subplots(figsize=(7.1, 5.4), dpi=150)
    ax.pcolormesh(
        alpha_edges,
        beta_edges,
        masked_grid,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors=(0.0, 0.0, 0.0, 0.10),
        linewidth=0.25,
    )
    rows_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in good_rows:
        method_key = _phase_row_method_key(row) or "phase_scan"
        rows_by_method.setdefault(method_key, []).append(row)
    for method_key, method_rows in rows_by_method.items():
        marker = METHOD_MARKERS.get(method_key, ".")
        ax.scatter(
            [float(row["alpha"]) for row in method_rows],
            [float(row["beta"]) for row in method_rows],
            c=METHOD_COLORS.get(method_key, "black"),
            s=42 if marker == "*" else 14,
            marker=marker,
            linewidths=0.35 if marker == "*" else 0,
            edgecolors="#222222" if marker == "*" else "none",
            zorder=3,
            label=phase_scan_method_display_name(method_key),
        )

    for phase in phase_order:
        phase_rows = [row for row in good_rows if str(row["phase_label"]) == phase]
        if len(phase_rows) == 0:
            continue
        alpha_center = float(np.median([float(row["alpha"]) for row in phase_rows]))
        beta_center = float(np.median([float(row["beta"]) for row in phase_rows]))
        label = phase.replace(" / ", "\n")
        ax.text(alpha_center, beta_center, label, ha="center", va="center", fontsize=8.5)

    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    ax.set_title(titled_for_run(title, title_label))
    legend_handles = [
        Patch(facecolor=phase_colors[phase], edgecolor="black", linewidth=0.4, label=phase)
        for phase in phase_order
        if np.any(code_grid == phase_to_code[phase])
    ]
    phase_legend = None
    if legend_handles:
        phase_legend = ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.add_artist(phase_legend)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS.get(method_key, "."),
            color="none",
            markerfacecolor=METHOD_COLORS.get(method_key, "black"),
            markeredgecolor="#222222" if METHOD_MARKERS.get(method_key) == "*" else "none",
            markersize=9 if METHOD_MARKERS.get(method_key) == "*" else 5,
            label=phase_scan_method_display_name(method_key),
        )
        for method_key in rows_by_method
        if method_key in METHOD_MARKERS
    ]
    if method_handles and (len(method_handles) > 1 or "quimb_ipeps" in rows_by_method):
        anchor_y = 0.46 if phase_legend is not None else 1.0
        ax.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(1.02, anchor_y), fontsize=8)
    ax.set_xlim(float(alpha_edges[0]), float(alpha_edges[-1]))
    ax.set_ylim(float(beta_edges[0]), float(beta_edges[-1]))
    ax.grid(color="black", alpha=0.12, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


# ----------------------------------------------------------------------

PHASE_DIAGRAM_TITLES: tuple[tuple[str, str], ...] = (
    ("classical_product", "Classical Product-State Phase Diagram"),
    ("quantum_ed", "Quantum ED Phase Diagram"),
    ("tenax_dmrg", "Tenax Finite-DMRG Phase Diagram"),
    ("tenpy_dmrg", "TeNPy Finite-DMRG Phase Diagram"),
    ("tenax_idmrg", "Tenax iDMRG Phase Diagram"),
    ("tenpy_idmrg", "TeNPy iDMRG Phase Diagram"),
    ("quimb_peps", "quimb PEPS Phase Diagram"),
    ("quimb_ipeps", "quimb iPEPS Phase Diagram"),
)

TENSOR_NETWORK_OBSERVABLE_TITLES: tuple[tuple[str, str], ...] = (
    ("tenpy_dmrg", "TeNPy finite-DMRG"),
    ("tenpy_idmrg", "TeNPy iDMRG"),
    ("quimb_peps", "quimb PEPS"),
    ("quimb_ipeps", "quimb iPEPS"),
)

BASE_PHASE_OBSERVABLE_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("S_E", ("observables", "S_E"), "Center-bond entanglement entropy"),
    ("Sz_center", ("observables", "local_order_parameters", "Sz_center_mean"), "Center-site <Sz>"),
    ("tau_z_center", ("observables", "local_order_parameters", "tau_z_center_mean"), "Center-site <tau_z>"),
    ("W_p", ("observables", "W_p"), "Plaquette flux W_p"),
)


def _load_json_object(json_path: str) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("The JSON file must contain an object.")
    return data


def _phase_scan_from_json_object(data: dict[str, Any]) -> dict[str, Any]:
    phase_scan = data.get("phase_scan")
    if isinstance(phase_scan, dict):
        return phase_scan
    if any(mode_key in data for mode_key, _title in PHASE_DIAGRAM_TITLES):
        return data
    raise KeyError("No phase_scan object found in the JSON file.")


def _default_phase_json_prefix(json_path: str, data: dict[str, Any]) -> str:
    prefix = data.get("run_output_prefix")
    if isinstance(prefix, str) and prefix.strip():
        return prefix.strip()

    stem = os.path.splitext(os.path.basename(json_path))[0]
    for suffix in ("_run_summary", "_phase_scan_summary"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _phase_diagram_base_name(mode_key: str) -> str:
    if mode_key == "classical_product":
        return "classical_phase_diagram.png"
    if mode_key == "quantum_ed":
        return "quantum_phase_diagram.png"
    return f"{mode_key}_phase_diagram.png"


def _labeled_filename(prefix: str, base_name: str) -> str:
    return f"{prefix}_{base_name}" if prefix else base_name


def _observable_specs_for_phase_mode(mode_key: str) -> list[tuple[str, tuple[str, ...], str]]:
    specs = list(BASE_PHASE_OBSERVABLE_SPECS)
    if mode_key == "tenpy_idmrg":
        specs.append(("xi", ("observables", "xi"), "Correlation length xi"))
    return specs


def save_phase_diagrams_from_json(
    json_path: str,
    *,
    output_folder: str | None = None,
    prefix: str | None = None,
    modes: list[str] | None = None,
    continue_on_error: bool = True,
) -> dict[str, str]:
    """Save phase diagram and phase-observable PNGs from a run-summary JSON."""
    data = _load_json_object(json_path)
    phase_scan = _phase_scan_from_json_object(data)

    target_folder = _resolve_target_folder(output_folder, os.path.dirname(os.path.abspath(json_path)))
    os.makedirs(target_folder, exist_ok=True)
    filename_prefix = prefix or _default_phase_json_prefix(json_path, data)
    requested_modes = set(modes or [])
    saved: dict[str, str] = {}

    def should_plot_mode(mode_key: str) -> bool:
        return not requested_modes or mode_key in requested_modes

    def handle_plot_error(label: str, exc: Exception) -> None:
        if not continue_on_error:
            raise exc
        print(f"[phase-plot] skipped {label}: {exc}")

    for mode_key, title in PHASE_DIAGRAM_TITLES:
        if not should_plot_mode(mode_key):
            continue
        mode_data = phase_scan.get(mode_key)
        if not isinstance(mode_data, dict):
            continue
        rows = list(mode_data.get("rows", []))
        if not rows:
            print(f"[phase-plot] skipped {mode_key}: no rows available")
            continue
        base_name = _phase_diagram_base_name(mode_key)
        filepath = os.path.join(target_folder, _labeled_filename(filename_prefix, base_name))
        try:
            save_phase_diagram_plot(rows, filepath, title)
        except Exception as exc:
            handle_plot_error(f"{mode_key} phase diagram", exc)
            continue
        saved[f"{mode_key}_phase_diagram"] = filepath
        print(f"[phase-plot] saved: {filepath}")

    for mode_key, title_prefix in TENSOR_NETWORK_OBSERVABLE_TITLES:
        if not should_plot_mode(mode_key):
            continue
        mode_data = phase_scan.get(mode_key)
        if not isinstance(mode_data, dict):
            continue
        rows = list(mode_data.get("rows", []))
        if not rows:
            print(f"[phase-plot] skipped {mode_key} observables: no rows available")
            continue
        for observable_name, observable_path, colorbar_label in _observable_specs_for_phase_mode(mode_key):
            base_name = f"{mode_key}_{observable_name}_phase_observable.png"
            filepath = os.path.join(target_folder, _labeled_filename(filename_prefix, base_name))
            title = f"{title_prefix} {colorbar_label}"
            try:
                save_phase_observable_heatmap(
                    rows,
                    filepath,
                    observable_path,
                    title,
                    colorbar_label,
                )
            except Exception as exc:
                handle_plot_error(f"{mode_key} {observable_name}", exc)
                continue
            saved[f"{mode_key}_{observable_name}"] = filepath
            print(f"[phase-plot] saved: {filepath}")

    if requested_modes:
        known_modes = {key for key, _title in PHASE_DIAGRAM_TITLES}
        missing_modes = sorted(requested_modes.difference(known_modes))
        if missing_modes:
            print(f"[phase-plot] warning: unknown mode(s): {', '.join(missing_modes)}")

    return saved


def _geometry_from_run_summary(summary: dict[str, Any]) -> GeometryData:
    parameters = summary.get("parameters") or {}
    geometry_summary = summary.get("geometry") or {}
    lattice = parameters.get("lattice") or geometry_summary.get("lattice") or "honeycomb"
    try:
        length_x = int(parameters.get("length_x", geometry_summary["length_x"]))
        length_y = int(parameters.get("length_y", geometry_summary["length_y"]))
    except KeyError as exc:
        raise KeyError(
            "Run summary is missing length_x/length_y. Regenerate it with the current ylmodel_main.py."
        ) from exc
    circumference_x = bool(parameters.get("circumference_x", geometry_summary.get("circumference_x", False)))
    circumference_y = bool(parameters.get("circumference_y", geometry_summary.get("circumference_y", True)))
    return build_lattice_geometry(
        str(lattice),
        length_x=length_x,
        length_y=length_y,
        circumference_x=circumference_x,
        circumference_y=circumference_y,
    )


def _pattern_payload_from_summary(summary: dict[str, Any], method: str) -> dict[str, Any]:
    method_payload = summary.get(method)
    if not isinstance(method_payload, dict):
        raise KeyError(f"Run summary has no '{method}' section.")
    patterns = method_payload.get("real_space_patterns")
    if not isinstance(patterns, dict):
        raise KeyError(
            f"Run summary has no {method}.real_space_patterns data. "
            "Run ylmodel_main.py with --calculate-real-space-patterns."
        )
    correlations = patterns.get("correlations")
    if not isinstance(correlations, dict) or "S" not in correlations:
        raise KeyError(
            f"{method}.real_space_patterns must contain spin 'S' correlation rows."
        )
    return patterns


def _bond_rows_from_summary(summary: dict[str, Any], method: str) -> list[dict[str, Any]]:
    method_payload = summary.get(method)
    if not isinstance(method_payload, dict):
        return []
    for container in (method_payload, method_payload.get("info")):
        if not isinstance(container, dict):
            continue
        for key in ("bond_energies", "bond_rows"):
            rows = container.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _default_pattern_json_prefix(summary_json: str, method: str) -> str:
    stem = os.path.splitext(os.path.basename(summary_json))[0]
    suffix = "_run_summary"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return f"{stem}_{method}"


def save_patterns_from_summary(
    summary_json: str,
    *,
    method: str = "dmrg",
    output_folder: str | None = None,
    prefix: str | None = None,
) -> dict[str, str]:
    """Save the compact spin-direction pattern with resolved bond energies."""
    summary = _load_json_object(summary_json)
    geometry = _geometry_from_run_summary(summary)
    patterns = _pattern_payload_from_summary(summary, method)
    reference_site_idx = int(patterns["reference_site_idx"])
    correlations = patterns["correlations"]
    bond_rows = _bond_rows_from_summary(summary, method)
    if len(bond_rows) == 0:
        raise KeyError(
            f"Run summary has no {method}.bond_energies rows to overlay. "
            "Regenerate the summary with the current ylmodel_main.py and --calculate-bond-energies."
        )

    target_folder = _resolve_target_folder(output_folder, os.path.dirname(os.path.abspath(summary_json)))
    os.makedirs(target_folder, exist_ok=True)
    filename_prefix = prefix or _default_pattern_json_prefix(summary_json, method)
    method_label = {
        "dmrg": "DMRG",
        "peps": "PEPS",
        "ed": "ED",
    }.get(method, method.upper())
    title_label = summary.get("plot_title_label")
    external_field_payload = summary.get("external_field")
    external_field_vector = (
        external_field_payload.get("field_vector_hx_hy_hz")
        if isinstance(external_field_payload, dict)
        else None
    )

    combined_path = os.path.join(target_folder, f"{filename_prefix}_spin_vectors_bond_energy.png")

    save_phase_representative_pattern(
        geometry=geometry,
        spin_correlation_array=np.asarray(correlations["S"], dtype=float),
        reference_site_idx=reference_site_idx,
        bond_rows=bond_rows,
        filepath=combined_path,
        title=titled_for_run(f"{method_label} Spin Pattern + Resolved Bond Energy", str(title_label) if title_label else None),
        external_field_vector=external_field_vector,
    )
    return {"combined": combined_path}


def _build_plot_outputs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plotting utilities for Yao-Lee run summaries. All plotting entrypoints live here."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase_parser = subparsers.add_parser(
        "phase-json",
        help="Regenerate phase-scan diagrams from a *_run_summary.json or *_phase_scan_summary.json file.",
    )
    phase_parser.add_argument("json_path", help="Path to the saved run summary JSON.")
    phase_parser.add_argument("--output-folder", default=None, help="Folder for generated PNG files.")
    phase_parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to run_output_prefix.")
    phase_parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Only plot one phase-scan mode. May be supplied multiple times.",
    )
    phase_parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first plot error instead of continuing with other figures.",
    )

    pattern_parser = subparsers.add_parser(
        "real-space-json",
        help="Regenerate the compact spin-pattern/bond-energy diagram from a run summary.",
    )
    pattern_parser.add_argument("summary_json", help="Path to a *_run_summary.json file.")
    pattern_parser.add_argument("--method", choices=("dmrg", "peps", "ed"), default="dmrg", help="Summary method section to plot.")
    pattern_parser.add_argument("--output-folder", default=None, help="Folder for generated PNG files.")
    pattern_parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    return parser


def main() -> None:
    parser = _build_plot_outputs_parser()
    args = parser.parse_args()
    if args.command == "phase-json":
        saved = save_phase_diagrams_from_json(
            args.json_path,
            output_folder=args.output_folder,
            prefix=args.prefix,
            modes=args.mode,
            continue_on_error=not args.strict,
        )
        if not saved:
            raise SystemExit("[phase-plot] no plots were generated.")
        return
    if args.command == "real-space-json":
        saved = save_patterns_from_summary(
            args.summary_json,
            method=args.method,
            output_folder=args.output_folder,
            prefix=args.prefix,
        )
        print(f"[pattern] saved combined: {saved['combined']}")
        return
    parser.error(f"Unknown command '{args.command}'.")


if __name__ == "__main__":
    main()
