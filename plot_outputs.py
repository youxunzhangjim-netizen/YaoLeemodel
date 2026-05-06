#!/usr/bin/env python3
"""PNG output helpers for the Yao-Lee driver.

This module owns plotting, diagram rendering, and filesystem output helpers
only. Hamiltonian construction remains in ``models.py``; scan analysis remains
in ``analysis.py``; Tenax execution remains in ``backend.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from analysis import ENTROPY_ORDERS
from models import GeometryData, lattice_display_name


METHOD_ORDER = ("DMRG", "ED", "iDMRG-x")
METHOD_COLORS = {
    "DMRG": "#1f77b4",
    "ED": "#ff7f0e",
    "iDMRG-x": "#2ca02c",
}
METHOD_MARKERS = {
    "DMRG": "o",
    "ED": "s",
    "iDMRG-x": "^",
}
METHOD_LINESTYLES = {
    "DMRG": "-",
    "ED": "--",
    "iDMRG-x": ":",
}


def titled_for_run(base_title: str, title_label: str | None = None) -> str:
    if title_label:
        return f"{base_title}\n{title_label}"
    return base_title


def _ordered_available_methods(data: Dict[str, Any]) -> List[str]:
    return [method for method in METHOD_ORDER if method in data]


def ensure_folder_exists(folder_path: str) -> None:
    os.makedirs(folder_path, exist_ok=True)


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


def save_geometry_diagram(
    geometry: GeometryData,
    filepath: str,
    lattice: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
    positions = _geometry_positions(geometry)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for bond in geometry.bond_list:
        site_i, site_j, gamma = _bond_i_j_gamma(bond)
        p_i = positions[site_i]
        p_j = positions[site_j]
        ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], color=colors.get(gamma, "#666666"), linewidth=1.5, alpha=0.9)

    if hasattr(geometry, "sublattice_indices"):
        sublattice = np.asarray(geometry.sublattice_indices)
        if np.any(sublattice == 1):
            a_idx = np.where(sublattice == 0)[0]
            b_idx = np.where(sublattice == 1)[0]
            ax.scatter(positions[a_idx, 0], positions[a_idx, 1], s=20, c="#111111", label="A")
            ax.scatter(positions[b_idx, 0], positions[b_idx, 1], s=20, c="#ff7f0e", label="B")
        else:
            ax.scatter(positions[:, 0], positions[:, 1], s=16, c="#111111", label="sites")
    else:
        ax.scatter(positions[:, 0], positions[:, 1], s=16, c="#111111", label="sites")
    ax.set_title(titled_for_run(f"{lattice_display_name(lattice)} Cylinder Geometry", title_label))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
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
    from matplotlib.collections import LineCollection

    positions = _geometry_positions(geometry)
    segments = []
    values = []
    for row in bond_rows:
        i, j = int(row["i"]), int(row["j"])
        segments.append([positions[i], positions[j]])
        values.append(float(row["O_ij_gamma"]))

    values_arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    collection = LineCollection(segments, cmap="coolwarm", linewidths=3.0)
    collection.set_array(values_arr)
    ax.add_collection(collection)
    ax.scatter(positions[:, 0], positions[:, 1], c="black", s=10, zorder=3)
    ax.autoscale()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    cbar = fig.colorbar(collection, ax=ax, shrink=0.9)
    cbar.set_label("Bond energy O_ij_gamma")
    fig.tight_layout()
    fig.savefig(filepath)
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
    axes[0].set_ylabel("Value")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        axes[0].legend(loc="best")
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


def save_phase_diagram_plot(
    rows: List[Dict[str, Any]],
    filepath: str,
    title: str,
    title_label: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    good_rows = [
        row for row in rows
        if str(row.get("status", "completed")) == "completed" and "phase_label" in row
    ]
    if len(good_rows) == 0:
        raise RuntimeError("No completed phase-scan rows available to plot.")

    phase_order = [
        "Spin liquid",
        "NP1",
        "NP2",
        "NP3",
        "Stripy S / AFO",
        "AFM / AFO",
        "Weak/undetermined",
    ]
    phase_colors = {
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
    ax.scatter(
        [float(row["alpha"]) for row in good_rows],
        [float(row["beta"]) for row in good_rows],
        c="black",
        s=8,
        marker=".",
        linewidths=0,
        zorder=3,
    )

    for phase in phase_order:
        phase_rows = [row for row in good_rows if str(row["phase_label"]) == phase]
        if len(phase_rows) == 0:
            continue
        alpha_center = float(np.median([float(row["alpha"]) for row in phase_rows]))
        beta_center = float(np.median([float(row["beta"]) for row in phase_rows]))
        label = phase.replace(" / ", "\n")
        ax.text(alpha_center, beta_center, label, ha="center", va="center", fontsize=8.5)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(titled_for_run(title, title_label))
    legend_handles = [
        Patch(facecolor=phase_colors[phase], edgecolor="black", linewidth=0.4, label=phase)
        for phase in phase_order
        if np.any(code_grid == phase_to_code[phase])
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.set_xlim(float(alpha_edges[0]), float(alpha_edges[-1]))
    ax.set_ylim(float(beta_edges[0]), float(beta_edges[-1]))
    ax.grid(color="black", alpha=0.12, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


# ----------------------------------------------------------------------
