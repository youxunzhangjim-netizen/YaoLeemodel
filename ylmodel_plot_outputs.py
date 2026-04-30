#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from ylmodel_physics import GeometryData, lattice_display_name


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


def save_geometry_diagram(geometry: GeometryData, filepath: str, lattice: str) -> None:
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
    ax.set_title(f"{lattice_display_name(lattice)} Cylinder Geometry")
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
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred_order = ["DMRG", "ED", "iDMRG-x"]
    labels = [label for label in preferred_order if label in method_to_energy]
    labels += [label for label in method_to_energy.keys() if label not in labels]
    values = [float(method_to_energy[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    color_map = {
        "DMRG": "#1f77b4",
        "ED": "#ff7f0e",
        "iDMRG-x": "#2ca02c",
    }
    colors = [color_map.get(label, "#666666") for label in labels]
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Energy per site")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_entropy_profiles_comparison(
    entropy_profiles: Dict[str, Dict[str, Any]],
    filepath: str,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, sharex=True)
    axes_flat = list(axes.flatten())
    method_order = ["DMRG", "ED", "iDMRG-x"]
    colors = {
        "DMRG": "#1f77b4",
        "ED": "#ff7f0e",
        "iDMRG-x": "#2ca02c",
    }

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
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                color=colors.get(method, None),
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
    fig.suptitle("Entanglement Entropy Profiles by Method")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_entropy_method_means_comparison(
    entropy_profiles: Dict[str, Dict[str, Any]],
    filepath: str,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = [method for method in ["DMRG", "ED", "iDMRG-x"] if method in entropy_profiles]
    if len(method_order) == 0:
        raise RuntimeError("No entropy profiles available for method-mean comparison.")

    x = np.arange(len(orders), dtype=float)
    width = 0.8 / float(len(method_order))
    offsets = np.linspace(-0.4 + width / 2.0, 0.4 - width / 2.0, len(method_order))
    color_map = {
        "DMRG": "#1f77b4",
        "ED": "#ff7f0e",
        "iDMRG-x": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for idx, method in enumerate(method_order):
        profile = entropy_profiles[method]
        summary = profile.get("summary", {})
        means = [float(summary.get(f"S{order_n}_mean", np.nan)) for order_n in orders]
        ax.bar(
            x + offsets[idx],
            means,
            width=width,
            label=method,
            color=color_map.get(method, "#666666"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={order_n}" for order_n in orders])
    ax.set_xlabel("Renyi order")
    ax.set_ylabel("Mean entropy across cuts")
    ax.set_title("Method Comparison: Mean Entanglement Entropies")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def save_dmrg_ed_energy_comparison(dmrg_energy: float, ed_energy: float, filepath: str) -> None:
    save_multi_method_energy_comparison(
        method_to_energy={"DMRG": float(dmrg_energy), "ED": float(ed_energy)},
        filepath=filepath,
        title="Ground-State Energy Comparison",
    )


def save_dmrg_ed_structure_comparison(
    dmrg_rows: List[Dict[str, Any]],
    ed_rows: List[Dict[str, Any]],
    filepath: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dmrg_map = {row["Q_label"]: row for row in dmrg_rows}
    ed_map = {row["Q_label"]: row for row in ed_rows}
    labels = [label for label in dmrg_map.keys() if label in ed_map]
    x = np.arange(len(labels), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150, sharex=True)
    channels = ("S(Q)", "T(Q)", "ST(Q)")
    for ax, channel in zip(axes, channels):
        dmrg_values = [dmrg_map[label][channel] for label in labels]
        ed_values = [ed_map[label][channel] for label in labels]
        ax.plot(x, dmrg_values, marker="o", linewidth=1.8, label="DMRG")
        ax.plot(x, ed_values, marker="s", linewidth=1.8, label="ED")
        ax.set_title(channel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")
    fig.suptitle("DMRG vs ED Structure Factors")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


# ----------------------------------------------------------------------
