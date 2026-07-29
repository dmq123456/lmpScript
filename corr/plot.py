#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intensity-map rendering for S(q,w) along a q-path."""

from __future__ import annotations

from typing import List

import numpy as np

from geometry import q_path_distance
from result import THZ_TO_MEV


def axis_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("axis_edges expects a non-empty 1D array")
    if values.size == 1:
        delta = 0.5
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)

    edges = np.empty(values.size + 1, dtype=float)
    midpoints = 0.5 * (values[1:] + values[:-1])
    edges[1:-1] = midpoints
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def plot_sqw(
    q_vectors: np.ndarray,
    freq_thz: np.ndarray,
    sqw: np.ndarray,
    outfile: str | None = None,
    max_freq_thz: float | None = None,
    cbar_min: float | None = None,
    cbar_max: float | None = None,
    use_meV: bool = False,
    title: str | None = None,
    q_node_indices: np.ndarray | None = None,
    q_node_labels: List[str] | None = None,
    cmap: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_freq_thz is not None:
        mask = freq_thz <= max_freq_thz
        freq_plot = freq_thz[mask]
        sqw_plot = sqw[:, mask]
    else:
        freq_plot = freq_thz
        sqw_plot = sqw

    q_path = q_path_distance(q_vectors)
    y = freq_plot * THZ_TO_MEV if use_meV else freq_plot
    ylabel = "Energy (meV)" if use_meV else "Frequency (THz)"

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    q_edges = axis_edges(q_path)
    y_edges = axis_edges(y)
    mesh_kwargs = {
        "shading": "auto",
        "vmin": cbar_min,
        "vmax": cbar_max,
    }
    if cmap is not None:
        mesh_kwargs["cmap"] = cmap
    im = ax.pcolormesh(q_edges, y_edges, sqw_plot.T, **mesh_kwargs)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Intensity (arb. units)")

    ax.set_xlabel("q-path distance")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if q_node_indices is not None:
        tick_pos = q_path[q_node_indices]
        for xpos in tick_pos:
            ax.axvline(x=xpos, color="w", linestyle="--", linewidth=0.6)
        if q_node_labels is not None:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(q_node_labels)

    fig.tight_layout()
    prefix = "sqw_results" if outfile is None else str(Path(outfile).with_suffix(""))
    png_file = f"{prefix}.png"
    eps_file = f"{prefix}.eps"
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    # plt.savefig(eps_file, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {png_file}")
    # print(f"[INFO] Plot saved to: {eps_file}")

