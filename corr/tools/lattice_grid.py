#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The regular lattice behind a frame, read off the atom positions.

Several derived quantities -- the topological charge density, the spin-current
polarization -- need the same thing first: the magnetic sites arranged as one
per primitive cell on a regular n1 x n2 grid, with the cell indices known, so
that neighbours can be reached by wrapping the index rather than by a distance
search. That grid is what this module builds, once, so `topocharge.py` and
`polarization.py` cannot drift apart in how they read it.

Nothing here knows about triangles or about bonds; it stops at `site_of`, the
(n1, n2) table of site indices.
"""

from __future__ import annotations

import numpy as np


def grid_period(frac: np.ndarray, max_n: int, sharpness: float = 0.9) -> int:
    """Number of cells along one axis, from the fractional coordinates.

    For a perfect grid the coordinates are f = k/n, so |<exp(2 pi i n f)>| = 1
    at n and vanishes below it. Thermal displacement only lowers the peak, so a
    single threshold works without a distance tolerance to tune.
    """
    frac = np.asarray(frac, dtype=float).ravel()
    for start in range(1, max_n + 1, 256):
        ns = np.arange(start, min(start + 256, max_n + 1))
        strength = np.abs(np.exp(2j * np.pi * np.outer(ns, frac)).mean(axis=1))
        hits = np.nonzero(strength > sharpness)[0]
        if hits.size:
            return int(ns[hits[0]])
    raise ValueError(
        "Could not read a regular grid off the magnetic-site positions. Either "
        "the sites are not one per primitive cell, or they are displaced too "
        "far from their ideal positions. Pass the repeat counts explicitly with "
        "--lattice-grid N1 N2."
    )


def grid_index(frac: np.ndarray, n: int) -> np.ndarray:
    """Cell index 0..n-1 for each site along one axis.

    The grid's own offset is removed first (the phase of the same Fourier sum),
    so every cluster of sites is centred on an integer before rounding and the
    result cannot depend on where the box origin happens to sit.
    """
    frac = np.asarray(frac, dtype=float).ravel()
    offset = np.angle(np.exp(2j * np.pi * n * frac).mean()) / (2.0 * np.pi)
    return np.rint(n * frac - offset).astype(np.int64) % n


def layer_masks(z: np.ndarray, single_layer: bool) -> list[np.ndarray]:
    """Split at the midpoint of the z range, the way frame.py draws panels.

    Each layer is closed on itself, so a bilayer never gets stitched into one
    surface and never grows bonds between the layers.
    """
    z = np.asarray(z, dtype=float)
    if single_layer:
        return [np.ones(z.size, dtype=bool)]
    midpoint = 0.5 * (z.min() + z.max())
    return [z > midpoint, z <= midpoint]


def site_of_grid(
    positions: np.ndarray,
    lattice: np.ndarray,
    grid: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, int]:
    """(site_of, n1, n2) for one layer: site_of[i, j] is the site in cell (i, j).

    `positions` are Cartesian and belong to a single layer; `lattice` holds the
    box vectors as rows. The grid is read off the fractional coordinates unless
    it is given explicitly.
    """
    positions = np.asarray(positions, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    n_sites = positions.shape[0]

    frac = positions @ np.linalg.inv(lattice)
    frac[:, :2] -= np.floor(frac[:, :2])

    if grid is None:
        n1 = grid_period(frac[:, 0], max_n=n_sites)
        n2 = grid_period(frac[:, 1], max_n=n_sites)
    else:
        n1, n2 = int(grid[0]), int(grid[1])
    if n1 * n2 != n_sites:
        raise ValueError(
            f"Grid {n1} x {n2} = {n1 * n2} does not match the {n_sites} site(s) in this "
            "layer. The most common cause is non-magnetic atoms left in the frame -- "
            "select the magnetic sublattice with --element (e.g. --element Ni). For a "
            "bilayer, drop --single-layer so the two layers are handled separately."
        )

    i = grid_index(frac[:, 0], n1)
    j = grid_index(frac[:, 1], n2)
    site_of = np.full((n1, n2), -1, dtype=np.int64)
    site_of[i, j] = np.arange(n_sites, dtype=np.int64)
    if np.any(site_of < 0):
        raise ValueError(
            "Two magnetic sites landed in the same cell, so the sites are not one per "
            "primitive cell on a regular grid. Pass --lattice-grid N1 N2 if the grid was "
            "read wrongly, or check that only the magnetic sublattice is selected."
        )
    return site_of, n1, n2


def grid_from_cfg(cfg: dict | None) -> tuple[int, int] | None:
    """The explicit grid out of a render config, under either option name.

    `--topo-grid` was the original spelling, kept as an alias of
    `--lattice-grid` now that more than the topological charge uses it.
    """
    cfg = cfg or {}
    return cfg.get("lattice_grid") or cfg.get("topo_grid") or None


__all__ = [
    "grid_from_cfg",
    "grid_index",
    "grid_period",
    "layer_masks",
    "site_of_grid",
]
