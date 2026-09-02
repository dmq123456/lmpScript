#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topological charge density on the lattice.

    Q = (1/4pi) \\int m . (dm/dx x dm/dy) dx dy

is discretised the standard way (Berg and Luscher): the plane is covered by
elementary, non-overlapping, anticlockwise-ordered triangles, and each triangle
contributes the signed area A_l that its three spins span on the unit sphere,

    Q = (1/4pi) sum_l A_l.

What this module returns is not Q but the density behind it: A_l/4pi shared
equally between the triangle's three vertices, giving one number q_i per site
with sum_i q_i = Q. That is the form frame.py can colour, since tripcolor with
gouraud shading wants a value per vertex rather than per triangle.

A_l is fixed, sign included, by the pair of identities

    cos(A_l/2) = (1 + a + b + c) / rho
    sin(A_l/2) = m_i . (m_j x m_k) / rho,    rho = sqrt(2(1+a)(1+b)(1+c))

with a = m_i.m_j, b = m_j.m_k, c = m_k.m_i. Dividing them cancels rho and
leaves A_l = 2 atan2(sin, cos), which is what is computed here. Taking arccos
of the first identity on its own is algebraically the same but numerically
worse: a smooth texture has A_l near zero, which is exactly where the argument
sits at 1 and arccos amplifies rounding error by some eight orders of
magnitude. The arccos form is kept in the self-test as a cross-check, not in
the render path.

Triangulation
-------------
The magnetic sites are taken to be one per primitive cell on a regular n1 x n2
grid, which is what `replicate n1 n2 1` of a cell with a single magnetic atom
produces. Each cell contributes two triangles and the cell indices wrap, so the
surface is closed and Q comes out an exact integer -- the cheapest check that
the whole path is right. The grid is read off the fractional coordinates rather
than assumed, and the connectivity is built once and reused for every frame,
which requires the atom order to be the same in every frame
(`dump_modify ... sort id`).

Delaunay is deliberately not used: on a perfect triangular lattice four points
are cocircular, so the triangulation is not unique and flickers from frame to
frame.

Run this file directly to execute the self-test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from geometry import lammps_box_to_lattice  # noqa: E402


# ----------------------------------------------------------------------
# Reading the grid off the positions
# ----------------------------------------------------------------------
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
        "--topo-grid N1 N2."
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


# ----------------------------------------------------------------------
# Triangulation
# ----------------------------------------------------------------------
def _layer_masks(z: np.ndarray, single_layer: bool) -> list[np.ndarray]:
    """Same split as frame.py draws, so each panel gets its own closed surface."""
    if single_layer:
        return [np.ones(z.size, dtype=bool)]
    midpoint = 0.5 * (z.min() + z.max())
    return [z > midpoint, z <= midpoint]


def _triangles_one_layer(
    positions: np.ndarray,
    lattice: np.ndarray,
    grid: tuple[int, int] | None,
) -> np.ndarray:
    """Two triangles per cell, indices wrapping, anticlockwise in the xy plane."""
    n_sites = positions.shape[0]
    if n_sites < 3:
        raise ValueError(f"Need at least 3 magnetic sites to triangulate, got {n_sites}.")

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
            "bilayer, drop --single-layer so the two layers are triangulated separately."
        )

    i = grid_index(frac[:, 0], n1)
    j = grid_index(frac[:, 1], n2)
    site_of = np.full((n1, n2), -1, dtype=np.int64)
    site_of[i, j] = np.arange(n_sites, dtype=np.int64)
    if np.any(site_of < 0):
        raise ValueError(
            "Two magnetic sites landed in the same cell, so the sites are not one per "
            "primitive cell on a regular grid. Pass --topo-grid N1 N2 if the grid was "
            "read wrongly, or check that only the magnetic sublattice is selected."
        )

    right = np.roll(site_of, -1, axis=0)   # (i+1, j)
    up = np.roll(site_of, -1, axis=1)      # (i, j+1)
    diag = np.roll(right, -1, axis=1)      # (i+1, j+1)

    lower = np.stack([site_of.ravel(), right.ravel(), up.ravel()], axis=1)
    upper = np.stack([right.ravel(), diag.ravel(), up.ravel()], axis=1)
    triangles = np.concatenate([lower, upper], axis=0)

    # One global orientation test: (a1 x a2).z < 0 means the cell is indexed
    # clockwise, and every triangle has to be reversed rather than tested.
    a1, a2 = lattice[0], lattice[1]
    if a1[0] * a2[1] - a1[1] * a2[0] < 0.0:
        triangles = triangles[:, ::-1]
    return triangles


def build_triangles(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lattice: np.ndarray,
    *,
    single_layer: bool = True,
    grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """(n_triangles, 3) site indices, one closed surface per layer."""
    positions = np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)],
                         axis=1)
    blocks = []
    for mask in _layer_masks(positions[:, 2], single_layer):
        where = np.nonzero(mask)[0]
        if where.size == 0:
            continue
        local = _triangles_one_layer(positions[where], np.asarray(lattice, float), grid)
        blocks.append(where[local])
    return np.concatenate(blocks, axis=0)


# ----------------------------------------------------------------------
# The charge itself
# ----------------------------------------------------------------------
def _unit(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError(
            "Some sites carry a zero spin, which has no direction. Drop them with "
            "--drop-zero-vector, or select the magnetic sublattice with --element."
        )
    return vectors / norm


def triangle_charge(m: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """A_l/4pi for every triangle, from A_l = 2 atan2(sin(A_l/2), cos(A_l/2))."""
    mi, mj, mk = m[triangles[:, 0]], m[triangles[:, 1]], m[triangles[:, 2]]
    numerator = np.einsum("ij,ij->i", mi, np.cross(mj, mk))
    denominator = (
        1.0
        + np.einsum("ij,ij->i", mi, mj)
        + np.einsum("ij,ij->i", mj, mk)
        + np.einsum("ij,ij->i", mk, mi)
    )
    return 2.0 * np.arctan2(numerator, denominator) / (4.0 * np.pi)


def triangle_charge_arccos(m: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The literal cos(A_l/2) = ... form, for the self-test only.

    Same value as triangle_charge, but it loses precision exactly where the
    texture is smooth and needs a clip to survive rounding past 1.
    """
    mi, mj, mk = m[triangles[:, 0]], m[triangles[:, 1]], m[triangles[:, 2]]
    a = np.einsum("ij,ij->i", mi, mj)
    b = np.einsum("ij,ij->i", mj, mk)
    c = np.einsum("ij,ij->i", mk, mi)
    rho = np.sqrt(2.0 * (1.0 + a) * (1.0 + b) * (1.0 + c))
    cos_half = np.clip((1.0 + a + b + c) / rho, -1.0, 1.0)
    sign = np.sign(np.einsum("ij,ij->i", mi, np.cross(mj, mk)))
    return 2.0 * np.arccos(cos_half) * sign / (4.0 * np.pi)


def density_from_triangles(m: np.ndarray, triangles: np.ndarray, n_sites: int) -> np.ndarray:
    """Per-site q_i: each triangle's charge split equally over its vertices."""
    per_triangle = triangle_charge(m, triangles)
    q = np.zeros(n_sites, dtype=float)
    np.add.at(q, triangles.ravel(), np.repeat(per_triangle / 3.0, 3))
    return q


# ----------------------------------------------------------------------
# Frame interface
# ----------------------------------------------------------------------
_TRIANGLE_CACHE: dict[tuple, np.ndarray] = {}


def _lattice_from_frame(frame: dict) -> np.ndarray:
    if "box_lines" not in frame:
        raise ValueError(
            "This frame carries no box bounds, so the periodic images needed to close "
            "the lattice are unknown. Re-read the dump with a dumpframe that keeps the "
            "box (load_frames/load_single_frame store frame['box_lines'])."
        )
    return lammps_box_to_lattice(frame.get("box_header", ""), frame["box_lines"])


def site_density(frame: dict, cfg: dict | None = None) -> np.ndarray:
    """Topological charge density q_i for one frame, one value per site.

    The triangulation is built on the first frame and reused, so the atom order
    has to be the same in every frame -- LAMMPS guarantees that with
    `dump_modify <id> sort id`.
    """
    cfg = cfg or {}
    if "u" not in frame:
        raise ValueError(
            "--color topo needs the spin direction, so --vector is required, e.g. "
            "--vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]'."
        )
    single_layer = bool(cfg.get("single_layer", True))
    grid = cfg.get("topo_grid") or None
    n_sites = int(frame["x"].size)

    key = (n_sites, single_layer, tuple(grid) if grid else None)
    triangles = _TRIANGLE_CACHE.get(key)
    if triangles is None:
        triangles = build_triangles(
            frame["x"], frame["y"], frame["z"], _lattice_from_frame(frame),
            single_layer=single_layer, grid=grid,
        )
        _TRIANGLE_CACHE[key] = triangles

    m = _unit(np.stack([frame["u"], frame["v"], frame["w"]], axis=1).astype(float))
    return density_from_triangles(m, triangles, n_sites)


__all__ = [
    "build_triangles",
    "density_from_triangles",
    "grid_index",
    "grid_period",
    "site_density",
    "triangle_charge",
    "triangle_charge_arccos",
]


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _triangular_lattice(n1: int, n2: int, a: float = 3.9793, c: float = 20.0):
    lattice = np.array([[a * n1, 0.0, 0.0],
                        [-0.5 * a * n2, np.sqrt(3.0) / 2.0 * a * n2, 0.0],
                        [0.0, 0.0, c]])
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    frac = np.stack([i.ravel() / n1, j.ravel() / n2, np.zeros(n1 * n2)], axis=1)
    return frac @ lattice, lattice


def _skyrmion(positions, lattice, radius_frac=0.25):
    frac = positions @ np.linalg.inv(lattice)
    delta = frac - 0.5
    delta[:, :2] -= np.rint(delta[:, :2])          # minimum image, triclinic-safe
    offset = delta @ lattice
    r = np.linalg.norm(offset[:, :2], axis=1)
    scale = radius_frac * np.linalg.norm(lattice[0])
    theta = np.where(r < scale, np.pi * (1.0 - r / scale), 0.0)
    phi = np.arctan2(offset[:, 1], offset[:, 0])
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)


def _selftest() -> int:
    rng = np.random.default_rng(0)
    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failures += 0 if ok else 1

    for n1, n2 in ((24, 24), (30, 18)):
        positions, lattice = _triangular_lattice(n1, n2)
        triangles = build_triangles(*positions.T, lattice, single_layer=True)
        check(f"triangle count {n1}x{n2}",
              triangles.shape == (2 * n1 * n2, 3),
              f"{triangles.shape[0]} triangles for {n1 * n2} sites")

        # Berg-Luscher is exactly integer-valued on a closed lattice for *any*
        # configuration -- the sharpest test there is, and it needs no reference.
        m = _unit(rng.normal(size=(n1 * n2, 3)))
        q = density_from_triangles(m, triangles, n1 * n2).sum()
        check(f"random config is integer {n1}x{n2}",
              abs(q - round(q)) < 1e-9, f"Q = {q:.12f}")

        # A skyrmion has to come out at exactly one quantum.
        m = _skyrmion(positions, lattice)
        q_site = density_from_triangles(m, triangles, n1 * n2)
        q = q_site.sum()
        check(f"skyrmion is one quantum {n1}x{n2}",
              abs(abs(q) - 1.0) < 1e-9, f"Q = {q:+.12f}")

        # Positions jittered off the ideal sites: the grid must still be read.
        jittered = positions + rng.normal(scale=0.02 * lattice[0, 0] / n1,
                                          size=positions.shape) * [1, 1, 0]
        check(f"grid survives thermal jitter {n1}x{n2}",
              np.array_equal(build_triangles(*jittered.T, lattice, single_layer=True),
                             triangles),
              "same connectivity as the ideal lattice")

        # The arccos form of the paper must agree with the atan2 form actually
        # used. It does -- but only to ~1e-8, several orders short of what two
        # equivalent double-precision expressions normally reach. That gap is
        # arccos losing precision at an argument near 1, which is where a smooth
        # texture puts every triangle, and it is the reason the render path uses
        # atan2 instead.
        atan2_form = triangle_charge(m, triangles)
        arccos_form = triangle_charge_arccos(m, triangles)
        gap = float(np.max(np.abs(atan2_form - arccos_form)))
        check(f"arccos form agrees {n1}x{n2}", gap < 1e-6,
              f"max |difference| = {gap:.2e} (arccos precision floor, not a discrepancy)")

        mi, mj = m[triangles[:, 0]], m[triangles[:, 1]]
        closest = 1.0 + np.min(np.einsum("ij,ij->i", mi, mj))
        check(f"no degenerate triangle {n1}x{n2}",
              closest > 1e-6, f"min(1 + m_i.m_j) = {closest:.3e}")

    # Two decoupled layers, one skyrmion each, must not be stitched together.
    positions, lattice = _triangular_lattice(20, 20)
    bilayer = np.concatenate([positions, positions + [0.0, 0.0, 6.0]], axis=0)
    m = np.concatenate([_skyrmion(positions, lattice), _skyrmion(positions, lattice)], axis=0)
    triangles = build_triangles(*bilayer.T, lattice, single_layer=False)
    q = density_from_triangles(m, triangles, bilayer.shape[0]).sum()
    check("bilayer splits into two surfaces", abs(abs(q) - 2.0) < 1e-9, f"Q = {q:+.12f}")

    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(_selftest())
