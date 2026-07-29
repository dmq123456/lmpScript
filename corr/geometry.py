#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def lammps_box_to_lattice(box_header: str, box_lines: List[str]) -> np.ndarray:
    bounds = [line.split() for line in box_lines]
    if len(bounds) != 3:
        raise ValueError("BOX BOUNDS must contain exactly 3 lines.")

    if all(len(row) == 2 for row in bounds):
        xlo, xhi = (float(v) for v in bounds[0])
        ylo, yhi = (float(v) for v in bounds[1])
        zlo, zhi = (float(v) for v in bounds[2])
        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo
        return np.asarray(
            [
                [lx, 0.0, 0.0],
                [0.0, ly, 0.0],
                [0.0, 0.0, lz],
            ],
            dtype=float,
        )

    if not all(len(row) == 3 for row in bounds):
        raise ValueError(f"Unsupported BOX BOUNDS format: {box_header.strip()!r}")

    xlo_bound, xhi_bound, xy = (float(v) for v in bounds[0])
    ylo_bound, yhi_bound, xz = (float(v) for v in bounds[1])
    zlo_bound, zhi_bound, yz = (float(v) for v in bounds[2])

    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    zlo = zlo_bound
    zhi = zhi_bound

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    return np.asarray(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=float,
    )


def reciprocal_lattice_from_real(lattice: np.ndarray) -> np.ndarray:
    a1, a2, a3 = lattice
    volume = np.dot(a1, np.cross(a2, a3))
    if abs(volume) < 1e-14:
        raise ValueError("Real-space lattice vectors are singular.")
    return np.asarray(
        [
            2.0 * np.pi * np.cross(a2, a3) / volume,
            2.0 * np.pi * np.cross(a3, a1) / volume,
            2.0 * np.pi * np.cross(a1, a2) / volume,
        ],
        dtype=float,
    )


def frac_to_cart_q(q_frac: np.ndarray, reciprocal_lattice: np.ndarray) -> np.ndarray:
    return q_frac @ reciprocal_lattice


def cart_to_frac_q(q_cart: np.ndarray, reciprocal_lattice: np.ndarray) -> np.ndarray:
    return q_cart @ np.linalg.inv(reciprocal_lattice)


def build_q_path(path_points: np.ndarray, points_per_segment: int) -> Tuple[np.ndarray, np.ndarray]:
    if points_per_segment < 2:
        raise ValueError("--points-per-segment must be >= 2")

    q_list: List[np.ndarray] = []
    node_indices = [0]
    total_points = 0

    for iseg in range(len(path_points) - 1):
        start = path_points[iseg]
        end = path_points[iseg + 1]
        segment = np.linspace(start, end, points_per_segment, endpoint=True)
        if iseg > 0:
            segment = segment[1:]
        q_list.append(segment)
        total_points += len(segment)
        node_indices.append(total_points - 1)

    return np.vstack(q_list), np.asarray(node_indices, dtype=int)


def load_q_file(
    qfile: str | Path,
    points_per_segment: int,
) -> Tuple[np.ndarray, np.ndarray | None, List[str] | None]:
    labels: List[str] = []
    q_list: List[List[float]] = []
    labeled = None

    with open(qfile, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                label, coords = line.split(":", 1)
                fields = [label.strip(), *coords.replace(",", " ").split()]
            else:
                fields = line.replace(",", " ").split()

            if len(fields) == 3:
                if labeled is True:
                    raise ValueError("q file cannot mix labeled and unlabeled lines.")
                labeled = False
                q_list.append([float(v) for v in fields])
            elif len(fields) == 4:
                if labeled is False:
                    raise ValueError("q file cannot mix labeled and unlabeled lines.")
                labeled = True
                labels.append(fields[0].strip())
                q_list.append([float(v) for v in fields[1:]])
            else:
                raise ValueError(f"Invalid q line in {str(qfile)!r}: {raw_line.rstrip()!r}")

    if not q_list:
        raise ValueError(f"No q vectors found in file: {qfile}")

    q_frac = np.asarray(q_list, dtype=float)
    if labeled:
        q_frac, q_node_indices = build_q_path(q_frac, points_per_segment)
        return q_frac, q_node_indices, labels
    return q_frac, None, None


def q_path_distance(q_vectors: np.ndarray) -> np.ndarray:
    q_dist = np.zeros(q_vectors.shape[0], dtype=float)
    for i in range(1, len(q_dist)):
        q_dist[i] = q_dist[i - 1] + np.linalg.norm(q_vectors[i] - q_vectors[i - 1])
    return q_dist


def component_indices(components: str) -> List[int]:
    components = components.lower().strip()
    mapping = {"x": 0, "y": 1, "z": 2}
    if components in ("xyz", "all"):
        return [0, 1, 2]
    if components in ("xy", "transverse"):
        return [0, 1]
    idx = []
    for c in components:
        if c not in mapping:
            raise ValueError(f"Unsupported component specifier: {components!r}")
        idx.append(mapping[c])
    if not idx:
        raise ValueError("No valid components selected.")
    return idx


def primitive_lattice_from_supercell(
    supercell_lattice: np.ndarray,
    repeats: np.ndarray,
) -> np.ndarray:
    repeats = np.asarray(repeats, dtype=float)
    if repeats.shape != (3,) or np.any(repeats <= 0):
        raise ValueError("supercell repeats must be three positive integers")
    return supercell_lattice / repeats[:, None]


def generate_bz_points(
    reciprocal_lattice: np.ndarray,
    nh: int,
    nk: int,
    frac_limit: float,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nh < 2 or nk < 2:
        raise ValueError("nh and nk must be >= 2")

    h_vals = np.linspace(-frac_limit, frac_limit, nh)
    k_vals = np.linspace(-frac_limit, frac_limit, nk)
    hh, kk = np.meshgrid(h_vals, k_vals, indexing="ij")
    q_frac_all = np.column_stack([hh.ravel(), kk.ravel(), np.zeros(hh.size, dtype=float)])
    q_cart_all = frac_to_cart_q(q_frac_all, reciprocal_lattice)

    g_list = []
    b1 = reciprocal_lattice[0]
    b2 = reciprocal_lattice[1]
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            g_list.append(i * b1 + j * b2)
    g_vectors = np.asarray(g_list, dtype=float)

    q_norm2 = np.sum(q_cart_all * q_cart_all, axis=1)
    inside = np.ones(q_cart_all.shape[0], dtype=bool)
    for g in g_vectors:
        shifted = q_cart_all - g
        shifted_norm2 = np.sum(shifted * shifted, axis=1)
        inside &= q_norm2 <= shifted_norm2 + tol

    return q_frac_all, q_cart_all, inside


def generate_reciprocal_cell_grid(nh: int, nk: int) -> np.ndarray:
    if nh < 2 or nk < 2:
        raise ValueError("nh and nk must be >= 2")

    h_vals = np.linspace(0.0, 1.0, nh)
    k_vals = np.linspace(0.0, 1.0, nk)
    hh, kk = np.meshgrid(h_vals, k_vals, indexing="ij")
    return np.column_stack([hh.ravel(), kk.ravel(), np.zeros(hh.size, dtype=float)])


__all__ = [
    "build_q_path",
    "cart_to_frac_q",
    "component_indices",
    "frac_to_cart_q",
    "generate_bz_points",
    "generate_reciprocal_cell_grid",
    "lammps_box_to_lattice",
    "load_q_file",
    "primitive_lattice_from_supercell",
    "q_path_distance",
    "reciprocal_lattice_from_real",
]


# ----------------------------------------------------------------------
# Folded Brillouin zone
# ----------------------------------------------------------------------
def validate_bz_folded(bz_folded) -> np.ndarray:
    folded = np.asarray(bz_folded, dtype=np.int64)
    if folded.shape != (3,) or np.any(folded < 1):
        raise ValueError("--bz-folded must be three positive integers")
    return folded


def folded_reciprocal_lattice(primitive_reciprocal: np.ndarray, bz_folded) -> np.ndarray:
    """Reciprocal vectors of the folded cell: b_i^fold = b_i^prim / F_i."""
    folded = validate_bz_folded(bz_folded).astype(float)
    return primitive_reciprocal / folded[:, None]


def bz_fold_shift_vectors(bz_folded) -> np.ndarray:
    folded = validate_bz_folded(bz_folded)
    grids = np.meshgrid(*(np.arange(int(n), dtype=float) for n in folded), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


def unfold_q_path(q_frac_folded: np.ndarray, bz_folded) -> tuple[np.ndarray, np.ndarray]:
    """Map q given in folded-cell fractional coordinates onto the primitive cell.

    q_unfold = (q_fold + s) / F,  s_d in {0..F_d-1}

    Note q_frac_folded is expressed in fractional coordinates of the *folded*
    reciprocal cell, not the primitive one.
    """
    folded = validate_bz_folded(bz_folded).astype(float)
    shifts = bz_fold_shift_vectors(folded.astype(np.int64))
    unfolded = (q_frac_folded[:, None, :] + shifts[None, :, :]) / folded[None, None, :]
    return unfolded.reshape(-1, 3), shifts


def fold_by_max(sqw_unfolded: np.ndarray, nq_folded: int, multiplicity: int) -> np.ndarray:
    """Reduce the unfolded branches back onto the folded path, per frequency.

    Accepts (nq, nfreq) or (nch, nq, nfreq). The maximum -- rather than a sum --
    keeps the dominant branch visible without piling up background from the
    other branches, at the cost of not conserving total spectral weight.
    """
    arr = np.asarray(sqw_unfolded)
    if arr.ndim not in (2, 3):
        raise ValueError("sqw_unfolded must be 2D (nq, nfreq) or 3D (nch, nq, nfreq)")
    expected = nq_folded * multiplicity
    if arr.shape[-2] != expected:
        raise ValueError(
            f"Expected {expected} unfolded q points for nq_folded={nq_folded} and "
            f"multiplicity={multiplicity}, got {arr.shape[-2]}"
        )
    new_shape = arr.shape[:-2] + (nq_folded, multiplicity, arr.shape[-1])
    return arr.reshape(new_shape).max(axis=-2)
