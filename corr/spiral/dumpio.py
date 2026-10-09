#!/usr/bin/env python3
"""Reading a LAMMPS dump: the box and the frames, and nothing else.

Both converters in this directory need the same two things, and a triclinic box
in particular is easy to get subtly wrong, so it is written once here rather
than once per output format. Nothing in this module knows what the columns
mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_box(box_lines: list[str], header: str) -> np.ndarray:
    """Turn the BOX BOUNDS block back into cell vectors.

    A triclinic dump writes the bounding box of the tilted cell:

        xlo_bound = xlo + min(0, xy, xz, xy+xz)
        xhi_bound = xhi + max(0, xy, xz, xy+xz)
        ylo_bound = ylo + min(0, yz)
        yhi_bound = yhi + max(0, yz)

    so recovering the edges means subtracting those same terms back off.
    """
    values = [[float(v) for v in line.split()] for line in box_lines]
    triclinic = "xy" in header

    if triclinic:
        (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo, zhi, yz) = values
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
    else:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = values
        xy = xz = yz = 0.0

    return np.array([
        [xhi - xlo, 0.0, 0.0],
        [xy, yhi - ylo, 0.0],
        [xz, yz, zhi - zlo],
    ])


def read_frames(path: Path):
    """Yield (timestep, cell, columns, rows) for each frame."""
    with path.open() as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"expected 'ITEM: TIMESTEP', got {line.strip()!r}")
            timestep = int(fh.readline().strip())

            fh.readline()                                  # ITEM: NUMBER OF ATOMS
            natoms = int(fh.readline().strip())

            box_header = fh.readline().strip()
            box_lines = [fh.readline().strip() for _ in range(3)]
            cell = parse_box(box_lines, box_header)

            atoms_header = fh.readline().strip()
            columns = atoms_header.split()[2:]
            rows = [fh.readline().split() for _ in range(natoms)]
            yield timestep, cell, columns, rows


__all__ = ["parse_box", "read_frames"]
