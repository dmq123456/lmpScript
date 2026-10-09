#!/usr/bin/env python3
"""Build a spin-spiral structure and write it as a one-frame LAMMPS dump.

    make_spiral.py --supercell 30 30 1 --q 0 0.5 0 --e 1 0 0 --s0 0 1 0 \\
                   --spin-length 2.0 --out spiral.lammpstrj

The primitive cell is built into this file, so the script travels on its own --
no lmp.data needs to come with it.

The spiral is built by rotating a reference spin about a fixed axis, through an
angle that advances with position:

    theta_j = k . r_j ,   k = q @ B_prim
    S_j     = s0 cos(theta) + (e x s0) sin(theta) + e (e.s0)(1 - cos(theta))

That is Rodrigues' formula, so `s0` need not be perpendicular to `e`: a
perpendicular `s0` gives a flat spiral, and anything else gives a cone whose
opening angle is the angle between them.

The dump names its spin columns c_outsp[1..3] and stores unit directions, the
same convention `compute property/atom spx spy spz` produces, so the file drops
straight into the animation tools. The magnitude, which LAMMPS keeps in sp[3]
and does not dump, is written as an extra column so nothing is lost on the way
to an extxyz file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# The primitive cell, transcribed from lmp.data
# ----------------------------------------------------------------------
# LAMMPS restricted triclinic: a = (lx,0,0), b = (xy,ly,0), c = (xz,yz,lz)
LX, LY, LZ = 3.979286, 3.446163, 80.017130
XY, XZ, YZ = -1.989643, 0.0, 0.0

PRIM_CELL = np.array([
    [LX,  0.0, 0.0],
    [XY,  LY,  0.0],
    [XZ,  YZ,  LZ],
])

# id, type, x, y, z  -- taken verbatim from the Atoms section
BASIS = [
    (1, 0.00000000000000, 0.00000000000000, 41.35230297125722),
    (2, 1.98964306847537, 1.14872096103045, 39.80311619330681),
    (2, 0.00000001989645, 2.29744191057366, 42.90276100535079),
]
ELEMENTS = {1: "Ni", 2: "I"}
MAGNETIC_TYPE = 1          # only Ni carries a moment


def reciprocal(cell: np.ndarray) -> np.ndarray:
    """Rows of the reciprocal lattice, 2*pi convention."""
    return 2.0 * np.pi * np.linalg.inv(cell).T


def check_commensurate(q: np.ndarray, supercell: np.ndarray, tol: float = 1e-6) -> None:
    """A spiral must close on itself across the supercell.

    k . A_i = 2*pi n_i * q_i for supercell vectors A_i = n_i a_i, so the spiral
    is periodic exactly when every n_i q_i is an integer. Fail loudly: an
    incommensurate q puts a seam of mismatched spins at the box face, and under
    periodic boundary conditions that seam is a domain wall the model will
    happily compute forces for.

    The tolerance is loose on purpose. A q such as -1/15 cannot be typed
    exactly, and a value written to ten places misses a whole number by about
    1e-9 -- while a genuinely incommensurate q, say 0.33 on a 30-cell box,
    misses it by 0.1. Five orders of magnitude separate the two cases, so a
    tolerance anywhere between them rejects the right things. Use --q-index to
    sidestep the question entirely.
    """
    products = supercell * q
    bad = [i for i, value in enumerate(products) if abs(value - round(value)) > tol]
    if not bad:
        return
    axes = "abc"
    detail = ", ".join(
        f"{supercell[i]} x {q[i]:g} = {products[i]:g}" for i in bad
    )
    suggestions = ", ".join(
        f"{round(products[i]) / supercell[i]:g} along {axes[i]}" for i in bad
    )
    raise SystemExit(
        f"error: q is not commensurate with the supercell ({detail}).\n"
        f"       supercell[i] * q[i] must be a whole number on every axis, or the\n"
        f"       spiral does not match itself across the periodic boundary.\n"
        f"       Nearest values that would work: {suggestions}"
    )


def rodrigues(s0: np.ndarray, axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate s0 about axis by each angle in theta. Returns (len(theta), 3)."""
    axis = axis / np.linalg.norm(axis)
    cross = np.cross(axis, s0)
    along = axis * float(np.dot(axis, s0))
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    return s0 * cos_t + cross * sin_t + along * (1.0 - cos_t)


def polarization(s0: np.ndarray, axis: np.ndarray, k: np.ndarray,
                 theta: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """P = M x (curl M) / |M|^2, one vector per atom.

    The spiral is an analytic field, so the curl is taken in closed form rather
    than by differencing the lattice. Writing the Rodrigues rotation as

        u(theta) = A cos(theta) + B sin(theta) + C ,   theta = k . r

    with A = s0 - e (e.s0), B = e x s0 and C = e (e.s0), every derivative in
    space carries a factor k, so

        du/dtheta = -A sin(theta) + B cos(theta)
        curl M    = M_s (k x du/dtheta)

    and the magnitude cancels against |M|^2, leaving P = u x (k x du/dtheta) in
    units of 1/Angstrom. It vanishes for a proper screw, where k is parallel to
    the rotation axis.
    """
    axis = axis / np.linalg.norm(axis)
    perpendicular = s0 - axis * float(np.dot(axis, s0))
    turned = np.cross(axis, s0)
    derivative = (-perpendicular * np.sin(theta)[:, None]
                  + turned * np.cos(theta)[:, None])
    return np.cross(directions, np.cross(k, derivative))


def build(supercell: np.ndarray, q: np.ndarray, axis: np.ndarray,
          s0: np.ndarray, spin_length: float):
    """Replicate the cell and lay the spiral over it."""
    cell = PRIM_CELL * supercell[:, None]
    k = q @ reciprocal(PRIM_CELL)          # Cartesian, 1/Angstrom

    types, positions = [], []
    for i in range(supercell[0]):
        for j in range(supercell[1]):
            for m in range(supercell[2]):
                shift = i * PRIM_CELL[0] + j * PRIM_CELL[1] + m * PRIM_CELL[2]
                for atom_type, x, y, z in BASIS:
                    types.append(atom_type)
                    positions.append(np.array([x, y, z]) + shift)

    types = np.asarray(types, dtype=int)
    positions = np.asarray(positions, dtype=float)

    theta = positions @ k
    directions = rodrigues(np.asarray(s0, dtype=float), np.asarray(axis, dtype=float), theta)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    magnetic = types == MAGNETIC_TYPE
    lengths = np.where(magnetic, spin_length, 0.0)
    # Non-magnetic sites keep a placeholder direction; their length is zero, so
    # it never enters anything.
    directions[~magnetic] = np.array([0.0, 0.0, 1.0])

    return cell, types, positions, directions, lengths, k, theta


def write_dump(path: Path, cell, types, positions, directions, lengths,
               timestep: int = 0, polarizations=None) -> None:
    """One frame, in the column layout the LAMMPS runs produce."""
    lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]
    xy, xz, yz = cell[1, 0], cell[2, 0], cell[2, 1]

    # A triclinic dump reports the bounding box, not the cell edges.
    xlo_b = min(0.0, xy, xz, xy + xz)
    xhi_b = lx + max(0.0, xy, xz, xy + xz)
    ylo_b = min(0.0, yz)
    yhi_b = ly + max(0.0, yz)

    with path.open("w") as fh:
        fh.write("ITEM: TIMESTEP\n")
        fh.write(f"{timestep}\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{len(types)}\n")
        fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        fh.write(f"{xlo_b:.16e} {xhi_b:.16e} {xy:.16e}\n")
        fh.write(f"{ylo_b:.16e} {yhi_b:.16e} {xz:.16e}\n")
        fh.write(f"{0.0:.16e} {lz:.16e} {yz:.16e}\n")
        fh.write("ITEM: ATOMS id type element x y z "
                 "c_outsp[1] c_outsp[2] c_outsp[3] sp_len"
                 + (" px py pz\n" if polarizations is not None else "\n"))
        for i, (t, r, s, mag) in enumerate(zip(types, positions, directions, lengths), start=1):
            fh.write(f"{i} {t} {ELEMENTS[t]} "
                     f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} "
                     f"{s[0]:.8f} {s[1]:.8f} {s[2]:.8f} {mag:.6f}")
            if polarizations is not None:
                p = polarizations[i - 1]
                fh.write(f" {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}")
            fh.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_spiral.py",
        description="Generate a spin-spiral structure as a one-frame LAMMPS dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--supercell", type=int, nargs=3, metavar=("N1", "N2", "N3"),
                        default=(30, 30, 1), help="Cell repeats along a, b, c")
    parser.add_argument("--q-index", type=int, nargs=3, metavar=("N1", "N2", "N3"),
                        default=None,
                        help="The modulation as whole numbers of turns across the "
                             "supercell, so q = index / supercell. This is the form the "
                             "commensurability condition is stated in, and the form "
                             "spin_profile.py reports, so a measured q can be fed straight "
                             "back in. Takes precedence over --q")
    parser.add_argument("--q", type=float, nargs=3, metavar=("H", "K", "L"),
                        default=(0.0, 0.5, 0.0),
                        help="Propagation vector in primitive reciprocal fractional "
                             "coordinates. 0 0.5 0 is the M point")
    parser.add_argument("--e", type=float, nargs=3, metavar=("EX", "EY", "EZ"),
                        default=(1.0, 0.0, 0.0),
                        help="Rotation axis; spins turn about it as the spiral advances")
    parser.add_argument("--s0", type=float, nargs=3, metavar=("SX", "SY", "SZ"),
                        default=(0.0, 1.0, 0.0),
                        help="Spin direction at the origin, normalised on input. "
                             "Perpendicular to --e gives a flat spiral; otherwise a cone")
    parser.add_argument("--spin-length", type=float, default=2.0,
                        help="Magnitude on the magnetic sites; non-magnetic sites get 0")
    parser.add_argument("--polarization", action="store_true",
                        help="Also write px, py, pz for every magnetic atom, the local "
                             "P = M x (curl M) / |M|^2 of the spiral in 1/Angstrom; "
                             "non-magnetic sites get zeros")
    parser.add_argument("--out", type=Path, default=Path("spiral.lammpstrj"),
                        help="Output dump")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    supercell = np.asarray(args.supercell, dtype=int)
    if np.any(supercell < 1):
        sys.exit("error: --supercell entries must be positive")
    if args.q_index is not None:
        q = np.asarray(args.q_index, dtype=float) / supercell
    else:
        q = np.asarray(args.q, dtype=float)
    axis = np.asarray(args.e, dtype=float)
    s0 = np.asarray(args.s0, dtype=float)
    if np.linalg.norm(axis) < 1e-12:
        sys.exit("error: --e must be a non-zero vector")
    if np.linalg.norm(s0) < 1e-12:
        sys.exit("error: --s0 must be a non-zero vector")
    s0 = s0 / np.linalg.norm(s0)

    check_commensurate(q, supercell)

    cell, types, positions, directions, lengths, k, theta = build(
        supercell, q, axis, s0, args.spin_length
    )

    polarizations = None
    if args.polarization:
        polarizations = polarization(s0, axis, k, theta, directions)
        polarizations[types != MAGNETIC_TYPE] = 0.0

    cone = np.degrees(np.arccos(abs(np.dot(s0, axis / np.linalg.norm(axis)))))
    wavelength = 2 * np.pi / np.linalg.norm(k) if np.linalg.norm(k) > 1e-12 else float("inf")

    print(f"[INFO] Supercell      : {tuple(supercell)}  ->  {len(types)} atoms "
          f"({int((types == MAGNETIC_TYPE).sum())} magnetic)")
    turns = q * supercell
    print(f"[INFO] q (fractional) : {q}")
    print(f"[INFO] q (index)      : ({turns[0]:+.0f},{turns[1]:+.0f},{turns[2]:+.0f})"
          f"/({supercell[0]},{supercell[1]},{supercell[2]})   turns across the box")
    print(f"[INFO] k (Cartesian)  : {np.round(k, 6)} 1/A")
    print(f"[INFO] Wavelength     : {wavelength:.4f} A")
    print(f"[INFO] Rotation axis  : {axis / np.linalg.norm(axis)}")
    print(f"[INFO] s0             : {s0}")
    print(f"[INFO] Cone half-angle: {90.0 - cone:.2f} deg from the rotation plane "
          f"({'flat spiral' if abs(90.0 - cone) < 1e-6 else 'conical'})")
    print(f"[INFO] Spin length    : {args.spin_length} on type {MAGNETIC_TYPE}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_dump(args.out, cell, types, positions, directions, lengths,
               polarizations=polarizations)
    print(f"[INFO] Written        : {args.out} "
          f"({args.out.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
