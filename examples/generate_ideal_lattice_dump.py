#!/usr/bin/env python3

import numpy as np
from pathlib import Path


INPUT = Path("test.lammpstrj")
OUTPUT = Path("ideal_lattice_80_80_30.lammpstrj")
NXY = 20


def read_first_frame(path: Path):
    with path.open() as fh:
        if fh.readline().strip() != "ITEM: TIMESTEP":
            raise ValueError("unexpected dump format")
        timestep = fh.readline().strip()
        if fh.readline().strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError("unexpected dump format")
        natoms = int(fh.readline().strip())
        _ = fh.readline().rstrip("\n")
        for _ in range(3):
            fh.readline()
        atoms_header = fh.readline().rstrip("\n")
        atoms = [fh.readline().split() for _ in range(natoms)]
    return timestep, natoms, atoms_header, atoms


def main():
    timestep, natoms, atoms_header, atoms = read_first_frame(INPUT)
    xyz = np.array([[float(a[2]), float(a[3]), float(a[4])] for a in atoms], dtype=float)

    lattice = np.array(
        [
            [80.0, 0.0, 0.0],
            [-40.0, 69.28203230275509, 0.0],
            [0.0, 0.0, 30.0],
        ],
        dtype=float,
    )
    frac = np.linalg.solve(lattice.T, xyz.T).T
    frac[:, :2] %= 1.0
    frac[:, 2] %= 1.0

    low_mask = frac[:, 2] < 0.5
    high_mask = ~low_mask
    z_low = frac[low_mask, 2].mean()
    z_high = frac[high_mask, 2].mean()

    frac_ideal = frac.copy()
    frac_ideal[:, 0] = np.round(frac[:, 0] * NXY) / NXY
    frac_ideal[:, 1] = np.round(frac[:, 1] * NXY) / NXY
    frac_ideal[:, 0] %= 1.0
    frac_ideal[:, 1] %= 1.0
    frac_ideal[low_mask, 2] = z_low
    frac_ideal[high_mask, 2] = z_high

    xyz_ideal = frac_ideal @ lattice

    with OUTPUT.open("w") as out:
        out.write("ITEM: TIMESTEP\n")
        out.write(f"{timestep}\n")
        out.write("ITEM: NUMBER OF ATOMS\n")
        out.write(f"{natoms}\n")
        out.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        out.write("-4.0000000000000000e+01 8.0000000000000000e+01 -4.0000000000000000e+01\n")
        out.write("0.0000000000000000e+00 6.9282032302755092e+01 0.0000000000000000e+00\n")
        out.write("0.0000000000000000e+00 3.0000000000000000e+01 0.0000000000000000e+00\n")
        out.write(f"{atoms_header}\n")
        for atom, pos in zip(atoms, xyz_ideal):
            out.write(
                f"{atom[0]} {atom[1]} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f} "
                f"{atom[5]} {atom[6]} {atom[7]}\n"
            )


if __name__ == "__main__":
    main()
