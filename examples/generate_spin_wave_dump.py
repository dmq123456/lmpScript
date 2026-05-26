#!/usr/bin/env python3

import math
from pathlib import Path


INPUT = Path("ideal_lattice_80_80_30.lammpstrj")
OUTPUT = Path("spin_wave_ideal_800.lammpstrj")
NFRAMES = 800
K = 4.0 * math.pi / 20.0
# Make frame 0 to frame 799 span exactly 10 periods.
W = 2.0 * math.pi * 10.0 / (NFRAMES - 1)


def read_header_and_atoms(path: Path):
    with path.open() as fh:
        if fh.readline().strip() != "ITEM: TIMESTEP":
            raise ValueError("unexpected dump format: missing first timestep header")
        first_timestep = fh.readline().strip()

        if fh.readline().strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError("unexpected dump format: missing atom-count header")
        natoms = int(fh.readline().strip())

        box_header = fh.readline().rstrip("\n")
        box_lines = [fh.readline().rstrip("\n") for _ in range(3)]

        atoms_header = fh.readline().rstrip("\n")
        columns = atoms_header.split()[2:]
        if columns[:5] != ["type", "element", "x", "y", "z"]:
            raise ValueError("unexpected atom columns")

        atoms = []
        for _ in range(natoms):
            fields = fh.readline().split()
            atoms.append(fields)

    return first_timestep, natoms, box_header, box_lines, atoms_header, atoms


def read_timesteps(path: Path, nframes: int):
    timesteps = []
    with path.open() as fh:
        while len(timesteps) < nframes:
            line = fh.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                ts = fh.readline().strip()
                timesteps.append(ts)
    if len(timesteps) == 0:
        raise ValueError("no timestep records found in input dump")
    if len(timesteps) < nframes:
        if len(timesteps) == 1:
            timesteps = [str(i) for i in range(nframes)]
        else:
            raise ValueError(f"requested {nframes} frames, found only {len(timesteps)}")
    return timesteps


def main():
    _, natoms, box_header, box_lines, atoms_header, atoms = read_header_and_atoms(INPUT)
    timesteps = read_timesteps(INPUT, NFRAMES)

    x_index = atoms_header.split()[2:].index("x")

    with OUTPUT.open("w") as out:
        for frame, timestep in enumerate(timesteps):
            phase_t = W * frame
            out.write("ITEM: TIMESTEP\n")
            out.write(f"{timestep}\n")
            out.write("ITEM: NUMBER OF ATOMS\n")
            out.write(f"{natoms}\n")
            out.write(f"{box_header}\n")
            for line in box_lines:
                out.write(f"{line}\n")
            out.write(f"{atoms_header}\n")

            for fields in atoms:
                x = float(fields[x_index])
                phase = K * x - phase_t
                sx = math.cos(phase)
                sy = math.sin(phase)
                out.write(
                    f"{fields[0]} {fields[1]} {fields[2]} {fields[3]} {fields[4]} "
                    f"{sx:.10f} {sy:.10f} 0.0000000000\n"
                )


if __name__ == "__main__":
    main()
