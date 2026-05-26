#!/usr/bin/env python3
"""
Generate a LAMMPS dump file with a hexagonal lattice, oscillating spins,
and oscillating atomic positions.  No input files are read – everything
is built from the primitive cell (one atom at the origin).

Usage examples:

  # Spin precession at 4 THz, static positions
  python generate_oscillating_dump.py --supercell 10 10 1 --spin-freqs 4.0

  # Spin at 4 THz  +  position oscillation along y at 4 THz (0.1 Å ampl.)
  python generate_oscillating_dump.py --supercell 10 10 1   \
      --spin-freqs 4.0 --pos-dir y --pos-freq 4.0 --pos-amp 0.1

  # Multiple spin frequencies  +  position oscillation along x
  python generate_oscillating_dump.py --supercell 20 20 1   \
      --spin-freqs 4.0 12.0 --spin-amps 1.0 0.3            \
      --pos-dir x --pos-freq 4.0 --pos-amp 0.05             \
      --nframes 5000 --dt-fs 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# lattice helpers
# ---------------------------------------------------------------------------

def build_primitive_lattice(a: float = 4.0, c: float = 80.0) -> np.ndarray:
    """Hexagonal (triangular) primitive lattice vectors (rows)."""
    return np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )


def build_supercell_lattice(
    prim_lattice: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    """Supercell lattice = prim_lattice scaled by (nx, ny, nz) per row."""
    return prim_lattice * np.array([nx, ny, nz], dtype=float)[:, None]


def generate_atomic_positions(
    prim_lattice: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    """Return (natoms, 3) equilibrium positions for the supercell."""
    positions: list[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = (
                    (i / nx) * (nx * prim_lattice[0])
                    + (j / ny) * (ny * prim_lattice[1])
                    + (k / nz) * (nz * prim_lattice[2])
                )
                positions.append(r)
    return np.array(positions, dtype=float)


def build_box_bounds(super_lattice: np.ndarray) -> list[str]:
    """LAMMPS triclinic BOX BOUNDS lines (xy xz yz pp pp pp)."""
    lx = super_lattice[0, 0]
    xy = super_lattice[1, 0]
    ly = super_lattice[1, 1]
    xz = super_lattice[2, 0]
    yz = super_lattice[2, 1]
    lz = super_lattice[2, 2]

    xlo_bound = 0.0 + min(0.0, xy, xz, xy + xz)
    xhi_bound = lx + max(0.0, xy, xz, xy + xz)
    ylo_bound = 0.0 + min(0.0, yz)
    yhi_bound = ly + max(0.0, yz)
    zlo_bound = 0.0
    zhi_bound = lz

    return [
        f"{xlo_bound:.16e} {xhi_bound:.16e} {xy:.16e}",
        f"{ylo_bound:.16e} {yhi_bound:.16e} {xz:.16e}",
        f"{zlo_bound:.16e} {zhi_bound:.16e} {yz:.16e}",
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LAMMPS dump with hexagonal lattice, "
        "spin precession, and position oscillation."
    )
    # lattice
    parser.add_argument(
        "--lattice-constant", type=float, default=4.0,
        help="Hexagonal in-plane lattice constant a (Angstrom), default 4.0",
    )
    parser.add_argument(
        "--layer-spacing", type=float, default=80.0,
        help="Interlayer spacing c (Angstrom), default 80.0",
    )
    parser.add_argument(
        "--supercell", type=int, nargs=3, default=[10, 10, 1],
        metavar=("NX", "NY", "NZ"),
        help="Supercell dimensions, default '10 10 1'",
    )

    # spin oscillation
    parser.add_argument(
        "--spin-freqs", type=float, nargs="+", default=[4.0],
        metavar="FREQ_THZ",
        help="Spin precession frequencies in THz, default 4.0",
    )
    parser.add_argument(
        "--spin-amps", type=float, nargs="+", default=None,
        metavar="AMP",
        help="Amplitude per spin frequency (default 1.0 each)",
    )

    # position oscillation
    parser.add_argument(
        "--pos-dir", type=str, default=None, choices=["x", "y", "z"],
        help="Direction for position oscillation.  Omit for static positions.",
    )
    parser.add_argument(
        "--pos-freq", type=float, default=4.0,
        help="Position oscillation frequency in THz, default 4.0",
    )
    parser.add_argument(
        "--pos-amp", type=float, default=0.1,
        help="Position oscillation amplitude in Angstrom, default 0.1",
    )

    # time axis
    parser.add_argument(
        "--dt-fs", type=float, default=2.0,
        help="Time step between frames in fs, default 2.0",
    )
    parser.add_argument(
        "--nframes", type=int, default=10000,
        help="Number of frames, default 10000",
    )

    # output
    parser.add_argument(
        "--output", type=str, default="oscillating.lammpstrj",
        help="Output file path, default 'oscillating.lammpstrj'",
    )

    args = parser.parse_args()

    nx, ny, nz = args.supercell

    # --- lattice & positions -------------------------------------------------
    prim_lattice = build_primitive_lattice(args.lattice_constant,
                                           args.layer_spacing)
    super_lattice = build_supercell_lattice(prim_lattice, nx, ny, nz)
    positions_eq = generate_atomic_positions(prim_lattice, nx, ny, nz)
    natoms = len(positions_eq)

    # --- spin frequencies & amplitudes ---------------------------------------
    spin_freqs = np.asarray(args.spin_freqs, dtype=float)
    if args.spin_amps is None:
        spin_amps = np.ones_like(spin_freqs)
    else:
        spin_amps = np.asarray(args.spin_amps, dtype=float)
        if len(spin_amps) != len(spin_freqs):
            raise ValueError(
                "--spin-amps must have the same length as --spin-freqs"
            )

    omega_spin = 2.0 * np.pi * spin_freqs * 1e12  # rad / s

    # --- position oscillation ------------------------------------------------
    pos_dir_idx: int | None = {"x": 0, "y": 1, "z": 2}.get(args.pos_dir) if args.pos_dir else None
    omega_pos = 2.0 * np.pi * args.pos_freq * 1e12  # rad / s

    dt_s = args.dt_fs * 1e-15

    # --- box bounds ----------------------------------------------------------
    box_lines = build_box_bounds(super_lattice)
    atom_header = (
        "ITEM: ATOMS type element x y z "
        "c_outsp[1] c_outsp[2] c_outsp[3]"
    )

    # --- report --------------------------------------------------------------
    print(f"Primitive lattice (rows):\n{prim_lattice}")
    print(f"Supercell : {nx} x {ny} x {nz}")
    print(f"Natoms    : {natoms}")
    print(f"Spin freqs (THz) : {spin_freqs.tolist()}")
    print(f"Spin amps        : {spin_amps.tolist()}")
    if pos_dir_idx is not None:
        print(f"Pos oscillation  : dir={args.pos_dir}, "
              f"freq={args.pos_freq} THz, amp={args.pos_amp} Å")
    else:
        print("Pos oscillation  : none (static positions)")
    print(f"dt = {args.dt_fs} fs  |  nframes = {args.nframes}  |  "
          f"total time = {args.nframes * args.dt_fs / 1000:.1f} ps")
    print(f"Nyquist freq     : {0.5 / dt_s / 1e12:.1f} THz")
    print(f"Freq resolution  : {1.0 / (args.nframes * dt_s) / 1e12:.4f} THz")

    # --- write frames --------------------------------------------------------
    with open(args.output, "w") as fh:
        for frame in range(args.nframes):
            t = frame * dt_s

            # spin: circular precession in xy-plane, Sz = 0
            sx = float(np.sum(spin_amps * np.cos(omega_spin * t)))
            sy = float(np.sum(spin_amps * np.sin(omega_spin * t)))

            # position displacement
            disp = args.pos_amp * np.sin(omega_pos * t) if pos_dir_idx is not None else 0.0

            fh.write("ITEM: TIMESTEP\n")
            fh.write(f"{frame}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n")
            fh.write(f"{natoms}\n")
            fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            for line in box_lines:
                fh.write(line + "\n")
            fh.write(atom_header + "\n")

            for pos_eq in positions_eq:
                x, y, z = pos_eq
                if pos_dir_idx == 0:
                    x += disp
                elif pos_dir_idx == 1:
                    y += disp
                elif pos_dir_idx == 2:
                    z += disp
                fh.write(
                    f"1 C {x:.10f} {y:.10f} {z:.10f} "
                    f"{sx:.10f} {sy:.10f} 0.0000000000\n"
                )

    print(f"Wrote {args.nframes} frames -> {args.output}")


if __name__ == "__main__":
    main()
