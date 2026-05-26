#!/usr/bin/env python3
"""
Generate a LAMMPS dump file with a hexagonal lattice, spin precession,
and atomic-position oscillation.  No input files are read — everything
is built from the primitive cell (one atom at the origin).

Both spin and position waves support finite wavevectors (propagating
modes) in addition to the q=0 uniform case.

Usage examples:

  # Uniform (q=0) spin precession at 4 THz, static positions
  python generate_oscillating_dump.py --supercell 10 10 1 --spin-freqs 4.0

  # Propagating spin wave at the M point (1/2, 1/2, 0)
  python generate_oscillating_dump.py --supercell 20 20 1   \
      --spin-freqs 4.0 --spin-wavevector 0.5 0.5 0.0

  # Spin + position oscillation, both at finite q
  python generate_oscillating_dump.py --supercell 20 20 1   \
      --spin-freqs 4.0 --spin-wavevector 0.5 0.5 0.0        \
      --pos-dir y --pos-freq 4.0 --pos-wavevector 0.0 0.5 0.0
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


def reciprocal_lattice(real_lattice: np.ndarray) -> np.ndarray:
    """Return reciprocal lattice vectors (rows) for a given real-space lattice."""
    a1, a2, a3 = real_lattice
    vol = float(np.dot(a1, np.cross(a2, a3)))
    return np.array(
        [
            2.0 * np.pi * np.cross(a2, a3) / vol,
            2.0 * np.pi * np.cross(a3, a1) / vol,
            2.0 * np.pi * np.cross(a1, a2) / vol,
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
    parser.add_argument(
        "--spin-wavevector", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        metavar=("QX", "QY", "QZ"),
        help="Spin wavevector in primitive-cell fractional reciprocal coords "
        "(default 0 0 0 = uniform q=0 mode)",
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
    parser.add_argument(
        "--pos-wavevector", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        metavar=("QX", "QY", "QZ"),
        help="Position wavevector in primitive-cell fractional reciprocal coords "
        "(default 0 0 0 = uniform q=0 mode)",
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
    prim_lattice = build_primitive_lattice(args.lattice_constant, args.layer_spacing)
    prim_recip = reciprocal_lattice(prim_lattice)
    super_lattice = build_supercell_lattice(prim_lattice, nx, ny, nz)
    positions_eq = generate_atomic_positions(prim_lattice, nx, ny, nz)
    natoms = len(positions_eq)

    # --- spin frequencies, amplitudes & wavevector ----------------------------
    spin_freqs = np.asarray(args.spin_freqs, dtype=float)
    if args.spin_amps is None:
        spin_amps = np.ones_like(spin_freqs)
    else:
        spin_amps = np.asarray(args.spin_amps, dtype=float)
        if len(spin_amps) != len(spin_freqs):
            raise ValueError("--spin-amps must have the same length as --spin-freqs")

    spin_q_frac = np.asarray(args.spin_wavevector, dtype=float)
    spin_q_cart = spin_q_frac @ prim_recip
    omega_spin = 2.0 * np.pi * spin_freqs * 1e12  # rad / s

    # precompute q·r_i for each atom (spin)
    spin_qr = positions_eq @ spin_q_cart  # shape (natoms,)

    # --- position oscillation -------------------------------------------------
    pos_dir_idx: int | None = (
        {"x": 0, "y": 1, "z": 2}.get(args.pos_dir) if args.pos_dir else None
    )
    omega_pos = 2.0 * np.pi * args.pos_freq * 1e12  # rad / s

    pos_q_frac = np.asarray(args.pos_wavevector, dtype=float)
    pos_q_cart = pos_q_frac @ prim_recip
    pos_qr = positions_eq @ pos_q_cart  # shape (natoms,)

    # --- time step -----------------------------------------------------------
    dt_s = args.dt_fs * 1e-15

    # --- box bounds ----------------------------------------------------------
    box_lines = build_box_bounds(super_lattice)
    atom_header = (
        "ITEM: ATOMS type element x y z "
        "c_outsp[1] c_outsp[2] c_outsp[3]"
    )

    # --- report --------------------------------------------------------------
    print(f"Primitive lattice (rows):\n{prim_lattice}")
    print(f"Primitive reciprocal lattice (rows):\n{prim_recip}")
    print(f"Supercell : {nx} x {ny} x {nz}")
    print(f"Natoms    : {natoms}")
    print(f"Spin freqs (THz)      : {spin_freqs.tolist()}")
    print(f"Spin amps             : {spin_amps.tolist()}")
    print(f"Spin wavevector (frac): {spin_q_frac.tolist()}")
    print(f"Spin wavevector (cart): {spin_q_cart.tolist()}")
    if pos_dir_idx is not None:
        print(f"Pos dir    : {args.pos_dir}")
        print(f"Pos freq   : {args.pos_freq} THz, amp = {args.pos_amp} Å")
        print(f"Pos wavevector (frac): {pos_q_frac.tolist()}")
        print(f"Pos wavevector (cart): {pos_q_cart.tolist()}")
    else:
        print("Pos oscillation : none (static positions)")
    print(
        f"dt = {args.dt_fs} fs  |  nframes = {args.nframes}  |  "
        f"total time = {args.nframes * args.dt_fs / 1000:.1f} ps"
    )
    print(f"Nyquist freq     : {0.5 / dt_s / 1e12:.1f} THz")
    print(f"Freq resolution  : {1.0 / (args.nframes * dt_s) / 1e12:.4f} THz")

    # --- write frames --------------------------------------------------------
    with open(args.output, "w") as fh:
        for frame in range(args.nframes):
            t = frame * dt_s

            fh.write("ITEM: TIMESTEP\n")
            fh.write(f"{frame}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n")
            fh.write(f"{natoms}\n")
            fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            for line in box_lines:
                fh.write(line + "\n")
            fh.write(atom_header + "\n")

            for i, pos_eq in enumerate(positions_eq):
                x, y, z = pos_eq

                # spin: circular precession in xy-plane, Sz = 0
                #   phase_i = q·r_i - ω t
                spin_phase = spin_qr[i]
                sx = float(np.sum(
                    spin_amps * np.cos(spin_phase - omega_spin * t)
                ))
                sy = float(np.sum(
                    spin_amps * np.sin(spin_phase - omega_spin * t)
                ))

                # position displacement
                #   phase_i = q·r_i - ω t
                if pos_dir_idx is not None:
                    pos_phase = pos_qr[i] - omega_pos * t
                    disp = args.pos_amp * np.sin(pos_phase)
                else:
                    disp = 0.0

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
