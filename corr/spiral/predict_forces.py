#!/usr/bin/env python3
"""Run the spin model over an extxyz and write the forces back out as a dump.

    predict_forces.py spiral.extxyz forces.lammpstrj --model deployed2-1.pth

The output keeps the column layout of the input structure and appends the two
force fields the model predicts, so it stays readable by the same animation and
analysis tools:

    id type element x y z c_outsp[1..3] sp_len fx fy fz sfx sfy sfz

With --bec the Born effective charges follow as c_outbec[1..9], the row-major
3x3 tensor of each atom, the same nine columns the LAMMPS runs dump.

`sf` is `spin_forces_vert`, the component of the spin force perpendicular to
the moment. That is the part that actually turns a spin: any component along
the moment only changes its length, which the model holds fixed. The full
`spin_forces` is available through --spin-force full if the parallel part is
wanted too.

No LAMMPS is involved. The neighbour list the model needs is built by
`AtomicData.from_ase` from the positions, the cutoff and the periodic boundary
-- exactly as it is during training and validation, so the numbers come from
the same code path that produced the fitted model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from ase.io import read
    from nequip.ase import NequIPCalculator
    from nequip.data import AtomicData, AtomicDataDict
except ImportError as exc:
    sys.exit(f"error: {exc}\n       Try:  conda activate spingnn")


def resolve_device(requested: str) -> str:
    """Pick the device to run on.

    NequIPCalculator defaults to the CPU and stays there whatever hardware is
    present, so a GPU has to be asked for. 'auto' takes one when it exists,
    which is what makes the same command sensible on a laptop and on a node.
    """
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"error: {requested} requested but torch reports no CUDA device")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


SPIN_FORCE_KEYS = {
    "vert": "spin_forces_vert",
    "full": "spin_forces",
}

BEC_KEY = "born_effective_charges"
BEC_COLUMNS = " ".join(f"c_outbec[{i}]" for i in range(1, 10))


def cell_to_bounds(cell: np.ndarray):
    """Cell vectors -> the bounding box a triclinic dump reports."""
    lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]
    xy, xz, yz = cell[1, 0], cell[2, 0], cell[2, 1]
    return (
        (min(0.0, xy, xz, xy + xz), lx + max(0.0, xy, xz, xy + xz), xy),
        (min(0.0, yz), ly + max(0.0, yz), xz),
        (0.0, lz, yz),
    )


def write_dump(path: Path, frames: list[dict]) -> None:
    with path.open("w") as fh:
        for frame in frames:
            atoms = frame["atoms"]
            cell = np.asarray(atoms.cell[:])
            (xb, yb, zb) = cell_to_bounds(cell)

            fh.write("ITEM: TIMESTEP\n")
            fh.write(f"{frame['index']}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n")
            fh.write(f"{len(atoms)}\n")
            fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            for lo, hi, tilt in (xb, yb, zb):
                fh.write(f"{lo:.16e} {hi:.16e} {tilt:.16e}\n")
            bec = frame.get("bec")
            fh.write("ITEM: ATOMS id type element x y z "
                     "c_outsp[1] c_outsp[2] c_outsp[3] sp_len "
                     "fx fy fz sfx sfy sfz"
                     + (f" {BEC_COLUMNS}\n" if bec is not None else "\n"))

            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            spins = frame["spins"]
            lengths = frame["lengths"]
            forces = frame["forces"]
            spin_forces = frame["spin_forces"]
            type_of = frame["type_of"]

            for i in range(len(atoms)):
                fh.write(
                    f"{i + 1} {type_of[symbols[i]]} {symbols[i]} "
                    f"{positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f} "
                    f"{spins[i, 0]:.8f} {spins[i, 1]:.8f} {spins[i, 2]:.8f} "
                    f"{lengths[i]:.6f} "
                    f"{forces[i, 0]:.8f} {forces[i, 1]:.8f} {forces[i, 2]:.8f} "
                    f"{spin_forces[i, 0]:.8f} {spin_forces[i, 1]:.8f} {spin_forces[i, 2]:.8f}"
                )
                if bec is not None:
                    fh.write("".join(f" {value:.8f}" for value in bec[i]))
                fh.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_forces.py",
        description="Predict forces with the spin model and write them as a LAMMPS dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Input .extxyz")
    parser.add_argument("target", type=Path, help="Output dump")
    parser.add_argument("--model", type=Path, default=Path("deployed2-1.pth"),
                        help="Deployed model")
    parser.add_argument("--species", nargs="+", default=["Ni", "I"], metavar="EL",
                        help="Chemical symbols in model type order. Passed to the "
                             "calculator explicitly, so the element-to-type mapping is "
                             "never guessed from alphabetical order")
    parser.add_argument("--spin-force", choices=sorted(SPIN_FORCE_KEYS), default="vert",
                        help="Which spin force to write: 'vert' is the component "
                             "perpendicular to the moment, the part that turns it")
    parser.add_argument("--bec", action="store_true",
                        help="Also predict the Born effective charges and append them as "
                             "nine columns c_outbec[1..9], the row-major 3x3 tensor per "
                             "atom, in the same layout the LAMMPS runs dump")
    parser.add_argument("--frames", default=":", metavar="SLICE",
                        help="Frames to evaluate, as a Python slice")
    parser.add_argument("--energy-file", type=Path, default=None,
                        help="Also write one line of 'frame energy' per frame here")
    parser.add_argument("--device", default="auto",
                        help="Where to run: 'auto' uses a GPU when torch sees one, "
                             "or name it directly ('cpu', 'cuda', 'cuda:1')")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not args.source.exists():
        sys.exit(f"error: no such file: {args.source}")
    if not args.model.exists():
        sys.exit(f"error: no such model: {args.model}")

    mapping = {symbol: symbol for symbol in args.species}
    type_of = {symbol: i + 1 for i, symbol in enumerate(args.species)}

    device = resolve_device(args.device)
    calc = NequIPCalculator.from_deployed_model(
        model_path=str(args.model),
        device=device,
        set_global_options=True,
        species_to_type_name=mapping,
    )
    print(f"[INFO] Model        : {args.model}")
    print(f"[INFO] Device       : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else ""))
    print(f"[INFO] r_max        : {calc.r_max}")
    print(f"[INFO] Species map  : {mapping}")

    images = read(args.source, index=args.frames)
    if not isinstance(images, list):
        images = [images]
    print(f"[INFO] Frames       : {len(images)}")

    spin_key = SPIN_FORCE_KEYS[args.spin_force]
    frames = []
    energies = []
    started = time.perf_counter()

    for index, atoms in enumerate(images):
        if "initial_magmoms" not in atoms.arrays:
            sys.exit(
                f"error: frame {index} has no 'initial_magmoms'. The model is a spin "
                f"model; a structure without spins cannot be evaluated."
            )
        data = calc.transform(AtomicData.from_ase(atoms=atoms, r_max=calc.r_max))
        batch = AtomicData.to_AtomicDataDict(data)
        # from_ase builds its tensors on the CPU, so they have to follow the
        # model across; otherwise torch raises a device mismatch.
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        out = calc.model(batch)

        forces = out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        spin_forces = out[spin_key].detach().cpu().numpy()
        energy = float(out[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu())
        energies.append(energy)

        bec = None
        if args.bec:
            if BEC_KEY not in out:
                sys.exit(f"error: the model does not output '{BEC_KEY}'; "
                         f"it predicts {sorted(out)}")
            bec = out[BEC_KEY].detach().cpu().numpy()

        frames.append({
            "index": index,
            "atoms": atoms,
            "spins": atoms.arrays["initial_magmoms"],
            "lengths": atoms.arrays.get("spin_length", np.zeros(len(atoms))),
            "forces": forces,
            "spin_forces": spin_forces,
            "bec": bec,
            "type_of": type_of,
        })
        print(f"[INFO] frame {index:4d} : E = {energy:15.6f} eV   "
              f"max|F| = {np.abs(forces).max():.5f} eV/A   "
              f"max|sF| = {np.abs(spin_forces).max():.5f}")

    elapsed = time.perf_counter() - started
    print(f"[INFO] Evaluated    : {len(frames)} frame(s) in {elapsed:.1f} s")

    args.target.parent.mkdir(parents=True, exist_ok=True)
    write_dump(args.target, frames)
    print(f"[INFO] Columns      : id type element x y z c_outsp[1..3] sp_len "
          f"fx fy fz sfx sfy sfz" + (" c_outbec[1..9]" if args.bec else "")
          + f"    (sf = {spin_key})")
    print(f"[INFO] Written      : {args.target}")

    if args.energy_file is not None:
        np.savetxt(args.energy_file,
                   np.column_stack([np.arange(len(energies)), energies]),
                   fmt="%d %.10f", header="frame  total_energy_eV")
        print(f"[INFO] Energies     : {args.energy_file}")


if __name__ == "__main__":
    main()
