#!/usr/bin/env python3
"""Convert a LAMMPS dump into an extxyz the spin model can read.

    dump_to_extxyz.py spiral.lammpstrj spiral.extxyz

`AtomicData.from_ase` looks in `atoms.arrays` for `initial_magmoms` and
`spin_length`, both of which are in its default include_keys, so an extxyz
carrying those two arrays feeds the model with no further mapping. This script
writes exactly that.

Column conventions follow the LAMMPS runs: the spin columns hold unit
directions, as `compute property/atom spx spy spz` produces, and the magnitude
travels in a separate column. A dump straight out of LAMMPS has no magnitude
column at all -- LAMMPS keeps it in sp[3] and never dumps it -- so
--spin-length supplies a constant in that case.

The box needs undoing rather than copying: a triclinic dump reports the
bounding box of the tilted cell, not its edges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from ase import Atoms
    from ase.io import write as ase_write
except ImportError:
    sys.exit("error: this needs ASE. Try:  conda activate spingnn")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dumpio import parse_box, read_frames  # noqa: E402,F401


DEFAULT_SPIN_COLUMNS = ("c_outsp[1]", "c_outsp[2]", "c_outsp[3]")
DEFAULT_LENGTH_COLUMN = "sp_len"


def build_atoms(cell, columns, rows, args) -> Atoms:
    index = {name: i for i, name in enumerate(columns)}

    def column(name: str) -> np.ndarray:
        return np.array([float(row[index[name]]) for row in rows])

    missing = [c for c in ("x", "y", "z") if c not in index]
    if missing:
        raise SystemExit(f"error: dump has no {', '.join(missing)} column")

    positions = np.column_stack([column("x"), column("y"), column("z")])

    # Species: prefer the element column, else map the numeric type.
    if "element" in index:
        symbols = [row[index["element"]] for row in rows]
    elif "type" in index and args.type_map:
        mapping = {i + 1: s for i, s in enumerate(args.type_map)}
        types = [int(row[index["type"]]) for row in rows]
        unknown = sorted(set(types) - set(mapping))
        if unknown:
            raise SystemExit(
                f"error: --type-map covers types {sorted(mapping)} but the dump "
                f"also has {unknown}"
            )
        symbols = [mapping[t] for t in types]
    else:
        raise SystemExit(
            "error: no 'element' column; supply --type-map, e.g. --type-map Ni I"
        )

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    # Spin direction -> initial_magmoms, which from_ase maps to SPIN_KEY.
    spin_columns = tuple(args.spin_columns)
    if all(c in index for c in spin_columns):
        directions = np.column_stack([column(c) for c in spin_columns])
        norms = np.linalg.norm(directions, axis=1)
        nonzero = norms > 1e-12
        directions[nonzero] /= norms[nonzero, None]
        atoms.arrays["initial_magmoms"] = directions
    elif not args.allow_no_spin:
        raise SystemExit(
            f"error: dump has no spin columns {spin_columns}.\n"
            f"       columns present: {' '.join(columns)}\n"
            f"       Use --spin-columns to name them, or --allow-no-spin to skip."
        )

    # Magnitude -> spin_length, which from_ase maps to SPIN_LENGTH_KEY.
    if args.length_column in index:
        atoms.arrays["spin_length"] = column(args.length_column)
    elif args.spin_length is not None:
        magnetic = np.array([s in args.magnetic for s in symbols]) if args.magnetic \
            else np.ones(len(symbols), dtype=bool)
        atoms.arrays["spin_length"] = np.where(magnetic, args.spin_length, 0.0)
    elif "initial_magmoms" in atoms.arrays:
        raise SystemExit(
            f"error: no '{args.length_column}' column and no --spin-length.\n"
            f"       LAMMPS keeps the magnitude in sp[3] and never dumps it, so a\n"
            f"       dump written by a real run needs the value supplying here."
        )

    # Atoms without a moment (I) carry a placeholder unit direction in the dump,
    # because LAMMPS atom_style spin needs one; the LAMMPS pair style feeds them
    # to the model as zero (sp[3] = 0).  from_ase passes initial_magmoms as they
    # are, so zero them here, as in the training set.
    if "initial_magmoms" in atoms.arrays and "spin_length" in atoms.arrays:
        atoms.arrays["initial_magmoms"][atoms.arrays["spin_length"] <= 0] = 0.0
    return atoms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dump_to_extxyz.py",
        description="Convert a LAMMPS dump to extxyz with spins the model can read.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Input dump")
    parser.add_argument("target", type=Path, help="Output .extxyz")
    parser.add_argument("--spin-columns", nargs=3, default=list(DEFAULT_SPIN_COLUMNS),
                        metavar=("SX", "SY", "SZ"),
                        help="Dump columns holding the unit spin direction")
    parser.add_argument("--length-column", default=DEFAULT_LENGTH_COLUMN,
                        help="Dump column holding the spin magnitude, if present")
    parser.add_argument("--spin-length", type=float, default=None,
                        help="Magnitude to use when the dump carries no length column")
    parser.add_argument("--magnetic", nargs="+", default=None, metavar="EL",
                        help="Elements that carry a moment when --spin-length is used; "
                             "everything else gets zero. Default: all of them")
    parser.add_argument("--type-map", nargs="+", default=None, metavar="EL",
                        help="Element per numeric type, in order, for dumps without an "
                             "element column, e.g. --type-map Ni I")
    parser.add_argument("--frames", default=":", metavar="SLICE",
                        help="Frames to convert, as a Python slice, e.g. 0, :10, ::2")
    parser.add_argument("--allow-no-spin", action="store_true",
                        help="Proceed even if no spin columns are found")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not args.source.exists():
        sys.exit(f"error: no such file: {args.source}")

    selection = args.frames
    if ":" in selection:
        start, _, rest = selection.partition(":")
        stop, _, step = rest.partition(":")
        chosen = slice(int(start) if start else None,
                       int(stop) if stop else None,
                       int(step) if step else None)
    else:
        chosen = slice(int(selection), int(selection) + 1)

    frames = list(read_frames(args.source))[chosen]
    if not frames:
        sys.exit("error: no frames selected")

    images = [build_atoms(cell, columns, rows, args)
              for _, cell, columns, rows in frames]

    args.target.parent.mkdir(parents=True, exist_ok=True)
    ase_write(args.target, images, format="extxyz")

    first = images[0]
    print(f"[INFO] Frames         : {len(images)}")
    print(f"[INFO] Atoms          : {len(first)}")
    print(f"[INFO] Arrays written : {', '.join(sorted(first.arrays))}")
    if "spin_length" in first.arrays:
        lengths = first.arrays["spin_length"]
        magnetic = lengths > 0
        print(f"[INFO] Magnetic sites : {int(magnetic.sum())} of {len(first)}, "
              f"|S| = {np.unique(lengths[magnetic]) if magnetic.any() else 'none'}")
    print(f"[INFO] Cell           :\n{np.array2string(first.cell[:], precision=6, prefix='                        ')}")
    print(f"[INFO] Written        : {args.target}")


if __name__ == "__main__":
    main()
