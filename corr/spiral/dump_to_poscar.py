#!/usr/bin/env python3
"""Convert a LAMMPS dump frame into a VASP POSCAR plus a MAGMOM file.

    dump_to_poscar.py ppos9.lammpstrj ppos9

writes ppos9.POSCAR and ppos9.MAGMOM.

The two files have to be read together, because a POSCAR carries no magnetic
information and a MAGMOM line carries no identity: it is a bare list whose
meaning is entirely positional. So the order is fixed once, here, and the same
order is used for both. A POSCAR groups atoms by species while a LAMMPS dump
interleaves them, so this is a real reordering, not a copy -- writing MAGMOM in
dump order against a POSCAR in species order would put iodine moments on nickel
and would still run.

Moments are written for a non-collinear calculation, three numbers per atom, in
the same Cartesian frame as the cell. Their magnitude comes from the dump's
length column: the spin columns hold unit directions, and non-magnetic sites
carry a placeholder direction with zero length, which becomes three zeros.

A triclinic dump reports the bounding box of the tilted cell rather than its
edges, which `dumpio.parse_box` undoes. Positions are written as direct
coordinates wrapped into [0, 1), so an atom that LAMMPS left outside the box
lands where VASP expects it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dumpio import read_frames  # noqa: E402

DEFAULT_SPIN_COLUMNS = ("c_outsp[1]", "c_outsp[2]", "c_outsp[3]")
DEFAULT_LENGTH_COLUMN = "sp_len"


def column_index(columns: list[str], name: str) -> int:
    if name not in columns:
        raise SystemExit(
            f"error: the dump has no column {name!r}. Available: {', '.join(columns)}"
        )
    return columns.index(name)


def species_of(columns: list[str], rows: list[list[str]], type_map) -> np.ndarray:
    """One element symbol per atom, from the element column or from --type-map."""
    if type_map:
        index = column_index(columns, "type")
        types = np.array([int(r[index]) for r in rows])
        if types.max() > len(type_map):
            raise SystemExit(
                f"error: the dump holds atom type {types.max()} but --type-map gives only "
                f"{len(type_map)} symbol(s)."
            )
        return np.array([type_map[t - 1] for t in types])
    if "element" in columns:
        index = columns.index("element")
        return np.array([r[index] for r in rows])
    raise SystemExit(
        "error: the dump carries no 'element' column, so the species are unknown. "
        "Pass --type-map with one symbol per atom type, e.g. --type-map Ni I."
    )


def ordered_species(species: np.ndarray, wanted) -> list[str]:
    """The species in the order POSCAR will list them."""
    if wanted:
        missing = sorted(set(species) - set(wanted))
        if missing:
            raise SystemExit(
                f"error: --species does not mention {', '.join(missing)}, which the dump "
                "contains. Every species has to be listed."
            )
        return [s for s in wanted if s in set(species)]
    seen: list[str] = []
    for symbol in species:                       # order of first appearance
        if symbol not in seen:
            seen.append(symbol)
    return seen


def write_poscar(path: Path, comment: str, cell: np.ndarray, species: list[str],
                 counts: list[int], direct: np.ndarray) -> None:
    with path.open("w") as fh:
        fh.write(f"{comment}\n")
        fh.write("1.0\n")
        for vector in cell:
            fh.write(f"  {vector[0]:22.16f} {vector[1]:22.16f} {vector[2]:22.16f}\n")
        fh.write("  " + "  ".join(f"{s:>4s}" for s in species) + "\n")
        fh.write("  " + "  ".join(f"{c:>4d}" for c in counts) + "\n")
        fh.write("Direct\n")
        for position in direct:
            fh.write(f"  {position[0]:20.16f} {position[1]:20.16f} {position[2]:20.16f}\n")


def write_magmom(path: Path, poscar: Path, species: list[str], counts: list[int],
                 moments: np.ndarray) -> None:
    layout = "  ".join(f"{s}x{c}" for s, c in zip(species, counts))
    with path.open("w") as fh:
        fh.write(f"# MAGMOM for {poscar.name}: {moments.shape[0]} atoms, non-collinear,\n")
        fh.write(f"# three numbers per atom, in the POSCAR's own order ({layout}).\n")
        fh.write("# Pair it with, in INCAR:\n")
        fh.write("#     LNONCOLLINEAR = .TRUE.\n")
        fh.write("#     ISPIN = 2\n")
        fh.write("# The moments are Cartesian in the frame of the cell above, which is what\n")
        fh.write("# the default SAXIS = 0 0 1 means; set SAXIS only if you want another one.\n")
        fh.write("MAGMOM = " + " ".join(f"{v:.6f}" for v in moments.reshape(-1)) + "\n")
        fh.write("\n# One atom per line, for reading. Not INCAR syntax.\n")
        cursor = 0
        for symbol, count in zip(species, counts):
            for _ in range(count):
                m = moments[cursor]
                fh.write(f"#   {cursor + 1:5d}  {symbol:<3s} {m[0]:+.6f} {m[1]:+.6f} "
                         f"{m[2]:+.6f}   |m| = {np.linalg.norm(m):.6f}\n")
                cursor += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dump_to_poscar.py",
        description="Write a POSCAR and a matching MAGMOM from one dump frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Input dump")
    parser.add_argument("stem", type=Path,
                        help="Output stem: writes <stem>.POSCAR and <stem>.MAGMOM")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to convert")
    parser.add_argument("--spin-columns", nargs=3, default=list(DEFAULT_SPIN_COLUMNS),
                        metavar=("SX", "SY", "SZ"),
                        help="Columns holding the unit spin direction")
    parser.add_argument("--length-column", default=DEFAULT_LENGTH_COLUMN, metavar="COL",
                        help="Column holding the moment magnitude")
    parser.add_argument("--spin-length", type=float, default=None, metavar="M",
                        help="Constant magnitude to use instead, for a dump that carries "
                             "no length column. Applies to every atom, so use --magnetic "
                             "to say which ones are magnetic")
    parser.add_argument("--magnetic", nargs="+", default=None, metavar="EL",
                        help="Element symbols that carry a moment; the rest get zeros. "
                             "Only meaningful with --spin-length")
    parser.add_argument("--type-map", nargs="+", default=None, metavar="EL",
                        help="Element symbol per atom type, when the dump has no "
                             "'element' column")
    parser.add_argument("--species", nargs="+", default=None, metavar="EL",
                        help="Species order for the POSCAR, e.g. --species Ni I. "
                             "Defaults to order of first appearance, and must match the "
                             "order the POTCAR is concatenated in")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    frame = None
    for index, record in enumerate(read_frames(args.source)):
        if index == args.frame:
            frame = record
            break
    if frame is None:
        raise SystemExit(f"error: frame {args.frame} not found in {args.source}.")
    timestep, cell, columns, rows = frame

    positions = np.array([[float(r[column_index(columns, axis)]) for axis in "xyz"]
                          for r in rows])
    species = species_of(columns, rows, args.type_map)

    directions = np.array([[float(r[column_index(columns, c)]) for c in args.spin_columns]
                           for r in rows])
    if args.spin_length is not None:
        lengths = np.full(len(rows), float(args.spin_length))
        if args.magnetic:
            lengths[~np.isin(species, args.magnetic)] = 0.0
    else:
        index = column_index(columns, args.length_column)
        lengths = np.array([float(r[index]) for r in rows])
    moments = directions * lengths[:, None]

    # A POSCAR lists one species at a time; the dump interleaves them. Sorting
    # by species with a stable sort keeps each species in its original dump
    # order, so the mapping stays easy to check by eye against the dump.
    order_of = {symbol: i for i, symbol in enumerate(ordered_species(species, args.species))}
    order = np.argsort([order_of[s] for s in species], kind="stable")
    species, positions, moments = species[order], positions[order], moments[order]

    listed = ordered_species(species, args.species)
    counts = [int((species == symbol).sum()) for symbol in listed]

    direct = positions @ np.linalg.inv(cell)
    direct -= np.floor(direct)

    poscar = args.stem.with_suffix(".POSCAR")
    magmom = args.stem.with_suffix(".MAGMOM")
    formula = " ".join(f"{s}{c}" for s, c in zip(listed, counts))
    write_poscar(poscar, f"{formula}   from {args.source.name} frame {args.frame} "
                         f"(timestep {timestep})", cell, listed, counts, direct)
    write_magmom(magmom, poscar, listed, counts, moments)

    magnetic = np.linalg.norm(moments, axis=1) > 0.0
    print(f"[INFO] Frame          : {args.frame} (timestep {timestep})")
    print(f"[INFO] Cell           : {np.array2string(cell, precision=6)}")
    print(f"[INFO] Composition    : {formula}   ({len(species)} atoms)")
    print(f"[INFO] Magnetic sites : {int(magnetic.sum())}, "
          f"|m| = {np.linalg.norm(moments[magnetic], axis=1).mean():.4f} "
          f"(sum of moments = {np.array2string(moments.sum(axis=0), precision=6)})")
    print(f"[INFO] Written        : {poscar}")
    print(f"[INFO] Written        : {magmom}")


if __name__ == "__main__":
    main()
