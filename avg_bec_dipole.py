#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


TARGET_COLUMNS = [
    "c_bec_dipole[0]",
    "c_bec_dipole[1]",
    "c_bec_dipole[2]",
    "c_bec_dipole[3]",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a LAMMPS-style log file and compute the average values of "
            "c_bec_dipole[0:3]."
        )
    )
    parser.add_argument("logfile", type=Path, help="Input log file path")
    return parser


def parse_bec_dipole_averages(logfile: Path) -> tuple[dict[str, float], int]:
    sums = {column: 0.0 for column in TARGET_COLUMNS}
    sample_count = 0
    active_indices: dict[str, int] | None = None

    with logfile.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            fields = stripped.split()
            if fields[0] == "Step":
                if all(column in fields for column in TARGET_COLUMNS):
                    active_indices = {column: fields.index(column) for column in TARGET_COLUMNS}
                else:
                    active_indices = None
                continue

            if active_indices is None:
                continue

            try:
                values = {column: float(fields[idx]) for column, idx in active_indices.items()}
            except (IndexError, ValueError):
                continue

            for column, value in values.items():
                sums[column] += value
            sample_count += 1

    if sample_count == 0:
        raise ValueError(
            f"No thermo rows with columns {', '.join(TARGET_COLUMNS)} were found in {logfile}"
        )

    averages = {column: sums[column] / sample_count for column in TARGET_COLUMNS}
    return averages, sample_count


def main() -> None:
    args = build_arg_parser().parse_args()
    averages, sample_count = parse_bec_dipole_averages(args.logfile)

    print(f"Input file   : {args.logfile}")
    print(f"Sample count : {sample_count}")
    for column in TARGET_COLUMNS:
        print(f"{column:16s} average = {averages[column]: .10f}")


if __name__ == "__main__":
    main()
