#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a subset of frames from a LAMMPS dump file using "
            "frame_start:frame_stop:frame_step semantics."
        )
    )
    parser.add_argument("input_file", help="Input LAMMPS dump file")
    parser.add_argument("output_file", help="Output dump file")
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="First frame index to keep, inclusive (default: 0)",
    )
    parser.add_argument(
        "--frame-stop",
        type=int,
        default=None,
        help="Stop before this frame index, exclusive (default: keep through last frame)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep one frame every N selected frames (default: 1)",
    )
    return parser


def validate_args(frame_start: int, frame_stop: int | None, frame_step: int) -> None:
    if frame_start < 0:
        raise ValueError("frame_start must be >= 0")
    if frame_stop is not None and frame_stop < 0:
        raise ValueError("frame_stop must be >= 0")
    if frame_stop is not None and frame_stop <= frame_start:
        raise ValueError("frame_stop must be greater than frame_start")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")


def _read_required_line(fin, err_msg: str) -> str:
    line = fin.readline()
    if not line:
        raise ValueError(err_msg)
    return line


def stream_one_frame(fin, fout, keep_frame: bool) -> bool:
    first_line = fin.readline()
    if not first_line:
        return False
    if first_line.strip() != "ITEM: TIMESTEP":
        raise ValueError(f"Expected 'ITEM: TIMESTEP', got {first_line.strip()!r}")
    if keep_frame:
        fout.write(first_line)

    timestep_line = _read_required_line(fin, "Unexpected EOF after ITEM: TIMESTEP")
    if keep_frame:
        fout.write(timestep_line)

    number_header = _read_required_line(fin, "Unexpected EOF before ITEM: NUMBER OF ATOMS")
    if number_header.strip() != "ITEM: NUMBER OF ATOMS":
        raise ValueError("Expected 'ITEM: NUMBER OF ATOMS'")
    if keep_frame:
        fout.write(number_header)

    natoms_line = _read_required_line(fin, "Unexpected EOF after ITEM: NUMBER OF ATOMS")
    natoms = int(natoms_line.strip())
    if keep_frame:
        fout.write(natoms_line)

    box_header = _read_required_line(fin, "Unexpected EOF before ITEM: BOX BOUNDS")
    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError("Expected 'ITEM: BOX BOUNDS'")
    if keep_frame:
        fout.write(box_header)

    for _ in range(3):
        box_line = _read_required_line(fin, "Unexpected EOF while reading BOX BOUNDS")
        if keep_frame:
            fout.write(box_line)

    atoms_header = _read_required_line(fin, "Unexpected EOF before ITEM: ATOMS")
    if not atoms_header.startswith("ITEM: ATOMS"):
        raise ValueError("Expected 'ITEM: ATOMS'")
    if keep_frame:
        fout.write(atoms_header)

    for _ in range(natoms):
        atom_line = _read_required_line(fin, "Unexpected EOF while reading atom lines")
        if keep_frame:
            fout.write(atom_line)

    return True


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args.frame_start, args.frame_stop, args.frame_step)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    kept_frames = 0
    total_frames = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        while True:
            keep_frame = total_frames >= args.frame_start
            if args.frame_stop is not None and total_frames >= args.frame_stop:
                break
            elif keep_frame:
                keep_frame = ((total_frames - args.frame_start) % args.frame_step) == 0

            has_frame = stream_one_frame(fin, fout, keep_frame=keep_frame)
            if not has_frame:
                break
            if keep_frame:
                kept_frames += 1

            total_frames += 1

    print(f"[INFO] Input frames : {total_frames}")
    print(f"[INFO] Kept frames  : {kept_frames}")
    print(f"[INFO] Output file  : {output_path}")


if __name__ == "__main__":
    main()
