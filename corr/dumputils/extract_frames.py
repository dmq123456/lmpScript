#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path

HEADER = b"ITEM: TIMESTEP"
BLOCK = 1 << 20  # bytes read per step when scanning backwards


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a subset of frames from a LAMMPS dump file using "
            "frame_start:frame_stop:frame_step semantics. Negative start/stop "
            "count from the end as in Python (-1 = last frame); such frames are "
            "located by scanning the file backwards instead of reading it from the start."
        )
    )
    parser.add_argument("input_file", help="Input LAMMPS dump file")
    parser.add_argument("output_file", help="Output dump file")
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="First frame index to keep, inclusive; negative counts from the end (default: 0)",
    )
    parser.add_argument(
        "--frame-stop",
        type=int,
        default=None,
        help="Stop before this frame index, exclusive; negative counts from the end "
             "(default: keep through last frame)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep one frame every N selected frames (default: 1)",
    )
    return parser


def validate_args(frame_start: int, frame_stop: int | None, frame_step: int) -> None:
    # Only pairs of the same sign can be compared before the number of frames is known.
    if frame_stop is not None and (frame_start >= 0) == (frame_stop >= 0) and frame_stop <= frame_start:
        raise ValueError("frame_stop must be greater than frame_start")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")


def _read_required_line(fin, err_msg: str) -> bytes:
    line = fin.readline()
    if not line:
        raise ValueError(err_msg)
    return line


def stream_one_frame(fin, fout, keep_frame: bool) -> bool:
    first_line = fin.readline()
    if not first_line:
        return False
    if first_line.strip() != HEADER:
        raise ValueError(f"Expected 'ITEM: TIMESTEP', got {first_line.strip()!r}")
    if keep_frame:
        fout.write(first_line)

    timestep_line = _read_required_line(fin, "Unexpected EOF after ITEM: TIMESTEP")
    if keep_frame:
        fout.write(timestep_line)

    number_header = _read_required_line(fin, "Unexpected EOF before ITEM: NUMBER OF ATOMS")
    if number_header.strip() != b"ITEM: NUMBER OF ATOMS":
        raise ValueError("Expected 'ITEM: NUMBER OF ATOMS'")
    if keep_frame:
        fout.write(number_header)

    natoms_line = _read_required_line(fin, "Unexpected EOF after ITEM: NUMBER OF ATOMS")
    natoms = int(natoms_line.strip())
    if keep_frame:
        fout.write(natoms_line)

    box_header = _read_required_line(fin, "Unexpected EOF before ITEM: BOX BOUNDS")
    if not box_header.startswith(b"ITEM: BOX BOUNDS"):
        raise ValueError("Expected 'ITEM: BOX BOUNDS'")
    if keep_frame:
        fout.write(box_header)

    for _ in range(3):
        box_line = _read_required_line(fin, "Unexpected EOF while reading BOX BOUNDS")
        if keep_frame:
            fout.write(box_line)

    atoms_header = _read_required_line(fin, "Unexpected EOF before ITEM: ATOMS")
    if not atoms_header.startswith(b"ITEM: ATOMS"):
        raise ValueError("Expected 'ITEM: ATOMS'")
    if keep_frame:
        fout.write(atoms_header)

    for _ in range(natoms):
        atom_line = _read_required_line(fin, "Unexpected EOF while reading atom lines")
        if keep_frame:
            fout.write(atom_line)

    return True


def last_frame_offsets(path: Path, count: int) -> list[int]:
    """Byte offsets of the last `count` frame headers, in file order.

    The file is read backwards in blocks; a header counts only at the start of
    a line. Fewer offsets are returned if the file has fewer frames.
    """
    found: list[int] = []
    with path.open("rb") as f:
        pos = f.seek(0, os.SEEK_END)
        prev = b""                                   # block read in the previous step (later in the file)
        while pos > 0 and len(found) < count:
            n = min(BLOCK, pos)
            pos -= n
            f.seek(pos)
            block = f.read(n)
            buf = block + prev[: len(HEADER) - 1]    # catch a header split across two blocks
            end = len(buf)
            while len(found) < count:
                i = buf.rfind(HEADER, 0, end)
                if i < 0:
                    break
                if i > 0:
                    at_line_start = buf[i - 1:i] == b"\n"
                elif pos == 0:
                    at_line_start = True
                else:                                # previous byte lies in the next block back
                    f.seek(pos - 1)
                    at_line_start = f.read(1) == b"\n"
                if at_line_start:
                    found.append(pos + i)
                end = i
            prev = block
    return found[::-1]


def count_frames(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        first = f.read(len(HEADER))
        n += first == HEADER
        f.seek(0)
        prev = b""
        while True:
            block = f.read(BLOCK)
            if not block:
                return n
            n += (prev + block).count(b"\n" + HEADER)
            prev = block[-len(HEADER):]               # too short to hold a whole match: no double counting


def resolve(path: Path, start: int, stop: int | None):
    """Return (byte offset to seek to, frames to skip there, max frames to read, byte offset to stop at)."""
    if start >= 0 and (stop is None or stop >= 0):
        return 0, start, None if stop is None else stop - start, None
    if start < 0:
        offs = last_frame_offsets(path, -start)
        if not offs:
            return 0, 0, 0, None
        if stop is None:
            return offs[0], 0, None, None
        if stop < 0:                                 # both counted from the end
            return offs[0], 0, max(len(offs) + stop, 0), None
        first = count_frames(path) - len(offs)       # mixed signs: absolute index of the start frame
        return offs[0], 0, max(stop - first, 0), None
    # start >= 0, stop < 0: read from the start up to the header of frame `stop` (from the end)
    offs = last_frame_offsets(path, -stop)
    if len(offs) < -stop:                            # fewer frames than |stop|: nothing to keep
        return 0, start, 0, None
    return 0, start, None, offs[0]


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args.frame_start, args.frame_stop, args.frame_step)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    seek_to, skip, n_max, stop_offset = resolve(input_path, args.frame_start, args.frame_stop)

    kept_frames = 0
    read_frames = 0                                  # frames read from seek_to on, skipped ones included

    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        fin.seek(seek_to)
        while True:
            if stop_offset is not None and fin.tell() >= stop_offset:
                break
            rel = read_frames - skip                 # index relative to the first selected frame
            if n_max is not None and rel >= n_max:
                break
            keep_frame = rel >= 0 and rel % args.frame_step == 0
            if not stream_one_frame(fin, fout, keep_frame=keep_frame):
                break
            kept_frames += keep_frame
            read_frames += 1

    where = f"from byte {seek_to}" if seek_to else "from the start"
    print(f"[INFO] Frames read  : {read_frames} ({where})")
    print(f"[INFO] Kept frames  : {kept_frames}")
    print(f"[INFO] Output file  : {output_path}")


if __name__ == "__main__":
    main()
