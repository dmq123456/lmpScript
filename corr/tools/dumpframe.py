#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reading LAMMPS dump frames and pulling named columns out of them.

This is the only layer that knows the dump file format. Everything above it
works with the dictionary returned by `extract`, which carries plain arrays and
no notion of what the columns meant -- so the same renderer serves spins, Born
effective charges, applied fields or anything else a dump happens to hold.

Column specifications
---------------------
A column spec is one or more dump column names joined by '+', whose values are
summed:

    'c_outsp[3]'                      one column
    'c_outbec[1]+c_outbec[5]+c_outbec[9]'   the trace of a tensor

The grammar matches --component in the S(q,w) and DOS routes, so the same idea
of "sum these into one field" holds throughout the package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Dump parsing
# ----------------------------------------------------------------------
def _required_line(fh, message: str) -> str:
    line = fh.readline()
    if not line:
        raise ValueError(message)
    return line


def read_one_frame(fh):
    """Read the next frame from an open dump. Returns None at end of file.

    Yields (timestep, box_header, box_lines, column_names, atom_lines). The box
    is carried rather than discarded because a derived field can need the
    periodic images -- the topological charge density is defined on triangles
    that wrap around the cell.
    """
    first = fh.readline()
    if not first:
        return None
    if first.strip() != "ITEM: TIMESTEP":
        raise ValueError(
            f"Unexpected dump format: expected 'ITEM: TIMESTEP', got {first.strip()!r}"
        )

    timestep = int(_required_line(fh, "Unexpected EOF after ITEM: TIMESTEP").strip())

    if _required_line(fh, "Unexpected EOF before NUMBER OF ATOMS").strip() != "ITEM: NUMBER OF ATOMS":
        raise ValueError("Unexpected dump format: missing NUMBER OF ATOMS block")
    natoms = int(_required_line(fh, "Unexpected EOF after NUMBER OF ATOMS").strip())

    box_header = _required_line(fh, "Unexpected EOF before BOX BOUNDS").strip()
    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError("Unexpected dump format: missing BOX BOUNDS block")
    box_lines = [
        _required_line(fh, "Unexpected EOF while reading BOX BOUNDS").strip()
        for _ in range(3)
    ]

    atoms_header = _required_line(fh, "Unexpected EOF before ATOMS").strip()
    if not atoms_header.startswith("ITEM: ATOMS "):
        raise ValueError("Unexpected dump format: missing ATOMS block")
    atom_lines = [
        _required_line(fh, "Unexpected EOF while reading ATOMS lines").strip()
        for _ in range(natoms)
    ]
    return timestep, box_header, box_lines, atoms_header.split()[2:], atom_lines


def count_frames(dump_path: Path) -> int:
    total = 0
    with Path(dump_path).open() as fh:
        while read_one_frame(fh) is not None:
            total += 1
    return total


# ----------------------------------------------------------------------
# Column specs
# ----------------------------------------------------------------------
def parse_column_spec(spec: str) -> list[str]:
    """'a+b+c' -> ['a', 'b', 'c']; a single name -> [name]."""
    parts = [part.strip() for part in str(spec).split("+")]
    if any(not part for part in parts):
        raise ValueError(
            f"Malformed column spec {spec!r}: '+' must join two column names"
        )
    return parts


def spec_label(spec: str) -> str:
    """Readable label for a possibly-summed spec."""
    parts = parse_column_spec(spec)
    return parts[0] if len(parts) == 1 else " + ".join(parts)


def _column_values(parts: list[str], atom_parts: list[str],
                   index: dict[str, int]) -> float:
    return sum(float(atom_parts[index[name]]) for name in parts)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
def extract(
    columns: list[str],
    atom_lines: list[str],
    *,
    vector: tuple[str, str, str] | None = None,
    color: str | None = None,
    element: str = "all",
    drop_zero_vector: bool = False,
) -> dict[str, np.ndarray]:
    """Pull positions, an optional vector and an optional scalar from one frame.

    vector is a triple of column specs drawn as arrows; color is one spec mapped
    to colour. Either may be omitted, but not both -- a frame with neither has
    nothing to show.

    drop_zero_vector discards atoms whose vector is exactly zero, which is how
    the spin animation used to skip non-magnetic sites. It is off by default,
    since for a general field an exact zero is a legitimate value.
    """
    if vector is None and color is None:
        raise ValueError("Nothing to plot: give --vector, --color, or both.")

    index = {name: i for i, name in enumerate(columns)}
    vector_parts = [parse_column_spec(s) for s in vector] if vector else []
    color_parts = parse_column_spec(color) if color else []

    needed = ["x", "y", "z"]
    for parts in vector_parts:
        needed.extend(parts)
    needed.extend(color_parts)
    select_by_element = element.lower() != "all"
    if select_by_element:
        needed.append("element")

    missing = [name for name in dict.fromkeys(needed) if name not in index]
    if missing:
        raise ValueError(
            f"Missing required dump columns: {', '.join(missing)}. "
            f"Available: {', '.join(columns)}"
        )

    rows: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if select_by_element and parts[index["element"]] != element:
            continue
        row = [float(parts[index[axis]]) for axis in ("x", "y", "z")]
        if vector_parts:
            comps = [_column_values(p, parts, index) for p in vector_parts]
            if drop_zero_vector and not any(comps):
                continue
            row.extend(comps)
        if color_parts:
            row.append(_column_values(color_parts, parts, index))
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No atoms remain in this frame for element {element!r}"
            + (" after dropping zero vectors." if drop_zero_vector else ".")
        )

    data = np.asarray(rows, dtype=float)
    out = {"x": data[:, 0], "y": data[:, 1], "z": data[:, 2]}
    cursor = 3
    if vector_parts:
        out["u"], out["v"], out["w"] = data[:, cursor], data[:, cursor + 1], data[:, cursor + 2]
        cursor += 3
    if color_parts:
        out["c"] = data[:, cursor]
    return out


def load_frames(
    dump_path: Path,
    *,
    vector: tuple[str, str, str] | None = None,
    color: str | None = None,
    element: str = "all",
    drop_zero_vector: bool = False,
    frame_start: int = 0,
    frame_stop: int | None = None,
    frame_step: int = 1,
    progress_callback=None,
) -> tuple[list[dict], int]:
    """Read the selected frames into memory. Returns (frames, frames_scanned)."""
    frames: list[dict] = []
    scanned = 0
    with Path(dump_path).open() as fh:
        while True:
            if frame_stop is not None and scanned >= frame_stop:
                break
            record = read_one_frame(fh)
            if record is None:
                break
            timestep, box_header, box_lines, columns, atom_lines = record
            keep = scanned >= frame_start and (scanned - frame_start) % frame_step == 0
            if keep:
                frame = extract(
                    columns, atom_lines,
                    vector=vector, color=color, element=element,
                    drop_zero_vector=drop_zero_vector,
                )
                frame["frame_index"] = scanned
                frame["timestep"] = timestep
                frame["box_header"] = box_header
                frame["box_lines"] = box_lines
                frames.append(frame)
            scanned += 1
            if progress_callback is not None:
                progress_callback(scanned, len(frames))
    return frames, scanned


def load_single_frame(
    dump_path: Path,
    frame_index: int,
    **kwargs,
) -> dict:
    """Read exactly one frame by index, without retaining the others."""
    scanned = 0
    with Path(dump_path).open() as fh:
        while True:
            record = read_one_frame(fh)
            if record is None:
                raise ValueError(
                    f"Frame {frame_index} not found; the dump holds {scanned} frame(s)."
                )
            if scanned == frame_index:
                timestep, box_header, box_lines, columns, atom_lines = record
                frame = extract(columns, atom_lines, **kwargs)
                frame["frame_index"] = scanned
                frame["timestep"] = timestep
                frame["box_header"] = box_header
                frame["box_lines"] = box_lines
                return frame
            scanned += 1


__all__ = [
    "read_one_frame",
    "count_frames",
    "parse_column_spec",
    "spec_label",
    "extract",
    "load_frames",
    "load_single_frame",
]
