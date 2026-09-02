#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Animate a dump: render every selected frame and assemble a GIF.

    animate.py dump.lammpstrj spin.gif \\
        --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]' --color vz --drop-zero-vector

    animate.py dump.lammpstrj bec.gif \\
        --color 'c_outbec[1]+c_outbec[5]+c_outbec[9]' --single-layer \\
        --subtract-mean --cmap RdBu_r

This replaces the separate spin and BEC animation scripts: the difference
between them was only which columns to read and whether to draw arrows, both of
which are now arguments. Anything else a dump carries -- an applied field, a
velocity, a per-atom energy -- animates the same way.

Rendering is imported from frame.py rather than launched as a subprocess. A
process per frame would spend one to two seconds importing matplotlib before
drawing anything, which for a few thousand frames costs more than the entire
calculation.

Under mpirun, rank 0 reads the frames and every rank renders a share of them.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from frame import (  # noqa: E402
    DERIVED,
    add_common_arguments,
    check_vector_requirement,
    config_from_args,
    render_png_bytes,
    render_rgba,
    resolve_range,
)
from dumpframe import load_frames  # noqa: E402
from mpi import (  # noqa: E402
    _format_duration,
    _progress_report_interval,
    _q_indices_for_rank,
    resolve_mpi_comm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animate.py",
        description="Render a dump trajectory as an animated GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("output_gif", type=Path, nargs="?", default=Path("texture.gif"),
                        help="Output GIF path")
    parser.add_argument("--frame-start", type=int, default=0, help="First frame, inclusive")
    parser.add_argument("--frame-stop", type=int, default=None, help="Stop before this frame")
    parser.add_argument("--frame-step", type=int, default=1, help="Keep one frame every N")
    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Print loading and rendering progress")
    parser.add_argument("--progress-reports", type=int, default=20,
                        help="Approximate number of progress updates per stage")
    return parser


def render_all(frames, cfg, mpi_comm, rank, size, progress, reports):
    """Render every frame, splitting the work across ranks. Rank 0 gets the images."""
    if mpi_comm is None:
        images = []
        total = len(frames)
        interval = _progress_report_interval(total, reports)
        started = time.perf_counter()
        for i, frame in enumerate(frames):
            images.append(Image.fromarray(render_rgba(frame, cfg)))
            done = i + 1
            if progress and (done % interval == 0 or done == total):
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(f"[INFO] Rendered {done}/{total} frames "
                      f"({_format_duration((total - done) / max(rate, 1e-9))} left)")
        return images

    total = int(mpi_comm.bcast(len(frames) if rank == 0 else None, root=0))
    frames = mpi_comm.bcast(frames if rank == 0 else None, root=0)

    mine = _q_indices_for_rank(total, rank, size)
    started = time.perf_counter()
    interval = _progress_report_interval(max(mine.size, 1), reports)

    local = []
    for step, index in enumerate(mine, start=1):
        local.append((int(index), render_png_bytes(frames[int(index)], cfg)))
        if progress and rank == 0 and (step % interval == 0 or step == mine.size):
            print(f"[INFO] Rank 0 rendered {step}/{mine.size} of its share")

    gathered = mpi_comm.gather(local, root=0)
    if rank != 0:
        return None

    ordered: list[Image.Image | None] = [None] * total
    for chunk in gathered:
        for index, png in chunk:
            with Image.open(io.BytesIO(png)) as image:
                ordered[index] = image.copy()
    return [image for image in ordered if image is not None]


def save_gif(images, output_gif: Path, fps: float) -> None:
    if not images:
        raise ValueError("No frames selected for animation.")
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_gif, save_all=True, append_images=images[1:],
        duration=max(1, int(round(1000.0 / fps))), loop=0, disposal=2,
    )


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    mpi_comm, rank, size = resolve_mpi_comm()
    is_root = rank == 0

    if args.vector is None and args.color is None:
        raise ValueError("Nothing to plot: give --vector, --color, or both.")
    check_vector_requirement(args)

    frames = None
    error = None
    if is_root:
        try:
            if mpi_comm is not None:
                print(f"[INFO] MPI render enabled: ranks={size}")
            frames, scanned = load_frames(
                args.input_dump,
                vector=tuple(args.vector) if args.vector else None,
                color=None if args.color in DERIVED else args.color,
                element=args.element,
                drop_zero_vector=args.drop_zero_vector,
                frame_start=args.frame_start,
                frame_stop=args.frame_stop,
                frame_step=args.frame_step,
            )
            if not frames:
                error = "No frames selected; adjust --frame-start/--frame-stop/--frame-step."
            else:
                print(f"[INFO] Frames scanned  : {scanned}")
                print(f"[INFO] Frames selected : {len(frames)}")
                print(f"[INFO] Atoms per frame : {frames[0]['x'].size}")
        except Exception as exc:  # report once, on rank 0
            error = str(exc)

    if mpi_comm is not None:
        error = mpi_comm.bcast(error if is_root else None, root=0)
    if error is not None:
        raise ValueError(error)

    cfg = config_from_args(args)
    span = None
    if is_root:
        span = resolve_range(frames, cfg)
    if mpi_comm is not None:
        span = mpi_comm.bcast(span if is_root else None, root=0)
    cfg["vmin"], cfg["vmax"] = span

    if is_root:
        print(f"[INFO] Vector          : {args.vector or 'none'}")
        print(f"[INFO] Colour          : {args.color or 'vector magnitude'}")
        print(f"[INFO] Colour range    : [{span[0]:g}, {span[1]:g}]")
        print(f"[INFO] Colormap        : {args.cmap}")
        print(f"[INFO] Layer mode      : {'single' if args.single_layer else 'bilayer'}")
        print(f"[INFO] Mean subtracted : {'yes (per frame)' if args.subtract_mean else 'no'}")
        print("[INFO] Rendering frames...")

    images = render_all(frames, cfg, mpi_comm, rank, size,
                        args.progress, args.progress_reports)
    if not is_root:
        return

    save_gif(images, args.output_gif, args.fps)
    print(f"[INFO] Output GIF      : {args.output_gif}")


if __name__ == "__main__":
    main()
