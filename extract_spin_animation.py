#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from extract_frame_to_xsf import _read_one_frame, plot_layer, select_color_component, split_layers
from sqw_mpi import resolve_mpi_comm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected frames from a LAMMPS dump and render a spin-texture GIF. "
            "Under mpirun, rank 0 reads the selected frames and all ranks render different "
            "frames in parallel. The arrows always show the in-plane (Sx, Sy) components, "
            "while the color field can be chosen from Sx/Sy/Sz."
        )
    )
    parser.add_argument("input_dump", type=Path, help="Input LAMMPS dump file")
    parser.add_argument(
        "output_gif",
        type=Path,
        nargs="?",
        default=Path("spin_texture.gif"),
        help="Output GIF path (default: spin_texture.gif)",
    )
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
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Animation frame rate in frames per second (default: 4.0)",
    )
    parser.add_argument(
        "--color-component",
        choices=["x", "y", "z"],
        default="x",
        help="Spin component mapped to the color field in the plot (default: x).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print loading/rendering progress information (default: enabled)",
    )
    parser.add_argument(
        "--progress-reports",
        type=int,
        default=20,
        help="Approximate number of progress updates to print per stage (default: 20)",
    )
    parser.add_argument(
        "--single-layer",
        action="store_true",
        help="Render a single layer (skip bilayer splitting by z-midpoint).",
    )
    return parser


def validate_args(
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
    fps: float,
    progress_reports: int,
) -> None:
    if frame_start < 0:
        raise ValueError("frame_start must be >= 0")
    if frame_stop is not None and frame_stop < 0:
        raise ValueError("frame_stop must be >= 0")
    if frame_stop is not None and frame_stop <= frame_start:
        raise ValueError("frame_stop must be greater than frame_start")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if fps <= 0.0:
        raise ValueError("fps must be > 0")
    if progress_reports < 1:
        raise ValueError("progress_reports must be >= 1")


def _progress_report_interval(total_steps: int, target_reports: int) -> int:
    if total_steps <= 0:
        return 1
    return max(1, math.ceil(total_steps / max(int(target_reports), 1)))


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"

    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _report_progress(
    label: str,
    done: int,
    total: int | None,
    start_time: float,
    selected_count: int | None = None,
) -> None:
    elapsed = time.perf_counter() - start_time
    if total is None:
        suffix = ""
        if selected_count is not None:
            suffix = f", selected {selected_count}"
        print(
            f"[INFO] {label}: processed {done}{suffix}, elapsed {_format_duration(elapsed)}",
            flush=True,
        )
        return

    frac = float(done) / float(max(total, 1))
    rate = float(done) / elapsed if elapsed > 0.0 else 0.0
    eta = (total - done) / rate if rate > 0.0 else float("inf")
    suffix = ""
    if selected_count is not None:
        suffix = f", selected {selected_count}"
    print(
        f"[INFO] {label}: {done}/{total} ({100.0 * frac:.1f}%){suffix}, "
        f"elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)}",
        flush=True,
    )


def extract_spin_columns(
    columns: list[str],
    atom_lines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    col_index = {name: idx for idx, name in enumerate(columns)}
    required = ["x", "y", "z", "c_outsp[1]", "c_outsp[2]", "c_outsp[3]"]
    missing = [name for name in required if name not in col_index]
    if missing:
        raise ValueError(f"Missing required dump columns: {', '.join(missing)}")

    rows: list[tuple[float, float, float, float, float, float]] = []
    for line in atom_lines:
        parts = line.split()
        sx = float(parts[col_index["c_outsp[1]"]])
        sy = float(parts[col_index["c_outsp[2]"]])
        sz = float(parts[col_index["c_outsp[3]"]])
        if sx == 0.0 and sy == 0.0 and sz == 0.0:
            continue
        rows.append(
            (
                float(parts[col_index["x"]]),
                float(parts[col_index["y"]]),
                float(parts[col_index["z"]]),
                sx,
                sy,
                sz,
            )
        )

    if not rows:
        raise ValueError("No magnetic atoms remain in the selected frame.")

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]


def load_selected_frames(
    dump_path: Path,
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
    progress: bool,
    progress_reports: int,
) -> tuple[list[dict[str, object]], int]:
    selected_frames: list[dict[str, object]] = []
    total_frames_read = 0
    total_target = None if frame_stop is None else int(frame_stop)
    report_interval = 100
    if total_target is not None:
        report_interval = _progress_report_interval(total_target, progress_reports)
    progress_start = time.perf_counter()

    with dump_path.open() as fh:
        while True:
            if frame_stop is not None and total_frames_read >= frame_stop:
                break

            frame = _read_one_frame(fh)
            if frame is None:
                break

            timestep, _box_lines, columns, atom_lines = frame
            keep_frame = total_frames_read >= frame_start
            if keep_frame:
                keep_frame = ((total_frames_read - frame_start) % frame_step) == 0

            if keep_frame:
                x, y, z, sx, sy, sz = extract_spin_columns(columns, atom_lines)
                selected_frames.append(
                    {
                        "frame_index": total_frames_read,
                        "timestep": timestep,
                        "x": x,
                        "y": y,
                        "z": z,
                        "sx": sx,
                        "sy": sy,
                        "sz": sz,
                    }
                )

            total_frames_read += 1
            if progress and (
                total_frames_read % report_interval == 0
                or (frame_stop is not None and total_frames_read == frame_stop)
            ):
                _report_progress(
                    label="Load progress",
                    done=total_frames_read,
                    total=total_target,
                    start_time=progress_start,
                    selected_count=len(selected_frames),
                )

    return selected_frames, total_frames_read


def render_frame_rgba(frame_data: dict[str, object], color_component: str, single_layer: bool = False) -> np.ndarray:
    x = np.asarray(frame_data["x"], dtype=float)
    y = np.asarray(frame_data["y"], dtype=float)
    z = np.asarray(frame_data["z"], dtype=float)
    sx = np.asarray(frame_data["sx"], dtype=float)
    sy = np.asarray(frame_data["sy"], dtype=float)
    sz = np.asarray(frame_data["sz"], dtype=float)

    color_values, color_label = select_color_component(color_component, sx, sy, sz)

    if single_layer:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
        mappable = plot_layer(ax, x, y, sx, sy, color_values, "Spin Texture")
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.03)
    else:
        up, dn = split_layers(z)
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        mappable = plot_layer(axs[0], x[up], y[up], sx[up], sy[up], color_values[up], "Top Layer")
        mappable = plot_layer(axs[1], x[dn], y[dn], sx[dn], sy[dn], color_values[dn], "Bottom Layer")
        cbar = fig.colorbar(mappable, ax=axs, shrink=0.85, pad=0.03)

    cbar.set_label(color_label, fontsize=18)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle(
        f"Frame {int(frame_data['frame_index'])}, timestep {int(frame_data['timestep'])}",
        fontsize=16,
    )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    return rgba.copy()


def render_frame_png_bytes(frame_data: dict[str, object], color_component: str, single_layer: bool = False) -> bytes:
    image = Image.fromarray(render_frame_rgba(frame_data, color_component, single_layer))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _partition_frames_for_ranks(
    frames: list[dict[str, object]],
    mpi_size: int,
) -> list[list[tuple[int, dict[str, object]]]]:
    assignments: list[list[tuple[int, dict[str, object]]]] = [[] for _ in range(mpi_size)]
    for frame_idx, frame_data in enumerate(frames):
        assignments[frame_idx % mpi_size].append((frame_idx, frame_data))
    return assignments


def _report_render_progress_mpi(
    step: int,
    local_total: int,
    global_total: int,
    start_time: float,
    mpi_comm: object | None,
    mpi_rank: int,
) -> None:
    local_done = min(step, local_total)
    if mpi_comm is None:
        global_done = local_done
    else:
        global_done = int(mpi_comm.allreduce(int(local_done)))

    if mpi_rank != 0:
        return

    elapsed = time.perf_counter() - start_time
    frac = float(global_done) / float(max(global_total, 1))
    rate = float(global_done) / elapsed if elapsed > 0.0 else 0.0
    eta = (global_total - global_done) / rate if rate > 0.0 else float("inf")
    print(
        f"[INFO] Render progress: {global_done}/{global_total} frames "
        f"({100.0 * frac:.1f}%), elapsed {_format_duration(elapsed)}, "
        f"ETA {_format_duration(eta)}",
        flush=True,
    )


def render_images_mpi(
    frames: list[dict[str, object]] | None,
    color_component: str,
    progress: bool,
    progress_reports: int,
    mpi_comm: object | None,
    mpi_rank: int,
    mpi_size: int,
    single_layer: bool = False,
) -> list[Image.Image] | None:
    if mpi_comm is None:
        if frames is None:
            return []
        total_frames = len(frames)
        report_interval = _progress_report_interval(total_frames, progress_reports)
        render_start = time.perf_counter()
        images: list[Image.Image] = []
        for idx, frame_data in enumerate(frames):
            images.append(Image.fromarray(render_frame_rgba(frame_data, color_component, single_layer)))
            done = idx + 1
            if progress and (done % report_interval == 0 or done == total_frames):
                _report_render_progress_mpi(
                    step=done,
                    local_total=total_frames,
                    global_total=total_frames,
                    start_time=render_start,
                    mpi_comm=None,
                    mpi_rank=0,
                )
        return images

    total_frames = 0 if frames is None else len(frames)
    total_frames = int(mpi_comm.bcast(total_frames if mpi_rank == 0 else None, root=0))

    assignments = _partition_frames_for_ranks(frames, mpi_size) if mpi_rank == 0 else None
    local_tasks = mpi_comm.scatter(assignments, root=0)
    local_total = len(local_tasks)
    max_local_steps = (total_frames + mpi_size - 1) // mpi_size if total_frames > 0 else 0
    report_interval = _progress_report_interval(max_local_steps, progress_reports)
    render_start = time.perf_counter()

    local_results: list[tuple[int, bytes]] = []
    for iloc in range(max_local_steps):
        if iloc < local_total:
            frame_idx, frame_data = local_tasks[iloc]
            local_results.append((frame_idx, render_frame_png_bytes(frame_data, color_component, single_layer)))

        step = iloc + 1
        if progress and (step % report_interval == 0 or step == max_local_steps):
            _report_render_progress_mpi(
                step=step,
                local_total=local_total,
                global_total=total_frames,
                start_time=render_start,
                mpi_comm=mpi_comm,
                mpi_rank=mpi_rank,
            )

    gathered = mpi_comm.gather(local_results, root=0)
    if mpi_rank != 0:
        return None

    ordered_images: list[Image.Image | None] = [None] * total_frames
    for rank_results in gathered:
        for frame_idx, png_bytes in rank_results:
            with Image.open(io.BytesIO(png_bytes)) as image:
                ordered_images[frame_idx] = image.copy()
    return [image for image in ordered_images if image is not None]


def save_gif(images: list[Image.Image], output_gif: Path, fps: float) -> None:
    if not images:
        raise ValueError("No frames selected for animation.")

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / fps)))
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    mpi_comm, mpi_rank, mpi_size = resolve_mpi_comm()
    is_root = mpi_rank == 0
    validate_args(
        args.frame_start,
        args.frame_stop,
        args.frame_step,
        args.fps,
        args.progress_reports,
    )

    frames: list[dict[str, object]] | None = None
    scanned_frames = None
    selection_error = None
    if is_root:
        try:
            if mpi_comm is not None:
                print(f"[INFO] MPI render enabled: ranks={mpi_size}")
            frames, scanned_frames = load_selected_frames(
                dump_path=args.input_dump,
                frame_start=args.frame_start,
                frame_stop=args.frame_stop,
                frame_step=args.frame_step,
                progress=args.progress,
                progress_reports=args.progress_reports,
            )
            if not frames:
                selection_error = "No frames selected; adjust --frame-start/--frame-stop/--frame-step."
        except Exception as exc:
            selection_error = str(exc)

    if mpi_comm is not None:
        selection_error = mpi_comm.bcast(selection_error if is_root else None, root=0)
    if selection_error is not None:
        raise ValueError(selection_error)

    if is_root:
        print(f"[INFO] Frames scanned  : {scanned_frames}")
        print(f"[INFO] Frames selected : {len(frames)}")
        print(f"[INFO] Color component : {args.color_component}")
        print(f"[INFO] FPS             : {args.fps:g}")
        print("[INFO] Rendering GIF frames...")

    if is_root:
        print(f"[INFO] Layer mode     : {'single' if args.single_layer else 'bilayer'}")

    images = render_images_mpi(
        frames=frames,
        color_component=args.color_component,
        progress=args.progress,
        progress_reports=args.progress_reports,
        mpi_comm=mpi_comm,
        mpi_rank=mpi_rank,
        mpi_size=mpi_size,
        single_layer=args.single_layer,
    )
    if not is_root:
        return

    save_gif(images, args.output_gif, args.fps)
    print(f"[INFO] Output GIF      : {args.output_gif}")


# python3 extract_spin_animation.py dump.lammpstrj spin_texture.gif
# python3 extract_spin_animation.py dump.lammpstrj spin_texture.gif --frame-start 0 --frame-stop 20 --frame-step 2
# python3 extract_spin_animation.py dump.lammpstrj spin_texture.gif --color-component z --fps 6
# mpirun -np 4 python3 extract_spin_animation.py dump.lammpstrj spin_texture.gif --frame-start 0 --frame-stop 20
if __name__ == "__main__":
    main()
