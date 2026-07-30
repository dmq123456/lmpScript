#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqw_mpi import resolve_mpi_comm
from xsf_plot import _read_one_frame, split_layers

# Reuse the generic MPI/progress/GIF infrastructure from the spin animation script.
from animation import (
    _format_duration,
    _partition_frames_for_ranks,
    _progress_report_interval,
    _report_progress,
    _report_render_progress_mpi,
    save_gif,
)

# Row-major 3x3 Born-effective-charge tensor: c_outbec[1..9].
BEC_COMPONENT_LABELS = {
    1: "xx",
    2: "xy",
    3: "xz",
    4: "yx",
    5: "yy",
    6: "yz",
    7: "zx",
    8: "zy",
    9: "zz",
}


def parse_bec_component(token: str) -> list[int]:
    """Turn a --bec-component token into the list of indices to sum.

    '1' -> [1];  '1+5+9' -> [1, 5, 9].  The grammar matches --component in the
    S(q,w) and DOS routes: indices joined by '+' are added into one field.
    """
    parts = [part.strip() for part in str(token).split("+")]
    if any(not part for part in parts):
        raise ValueError(
            f"Malformed --bec-component {token!r}: '+' must join two indices, as in 1+5+9"
        )
    indices = []
    for part in parts:
        if not part.isdigit() or not 1 <= int(part) <= 9:
            raise ValueError(
                f"Invalid BEC component {part!r}; use 1..9 (row-major xx..zz), "
                f"optionally joined by '+'"
            )
        indices.append(int(part))
    return indices


def bec_component_symbol(components: list[int]) -> str:
    """LaTeX symbol for a possibly-summed component."""
    if sorted(components) == [1, 5, 9]:
        return r"\mathrm{Tr}\,Z^*"
    return " + ".join(rf"Z^*_{{{BEC_COMPONENT_LABELS[c]}}}" for c in components)


def bec_component_text(components: list[int]) -> str:
    """Plain-text description, for log lines."""
    names = "+".join(BEC_COMPONENT_LABELS[c] for c in components)
    cols = "+".join(f"c_outbec[{c}]" for c in components)
    return f"{cols} ({names})"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected frames from a LAMMPS dump and render a Born-effective-charge "
            "(BEC) GIF. Unlike the spin animation, no arrows are drawn: a chosen BEC tensor "
            "component (c_outbec[1..9], a row-major 3x3 tensor) is shown as the background "
            "color field. Under mpirun, rank 0 reads the selected frames and all ranks render "
            "different frames in parallel."
        )
    )
    parser.add_argument("input_dump", type=Path, help="Input LAMMPS dump file")
    parser.add_argument(
        "output_gif",
        type=Path,
        nargs="?",
        default=Path("bec_texture.gif"),
        help="Output GIF path (default: bec_texture.gif)",
    )
    parser.add_argument(
        "--bec-component",
        type=str,
        default="1",
        metavar="C",
        help=(
            "BEC tensor component (c_outbec[N]) mapped to the color field. Row-major 3x3: "
            "1=xx 2=xy 3=xz 4=yx 5=yy 6=yz 7=zx 8=zy 9=zz. Indices joined by '+' are "
            "summed, so 1+5+9 is the trace (default: 1)."
        ),
    )
    parser.add_argument(
        "--element",
        type=str,
        default="all",
        help="Element symbol to include (e.g. Ni, I), or 'all' for every atom (default: all).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Lower bound of the color scale (default: auto from all selected frames).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper bound of the color scale (default: auto from all selected frames).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for the BEC color field (default: viridis).",
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
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help=(
            "Colour the deviation from the frame average instead of the raw value. "
            "The mean is taken over every selected atom in that frame and printed in "
            "the title, so the removed offset stays visible. In bilayer mode a single "
            "whole-frame mean is used for both panels, which keeps the difference "
            "between the layers intact."
        ),
    )
    return parser


def validate_args(
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
    fps: float,
    progress_reports: int,
    vmin: float | None,
    vmax: float | None,
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
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be less than vmax")


def extract_bec_columns(
    columns: list[str],
    atom_lines: list[str],
    bec_component: list[int],
    element: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    col_index = {name: idx for idx, name in enumerate(columns)}
    bec_cols = [f"c_outbec[{c}]" for c in bec_component]
    required = ["x", "y", "z", *bec_cols]
    select_by_element = element.lower() != "all"
    if select_by_element:
        required.append("element")
    missing = [name for name in required if name not in col_index]
    if missing:
        raise ValueError(f"Missing required dump columns: {', '.join(missing)}")

    rows: list[tuple[float, float, float, float]] = []
    for line in atom_lines:
        parts = line.split()
        if select_by_element and parts[col_index["element"]] != element:
            continue
        rows.append(
            (
                float(parts[col_index["x"]]),
                float(parts[col_index["y"]]),
                float(parts[col_index["z"]]),
                sum(float(parts[col_index[col]]) for col in bec_cols),
            )
        )

    if not rows:
        raise ValueError(
            f"No atoms remain in the selected frame for element {element!r}."
        )

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def load_selected_frames(
    dump_path: Path,
    bec_component: list[int],
    element: str,
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
                x, y, z, bec = extract_bec_columns(
                    columns, atom_lines, bec_component, element
                )
                selected_frames.append(
                    {
                        "frame_index": total_frames_read,
                        "timestep": timestep,
                        "x": x,
                        "y": y,
                        "z": z,
                        "bec": bec,
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


def frame_values(frame_data: dict[str, object], subtract_mean: bool) -> tuple[np.ndarray, float]:
    """The values to colour by, and the mean that was removed (0 if none was)."""
    bec = np.asarray(frame_data["bec"], dtype=float)
    if not subtract_mean:
        return bec, 0.0
    mean = float(bec.mean())
    return bec - mean, mean


def resolve_color_range(
    frames: list[dict[str, object]],
    vmin: float | None,
    vmax: float | None,
    subtract_mean: bool = False,
) -> tuple[float, float]:
    if vmin is not None and vmax is not None:
        return vmin, vmax

    # The range has to be measured on whatever is actually plotted: with
    # --subtract-mean the values are centred per frame, and a range taken from
    # the raw data would leave every frame washed out.
    all_values = np.concatenate(
        [frame_values(f, subtract_mean)[0] for f in frames]
    )

    if subtract_mean and vmin is None and vmax is None:
        # Centred data deserves a centred scale: with limits +/-max|v| the
        # midpoint of the colormap lands exactly on zero, so a diverging map
        # puts its neutral colour on "no deviation" and the sign of a feature
        # can be read off directly. Letting the limits follow the raw min and
        # max would shift zero off centre and make positive and negative
        # excursions of equal size look different.
        extent = float(np.max(np.abs(all_values)))
        if extent == 0.0:
            extent = 0.5
        return -extent, extent

    data_min = float(np.min(all_values))
    data_max = float(np.max(all_values))
    if data_min == data_max:
        data_min -= 0.5
        data_max += 0.5
    resolved_min = vmin if vmin is not None else data_min
    resolved_max = vmax if vmax is not None else data_max
    return resolved_min, resolved_max


def plot_bec_layer(ax, x, y, color_values, title, cmap, vmin, vmax):
    tri = mtri.Triangulation(x, y)
    tcf = ax.tripcolor(tri, color_values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=18, weight="bold")
    ax.set_xlabel(r"L$_x$ ($\mathrm{\AA}$)", fontsize=16)
    ax.set_ylabel(r"L$_y$ ($\mathrm{\AA}$)", fontsize=16)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    return tcf


def render_frame_rgba(
    frame_data: dict[str, object],
    bec_component: list[int],
    cmap: str,
    vmin: float,
    vmax: float,
    single_layer: bool = False,
    subtract_mean: bool = False,
) -> np.ndarray:
    x = np.asarray(frame_data["x"], dtype=float)
    y = np.asarray(frame_data["y"], dtype=float)
    z = np.asarray(frame_data["z"], dtype=float)
    bec, mean = frame_values(frame_data, subtract_mean)

    symbol = bec_component_symbol(bec_component)
    if subtract_mean:
        color_label = rf"${symbol} - \langle {symbol} \rangle$"
    else:
        color_label = rf"${symbol}$"

    if single_layer:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
        mappable = plot_bec_layer(ax, x, y, bec, "BEC Field", cmap, vmin, vmax)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.03)
    else:
        up, dn = split_layers(z)
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        mappable = plot_bec_layer(axs[0], x[up], y[up], bec[up], "Top Layer", cmap, vmin, vmax)
        mappable = plot_bec_layer(axs[1], x[dn], y[dn], bec[dn], "Bottom Layer", cmap, vmin, vmax)
        cbar = fig.colorbar(mappable, ax=axs, shrink=0.85, pad=0.03)

    cbar.set_label(color_label, fontsize=18)
    cbar.ax.tick_params(labelsize=13)

    title = f"Frame {int(frame_data['frame_index'])}, timestep {int(frame_data['timestep'])}"
    if subtract_mean:
        title += rf"    $\langle {symbol} \rangle = {mean:.6g}$"
    fig.suptitle(title, fontsize=16)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    return rgba.copy()


def render_frame_png_bytes(
    frame_data: dict[str, object],
    bec_component: list[int],
    cmap: str,
    vmin: float,
    vmax: float,
    single_layer: bool = False,
    subtract_mean: bool = False,
) -> bytes:
    image = Image.fromarray(
        render_frame_rgba(
            frame_data, bec_component, cmap, vmin, vmax, single_layer, subtract_mean
        )
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_images_mpi(
    frames: list[dict[str, object]] | None,
    bec_component: list[int],
    cmap: str,
    vmin: float,
    vmax: float,
    progress: bool,
    progress_reports: int,
    mpi_comm: object | None,
    mpi_rank: int,
    mpi_size: int,
    single_layer: bool = False,
    subtract_mean: bool = False,
) -> list[Image.Image] | None:
    if mpi_comm is None:
        if frames is None:
            return []
        total_frames = len(frames)
        report_interval = _progress_report_interval(total_frames, progress_reports)
        render_start = time.perf_counter()
        images: list[Image.Image] = []
        for idx, frame_data in enumerate(frames):
            images.append(
                Image.fromarray(
                    render_frame_rgba(
                        frame_data, bec_component, cmap, vmin, vmax,
                        single_layer, subtract_mean,
                    )
                )
            )
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
            local_results.append(
                (
                    frame_idx,
                    render_frame_png_bytes(
                        frame_data, bec_component, cmap, vmin, vmax,
                        single_layer, subtract_mean,
                    ),
                )
            )

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


def main() -> None:
    args = build_arg_parser().parse_args()
    bec_component = parse_bec_component(args.bec_component)
    mpi_comm, mpi_rank, mpi_size = resolve_mpi_comm()
    is_root = mpi_rank == 0
    validate_args(
        args.frame_start,
        args.frame_stop,
        args.frame_step,
        args.fps,
        args.progress_reports,
        args.vmin,
        args.vmax,
    )

    frames: list[dict[str, object]] | None = None
    scanned_frames = None
    selection_error = None
    color_range: tuple[float, float] | None = None
    if is_root:
        try:
            if mpi_comm is not None:
                print(f"[INFO] MPI render enabled: ranks={mpi_size}")
            frames, scanned_frames = load_selected_frames(
                dump_path=args.input_dump,
                bec_component=bec_component,
                element=args.element,
                frame_start=args.frame_start,
                frame_stop=args.frame_stop,
                frame_step=args.frame_step,
                progress=args.progress,
                progress_reports=args.progress_reports,
            )
            if not frames:
                selection_error = "No frames selected; adjust --frame-start/--frame-stop/--frame-step."
            else:
                color_range = resolve_color_range(
                    frames, args.vmin, args.vmax, args.subtract_mean
                )
        except Exception as exc:
            selection_error = str(exc)

    if mpi_comm is not None:
        selection_error = mpi_comm.bcast(selection_error if is_root else None, root=0)
    if selection_error is not None:
        raise ValueError(selection_error)

    if mpi_comm is not None:
        color_range = mpi_comm.bcast(color_range if is_root else None, root=0)
    vmin, vmax = color_range

    if is_root:
        print(f"[INFO] Frames scanned  : {scanned_frames}")
        print(f"[INFO] Frames selected : {len(frames)}")
        print(f"[INFO] BEC component   : {bec_component_text(bec_component)}")
        print(f"[INFO] Element         : {args.element}")
        print(f"[INFO] Color range     : [{vmin:g}, {vmax:g}]")
        print(f"[INFO] Colormap        : {args.cmap}")
        print(f"[INFO] FPS             : {args.fps:g}")
        print(f"[INFO] Layer mode      : {'single' if args.single_layer else 'bilayer'}")
        print(f"[INFO] Mean subtracted : {'yes (per frame)' if args.subtract_mean else 'no'}")
        print("[INFO] Rendering GIF frames...")

    images = render_images_mpi(
        frames=frames,
        bec_component=bec_component,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        progress=args.progress,
        progress_reports=args.progress_reports,
        mpi_comm=mpi_comm,
        mpi_rank=mpi_rank,
        mpi_size=mpi_size,
        single_layer=args.single_layer,
        subtract_mean=args.subtract_mean,
    )
    if not is_root:
        return

    save_gif(images, args.output_gif, args.fps)
    print(f"[INFO] Output GIF      : {args.output_gif}")


if __name__ == "__main__":
    main()
