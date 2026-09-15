#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw S(q) over the two-dimensional Brillouin zone, for one frame.

    bzmap.py dump.lammpstrj --frame 1500 --element Ni --single-layer \\
        --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]' --out sq.png

This is `frame.py` in reciprocal space: same reading path, same config
dictionary, same `render_rgba(frame, cfg)` signature, so `animate.py` can point
its rendering loop here instead and produce a diffraction movie of the same
trajectory. The physics lives in `structurefactor.py`; everything here is
drawing.

Four things make a zone map different from a real-space panel, and all four are
handled here rather than left to the caller.

Skew. The allowed q sit on the reciprocal lattice, which for a hexagonal cell
is a 60 degree parallelogram, not a rectangle. `imshow` would silently shear
the picture; `pcolormesh` with explicit vertex coordinates does not.

Origin. The FFT puts q = 0 in the corner. It is moved to the middle, because a
map of a spiral is read by where the satellites sit relative to Gamma.

Coverage. The allowed q fill one reciprocal cell -- a parallelogram -- while the
first zone is a hexagon, and parts of that hexagon stick out of the
parallelogram. The map is therefore tiled over the neighbouring cells before
being clipped to the zone. S(q + G) = S(q) holds exactly for one magnetic site
per primitive cell, so this adds nothing that was not already measured.

Dynamic range. A magnetic Bragg peak stands orders of magnitude above the
diffuse background; on a linear scale the picture is a single dot on black. The
default is logarithmic over a fixed number of decades below the maximum, which
is also what keeps an animation's frames comparable.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Polygon
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from dumpframe import load_frames, load_single_frame  # noqa: E402
from frame import add_common_arguments, config_from_args  # noqa: E402
from structurefactor import (  # noqa: E402
    DEFAULT_CHANNEL,
    average_intensity,
    brillouin_zone_polygon,
    centre,
    frame_intensity,
    tile,
)

_CHANNEL_LABEL = {
    "total": r"$S(\mathbf{q})$",
    "trace": r"$S(\mathbf{q})$",
    "L": r"$S_\parallel(\mathbf{q})$",
    "T": r"$S_\perp(\mathbf{q})$",
}


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------
def add_bz_arguments(parser: argparse.ArgumentParser) -> None:
    """The options that only mean something in reciprocal space."""
    parser.add_argument("--channel", type=str, default=DEFAULT_CHANNEL, metavar="C",
                        help="Projection of s^a(q) s^b*(q): 'total' (= 1+5+9), 'L', 'T', "
                             "or any constant token of the --component grammar such as "
                             "xx, zz, 1+5+9. 'T' is what unpolarised neutrons see; "
                             "L/total at the magnetic peak is sin^2(theta)/2 for the "
                             "angle theta between the spiral normal and q")
    parser.add_argument("--scale", choices=("log", "linear"), default="log",
                        help="Intensity scale. A Bragg peak buries the background on a "
                             "linear scale")
    parser.add_argument("--log-range", type=float, default=5.0, metavar="DECADES",
                        help="Decades shown below the maximum when --scale log")
    parser.add_argument("--zone", choices=("bz", "cell"), default="bz",
                        help="'bz' clips to the first Brillouin zone; 'cell' shows the "
                             "reciprocal unit cell the transform natively covers")
    parser.add_argument("--mark-peaks", type=int, default=0, metavar="N",
                        help="Circle the N strongest wavevectors apart from Gamma")
    parser.add_argument("--frame-average", type=int, default=1, metavar="W",
                        help="Average S(q) over W consecutive frames. One snapshot is a "
                             "single thermal realisation; averaging cuts its noise as "
                             "1/sqrt(W) without blurring anything slower than W frames")


def bz_config_from_args(args) -> dict:
    cfg = config_from_args(args)
    cfg.update({
        "channel": args.channel,
        "scale": args.scale,
        "log_range": args.log_range,
        "zone": args.zone,
        "mark_peaks": args.mark_peaks,
        "frame_average": getattr(args, "frame_average", 1),
    })
    return cfg


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def _norm(values: np.ndarray, cfg: dict):
    """Colour normalisation, and the floor that was applied to the data."""
    vmax = cfg.get("vmax")
    vmin = cfg.get("vmin")
    if vmax is None:
        vmax = float(np.max(values))
    if cfg.get("scale", "log") == "linear":
        return Normalize(vmin=0.0 if vmin is None else vmin, vmax=vmax), None
    # A log scale cannot show the exact zeros an ideal texture produces, so the
    # data is floored a fixed number of decades below the peak rather than
    # masked -- masking would punch holes in the map wherever S(q) vanishes.
    floor = vmax * 10.0 ** (-abs(cfg.get("log_range", 5.0)))
    if vmin is not None and vmin > 0.0:
        floor = vmin
    return LogNorm(vmin=floor, vmax=max(vmax, floor * 10.0)), floor


def draw_zone(ax, q_grid, values, reciprocal, cfg, title):
    """One panel: the tiled, centred map clipped to the requested region."""
    repeats = 1 if cfg.get("zone", "bz") == "bz" else 0
    q_tiled, v_tiled = tile(centre(q_grid), centre(values), repeats=repeats)

    norm, floor = _norm(v_tiled, cfg)
    if floor is not None:
        v_tiled = np.maximum(v_tiled, floor)

    mesh = ax.pcolormesh(q_tiled[:, :, 0], q_tiled[:, :, 1], v_tiled,
                         shading="gouraud", cmap=cfg["cmap"], norm=norm)

    polygon = brillouin_zone_polygon(reciprocal)
    if cfg.get("zone", "bz") == "bz":
        patch = Polygon(polygon, closed=True, transform=ax.transData)
        mesh.set_clip_path(patch)
        ax.add_patch(Polygon(polygon, closed=True, fill=False,
                             edgecolor="w", lw=1.2, alpha=0.8))
        limit = 1.08 * float(np.max(np.abs(polygon)))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
    else:
        ax.add_patch(Polygon(polygon, closed=True, fill=False,
                             edgecolor="w", lw=1.0, alpha=0.5, ls="--"))

    n_marks = int(cfg.get("mark_peaks", 0))
    if n_marks > 0:
        flat = centre(values).ravel().copy()
        q_flat = centre(q_grid).reshape(-1, 3)
        flat[np.argmin(np.linalg.norm(q_flat, axis=1))] = -np.inf     # skip Gamma
        for index in np.argsort(-flat)[:n_marks]:
            ax.plot(q_flat[index, 0], q_flat[index, 1], "o", mfc="none",
                    mec="w", mew=1.4, ms=11, alpha=0.9)

    ax.set_title(title, fontsize=18, weight="bold")
    ax.set_xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)", fontsize=16)
    ax.set_ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)", fontsize=16)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    return mesh


def render_rgba(frame: dict, cfg: dict) -> np.ndarray:
    """Render the zone map of one frame (or of a list of frames) to RGBA.

    A list is averaged first, which is how --frame-average and any future
    sliding window reach this function without a second entry point.
    """
    frames = frame if isinstance(frame, (list, tuple)) else [frame]
    maps = average_intensity(frames, cfg) if len(frames) > 1 else frame_intensity(frames[0], cfg)
    first = frames[0]

    width_in, height_in = cfg["figsize"]
    n_panels = len(maps)
    fig, axs = plt.subplots(1, n_panels, figsize=(n_panels * width_in, height_in),
                            dpi=cfg["dpi"], constrained_layout=True, squeeze=False)
    names = ["Top Layer", "Bottom Layer"] if n_panels == 2 else [cfg["title"] or "S(q)"]
    mesh = None
    for ax, (q_grid, values, reciprocal), name in zip(axs[0], maps, names):
        mesh = draw_zone(ax, q_grid, values, reciprocal, cfg, name)

    channel = str(cfg.get("channel", DEFAULT_CHANNEL))
    label = _CHANNEL_LABEL.get(channel, rf"$S_{{{channel}}}(\mathbf{{q}})$")
    cbar = fig.colorbar(mesh, ax=axs[0].tolist(), shrink=0.85, pad=0.03)
    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=13)

    suptitle = f"Frame {int(first['frame_index'])}, timestep {int(first['timestep'])}"
    if len(frames) > 1:
        suptitle += f"    averaged over {len(frames)} frames"
    fig.suptitle(suptitle, fontsize=16)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    return rgba.copy()


def render_png_bytes(frame: dict, cfg: dict) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(render_rgba(frame, cfg)).save(buffer, format="PNG")
    return buffer.getvalue()


def resolve_range(frames: list, cfg: dict) -> tuple[float, float]:
    """Colour limits measured across every frame that will be drawn.

    The peak grows by orders of magnitude while a spiral is being selected, so
    a per-frame range would rescale the colours under the viewer and hide the
    very effect the animation is meant to show.
    """
    if cfg.get("vmin") is not None and cfg.get("vmax") is not None:
        return cfg["vmin"], cfg["vmax"]
    peak = 0.0
    for frame in frames:
        for _, values, _ in frame_intensity(frame, cfg):
            peak = max(peak, float(np.max(values)))
    if peak <= 0.0:
        peak = 1.0
    if cfg.get("scale", "log") == "linear":
        return (cfg.get("vmin") or 0.0, cfg.get("vmax") or peak)
    floor = peak * 10.0 ** (-abs(cfg.get("log_range", 5.0)))
    return (cfg.get("vmin") or floor, cfg.get("vmax") or peak)


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bzmap.py",
        description="Render S(q) over the Brillouin zone for one frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    add_bz_arguments(parser)
    parser.add_argument("--frame", type=int, default=0, help="First frame to use")
    parser.add_argument("--out", type=Path, default=Path("bzmap.png"), help="Output image")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.vector is None:
        raise ValueError(
            "S(q) is built from the spin field, so --vector is required, e.g. "
            "--vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]'."
        )
    cfg = bz_config_from_args(args)

    window = max(1, int(args.frame_average))
    if window == 1:
        frames = [load_single_frame(args.input_dump, args.frame,
                                    vector=tuple(args.vector), element=args.element,
                                    drop_zero_vector=args.drop_zero_vector)]
    else:
        frames, _ = load_frames(args.input_dump, vector=tuple(args.vector),
                                element=args.element,
                                drop_zero_vector=args.drop_zero_vector,
                                frame_start=args.frame, frame_stop=args.frame + window)
        if not frames:
            raise ValueError(f"No frames from index {args.frame}; the dump is shorter.")

    print(f"[INFO] Frame {args.frame}, timestep {int(frames[0]['timestep'])}, "
          f"{frames[0]['x'].size} atoms, channel {args.channel}"
          + (f", averaged over {len(frames)} frames" if len(frames) > 1 else ""))
    maps = average_intensity(frames, cfg) if len(frames) > 1 else frame_intensity(frames[0], cfg)
    for index, (q_grid, values, _) in enumerate(maps, start=1):
        flat = values.ravel()
        q_flat = q_grid.reshape(-1, 3)
        order = np.argsort(-flat)
        gamma = int(np.argmin(np.linalg.norm(q_flat, axis=1)))
        best = [i for i in order[:6] if i != gamma][:3]
        print(f"[INFO] layer {index}: {values.shape[0]} x {values.shape[1]} allowed q, "
              f"sum = {flat.sum():.4g}")
        for i in best:
            q = q_flat[i]
            print(f"[INFO]   q = ({q[0]:+.4f}, {q[1]:+.4f})  |q| = {np.linalg.norm(q):.4f}  "
                  f"S = {flat[i]:.5g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames_for_render = frames if len(frames) > 1 else frames[0]
    Image.fromarray(render_rgba(frames_for_render, cfg)).save(args.out)
    print(f"[INFO] Image saved to  : {args.out}")


if __name__ == "__main__":
    main()
