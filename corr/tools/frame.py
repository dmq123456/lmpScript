#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render one frame of a dump as an image.

    frame.py dump.lammpstrj --frame 1500 --color 'c_outsp[3]' --out check.png

This is both a library and a command. `animate.py` imports `render_rgba` and
calls it in a loop; run directly, it writes a single PNG, which is how to try
out a colormap or a colour range without waiting for a whole animation to
render.

A frame is drawn as a colour field over a triangulation of the atom positions,
optionally with arrows on top. Which columns supply the colour and the arrows is
entirely up to the caller -- spins, Born effective charges and applied fields
all render through the same path.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dumpframe import load_single_frame, spec_label  # noqa: E402
from topocharge import site_density  # noqa: E402


# ----------------------------------------------------------------------
# Derived scalars
# ----------------------------------------------------------------------
# Each entry is (getter, label). The getter takes the frame and the render
# config, because a derived field is not always a function of the vector alone
# -- the topological charge density also needs the positions and the box.
DERIVED = {
    "vx": (lambda f, cfg: f["u"], "$v_x$"),
    "vy": (lambda f, cfg: f["v"], "$v_y$"),
    "vz": (lambda f, cfg: f["w"], "$v_z$"),
    "norm": (lambda f, cfg: np.sqrt(f["u"] ** 2 + f["v"] ** 2 + f["w"] ** 2),
             r"$|\mathbf{v}|$"),
    "inplane": (lambda f, cfg: np.sqrt(f["u"] ** 2 + f["v"] ** 2),
                r"$|\mathbf{v}_\parallel|$"),
    "topo": (site_density, r"$q_i$"),
}


def colour_values(frame: dict, color_spec: str | None,
                  cfg: dict | None = None) -> tuple[np.ndarray, str]:
    """The scalar to colour by, and its label.

    A colour spec is either a dump column expression carried in frame['c'], or
    one of the names in DERIVED, which are computed from the vector instead.
    """
    if color_spec in DERIVED:
        if "u" not in frame:
            raise ValueError(
                f"--color {color_spec} is derived from the vector, so --vector is required."
            )
        getter, label = DERIVED[color_spec]
        return getter(frame, cfg or {}), label
    if "c" in frame:
        return frame["c"], f"${spec_label(color_spec)}$".replace("_", r"\_")
    # No scalar at all: colour by the in-plane vector magnitude so the arrows
    # still sit on something rather than a blank panel.
    return DERIVED["norm"][0](frame, cfg or {}), DERIVED["norm"][1]


def split_layers(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split at the midpoint of the z range, as the old scripts did."""
    midpoint = 0.5 * (z.min() + z.max())
    return z > midpoint, z <= midpoint


def frame_scalar(frame: dict, color_spec: str | None, subtract_mean: bool,
                 cfg: dict | None = None) -> tuple[np.ndarray, float]:
    """Values to colour by, and the frame mean that was removed (0 if none)."""
    values, _ = colour_values(frame, color_spec, cfg)
    if not subtract_mean:
        return values, 0.0
    mean = float(values.mean())
    return values - mean, mean


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def draw_panel(ax, x, y, values, title, cfg, u=None, v=None):
    tri = mtri.Triangulation(x, y)
    mappable = ax.tripcolor(
        tri, values, shading="gouraud",
        cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"],
    )
    if u is not None:
        ax.quiver(
            x, y, u, v, color="k", angles="xy", scale_units="xy",
            scale=cfg["arrow_scale"], width=0.0025,
            headwidth=3.2, headlength=4.2, headaxislength=3.8,
        )
    ax.set_title(title, fontsize=18, weight="bold")
    ax.set_xlabel(r"L$_x$ ($\mathrm{\AA}$)", fontsize=16)
    ax.set_ylabel(r"L$_y$ ($\mathrm{\AA}$)", fontsize=16)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    return mappable


def render_rgba(frame: dict, cfg: dict) -> np.ndarray:
    """Render one frame to an RGBA array."""
    x, y, z = frame["x"], frame["y"], frame["z"]
    values, mean = frame_scalar(frame, cfg["color"], cfg["subtract_mean"], cfg)
    _, label = colour_values(frame, cfg["color"], cfg)
    if cfg["subtract_mean"]:
        label = rf"{label} $-\ \langle\cdot\rangle$"

    has_arrows = "u" in frame and cfg["arrows"]
    u = frame["u"] if has_arrows else None
    v = frame["v"] if has_arrows else None

    width_in, height_in = cfg["figsize"]
    if cfg["single_layer"]:
        fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in),
                               dpi=cfg["dpi"], constrained_layout=True)
        mappable = draw_panel(ax, x, y, values, cfg["title"] or "Field", cfg, u, v)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.03)
    else:
        up, dn = split_layers(z)
        fig, axs = plt.subplots(1, 2, figsize=(2 * width_in, height_in),
                                dpi=cfg["dpi"], constrained_layout=True)
        mappable = draw_panel(axs[0], x[up], y[up], values[up], "Top Layer", cfg,
                              None if u is None else u[up], None if v is None else v[up])
        draw_panel(axs[1], x[dn], y[dn], values[dn], "Bottom Layer", cfg,
                   None if u is None else u[dn], None if v is None else v[dn])
        cbar = fig.colorbar(mappable, ax=axs, shrink=0.85, pad=0.03)

    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=13)

    suptitle = f"Frame {int(frame['frame_index'])}, timestep {int(frame['timestep'])}"
    if cfg["subtract_mean"]:
        suptitle += rf"    $\langle\cdot\rangle = {mean:.6g}$"
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


# ----------------------------------------------------------------------
# Colour range
# ----------------------------------------------------------------------
def resolve_range(frames: list[dict], cfg: dict) -> tuple[float, float]:
    """Colour limits measured on the values that are actually drawn.

    With --subtract-mean and no explicit limits the range is symmetric,
    +/-max|v|, so the midpoint of the colormap lands on zero and a diverging map
    reads sign directly.
    """
    vmin, vmax = cfg["vmin"], cfg["vmax"]
    if vmin is not None and vmax is not None:
        return vmin, vmax

    values = np.concatenate([
        frame_scalar(f, cfg["color"], cfg["subtract_mean"], cfg)[0] for f in frames
    ])
    if cfg["subtract_mean"] and vmin is None and vmax is None:
        extent = float(np.max(np.abs(values))) or 0.5
        return -extent, extent

    low, high = float(values.min()), float(values.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    return (vmin if vmin is not None else low, vmax if vmax is not None else high)


# ----------------------------------------------------------------------
# Shared CLI options
# ----------------------------------------------------------------------
def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dump", type=Path, help="Input LAMMPS dump file")
    parser.add_argument("--vector", nargs=3, metavar=("CX", "CY", "CZ"), default=None,
                        help="Three dump columns drawn as arrows, e.g. "
                             "--vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]'. "
                             "Each accepts '+'-joined names, which are summed")
    parser.add_argument("--color", type=str, default=None, metavar="C",
                        help="Dump column mapped to colour ('+'-joined names are summed), "
                             "or one of vx/vy/vz/norm/inplane/topo derived from --vector. "
                             "'topo' is the topological charge density q_i, which sums to "
                             "the integer charge Q over a layer. "
                             "Defaults to the vector magnitude when only --vector is given")
    parser.add_argument("--element", type=str, default="all",
                        help="Element symbol to include, or 'all'")
    parser.add_argument("--drop-zero-vector", action="store_true",
                        help="Skip atoms whose vector is exactly zero, the way the spin "
                             "animation skipped non-magnetic sites")
    parser.add_argument("--no-arrows", action="store_true",
                        help="Read the vector but do not draw arrows")
    parser.add_argument("--single-layer", action="store_true",
                        help="One panel instead of splitting at the z midpoint")
    parser.add_argument("--subtract-mean", action="store_true",
                        help="Colour the deviation from the frame average; the mean is "
                             "printed in the title")
    parser.add_argument("--topo-grid", type=int, nargs=2, metavar=("N1", "N2"), default=None,
                        help="Cell repeats behind --color topo. Read off the positions when "
                             "omitted; give it only if that inference fails")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--vmin", type=float, default=None, help="Lower colour limit")
    parser.add_argument("--vmax", type=float, default=None, help="Upper colour limit")
    parser.add_argument("--arrow-scale", type=float, default=0.35,
                        help="Quiver scale; smaller draws longer arrows")
    parser.add_argument("--title", type=str, default=None, help="Panel title")
    parser.add_argument("--dpi", type=float, default=100.0,
                        help="Figure resolution. The image is figsize x dpi pixels, so "
                             "--dpi 200 doubles both dimensions. Raise it for a still to "
                             "inspect; leave it low for an animation, where a GIF is "
                             "limited to 256 colours anyway and every frame is held in "
                             "memory until the file is written")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"),
                        default=(7.0, 5.0),
                        help="Panel size in inches. Bilayer mode doubles the width to "
                             "fit two panels")


def check_vector_requirement(args) -> None:
    """A derived colour is a function of the vector, so --vector is mandatory.

    Both commands blank a derived name before handing it to the reader, which
    would otherwise report the frame as having nothing to plot at all.
    """
    if args.color in DERIVED and args.vector is None:
        raise ValueError(
            f"--color {args.color} is derived from the vector, so --vector is required, "
            "e.g. --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]'."
        )


def config_from_args(args, vmin=None, vmax=None) -> dict:
    return {
        "color": args.color,
        "cmap": args.cmap,
        "vmin": args.vmin if vmin is None else vmin,
        "vmax": args.vmax if vmax is None else vmax,
        "single_layer": args.single_layer,
        "topo_grid": tuple(args.topo_grid) if args.topo_grid else None,
        "subtract_mean": args.subtract_mean,
        "arrows": not args.no_arrows,
        "arrow_scale": args.arrow_scale,
        "title": args.title,
        "dpi": args.dpi,
        "figsize": tuple(args.figsize),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frame.py",
        description="Render a single dump frame to an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--frame", type=int, default=0, help="Frame index to render")
    parser.add_argument("--out", type=Path, default=Path("frame.png"), help="Output image")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    check_vector_requirement(args)

    frame = load_single_frame(
        args.input_dump, args.frame,
        vector=tuple(args.vector) if args.vector else None,
        color=None if args.color in DERIVED else args.color,
        element=args.element,
        drop_zero_vector=args.drop_zero_vector,
    )
    print(f"[INFO] Frame {args.frame}, timestep {int(frame['timestep'])}, "
          f"{frame['x'].size} atoms")

    if args.vmin is None or args.vmax is None:
        print("[WARN] The colour range is being taken from this frame alone. An animation")
        print("[WARN] scales across all its frames, so the same frame will look different")
        print("[WARN] there. Pass --vmin and --vmax to compare like with like.")

    cfg = config_from_args(args)
    vmin, vmax = resolve_range([frame], cfg)
    cfg["vmin"], cfg["vmax"] = vmin, vmax
    print(f"[INFO] Colour range    : [{vmin:g}, {vmax:g}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rgba = render_rgba(frame, cfg)
    Image.fromarray(rgba).save(args.out)
    print(f"[INFO] Image size      : {rgba.shape[1]} x {rgba.shape[0]} px "
          f"({cfg['figsize'][0]:g}x{cfg['figsize'][1]:g} in at {cfg['dpi']:g} dpi)")
    print(f"[INFO] Image saved to  : {args.out}")


if __name__ == "__main__":
    main()
