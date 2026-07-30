#!/usr/bin/env python3
"""Per-frame statistics of a Born-effective-charge component.

    bec_mean.py dump.lammpstrj --element Ni --bec-component 1+5+9 --plot

Reports the frame average that `bec_animation.py --subtract-mean` removes, so
the part of the signal the animation discards can still be examined. The
animation shows spatial structure within each frame; this shows how the overall
level moves between frames.

The dump parsing, element filter and component grammar are imported from
bec_animation.py rather than duplicated, so the two always select the same
atoms. What is not imported is the rendering: producing one matplotlib figure
per frame to recover one number per frame would be absurd for a long
trajectory, and skipping it is the whole reason this is a separate script.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bec_animation import (  # noqa: E402
    bec_component_symbol,
    bec_component_text,
    load_selected_frames,
    parse_bec_component,
    split_layers,
)


def frame_statistics(frame_data: dict[str, object], by_layer: bool) -> dict[str, float]:
    """Mean and spread of one frame, optionally split by layer."""
    bec = np.asarray(frame_data["bec"], dtype=float)
    stats = {
        "frame_index": int(frame_data["frame_index"]),
        "timestep": int(frame_data["timestep"]),
        "mean": float(bec.mean()),
        "std": float(bec.std()),
        "min": float(bec.min()),
        "max": float(bec.max()),
        "count": int(bec.size),
    }
    if by_layer:
        z = np.asarray(frame_data["z"], dtype=float)
        up, dn = split_layers(z)
        # A layer can be empty when the selection leaves a single sheet; report
        # NaN rather than raising, so one odd frame does not kill the run.
        stats["mean_top"] = float(bec[up].mean()) if up.any() else float("nan")
        stats["mean_bottom"] = float(bec[dn].mean()) if dn.any() else float("nan")
        stats["mean_diff"] = stats["mean_top"] - stats["mean_bottom"]
        stats["count_top"] = int(up.sum())
        stats["count_bottom"] = int(dn.sum())
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bec_mean.py",
        description="Per-frame mean of a Born-effective-charge component.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dump", type=Path, help="Input LAMMPS dump file")
    p.add_argument("--bec-component", type=str, default="1", metavar="C",
                   help="Component c_outbec[N], row-major 1..9 (xx..zz). Indices joined "
                        "by '+' are summed, so 1+5+9 is the trace")
    p.add_argument("--element", type=str, default="all",
                   help="Element symbol to include, or 'all' for every atom")
    p.add_argument("--frame-start", type=int, default=0, help="First frame, inclusive")
    p.add_argument("--frame-stop", type=int, default=None, help="Stop before this frame")
    p.add_argument("--frame-step", type=int, default=1, help="Keep one frame every N")
    p.add_argument("--by-layer", action="store_true",
                   help="Also report the mean of each layer and their difference, "
                        "splitting at the z midpoint exactly as the animation does")
    p.add_argument("--csv", type=Path, default=None, help="Write the table to this CSV file")
    p.add_argument("--npz", type=Path, default=None, help="Write the arrays to this .npz file")
    p.add_argument("--plot", action="store_true", help="Plot the mean against timestep")
    p.add_argument("--plot-file", type=Path, default=Path("bec_mean.png"),
                   help="Figure output path")
    p.add_argument("--quiet", action="store_true", help="Do not print the table")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="Print dump-reading progress")
    p.add_argument("--progress-reports", type=int, default=20,
                   help="Approximate number of progress updates")
    return p


def plot_means(rows: list[dict[str, float]], symbol: str, by_layer: bool,
               outfile: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["timestep"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.plot(steps, [r["mean"] for r in rows], lw=1.6, label="all selected atoms")
    if by_layer:
        ax.plot(steps, [r["mean_top"] for r in rows], lw=1.2, ls="--", label="top layer")
        ax.plot(steps, [r["mean_bottom"] for r in rows], lw=1.2, ls="--", label="bottom layer")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(rf"$\langle {symbol} \rangle$")
    ax.set_title("Frame average of the Born effective charge")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {outfile}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    components = parse_bec_component(args.bec_component)

    frames, scanned = load_selected_frames(
        dump_path=args.input_dump,
        bec_component=components,
        element=args.element,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        frame_step=args.frame_step,
        progress=args.progress,
        progress_reports=args.progress_reports,
    )
    if not frames:
        raise ValueError("No frames selected; adjust --frame-start/--frame-stop/--frame-step.")

    print(f"[INFO] Frames scanned  : {scanned}")
    print(f"[INFO] Frames selected : {len(frames)}")
    print(f"[INFO] BEC component   : {bec_component_text(components)}")
    print(f"[INFO] Element         : {args.element}")

    rows = [frame_statistics(f, args.by_layer) for f in frames]

    if not args.quiet:
        if args.by_layer:
            header = f"{'frame':>7s}{'timestep':>12s}{'mean':>14s}{'std':>12s}" \
                     f"{'mean_top':>14s}{'mean_bottom':>14s}{'diff':>12s}"
            print(header)
            print("-" * len(header))
            for r in rows:
                print(f"{r['frame_index']:>7d}{r['timestep']:>12d}{r['mean']:>14.6g}"
                      f"{r['std']:>12.4g}{r['mean_top']:>14.6g}"
                      f"{r['mean_bottom']:>14.6g}{r['mean_diff']:>12.4g}")
        else:
            header = f"{'frame':>7s}{'timestep':>12s}{'mean':>14s}{'std':>12s}" \
                     f"{'min':>12s}{'max':>12s}{'atoms':>8s}"
            print(header)
            print("-" * len(header))
            for r in rows:
                print(f"{r['frame_index']:>7d}{r['timestep']:>12d}{r['mean']:>14.6g}"
                      f"{r['std']:>12.4g}{r['min']:>12.4g}{r['max']:>12.4g}{r['count']:>8d}")

    means = np.array([r["mean"] for r in rows])
    print(f"[INFO] Mean over frames: {means.mean():.6g}   "
          f"drift (last - first): {means[-1] - means[0]:+.6g}   "
          f"spread (max - min): {means.max() - means.min():.6g}")

    if args.csv is not None:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Table saved to  : {args.csv}")

    if args.npz is not None:
        arrays = {key: np.array([r[key] for r in rows]) for key in rows[0]}
        arrays["bec_component"] = np.asarray(args.bec_component)
        arrays["element"] = np.asarray(args.element)
        np.savez(args.npz, **arrays)
        print(f"[INFO] Arrays saved to : {args.npz}")

    if args.plot:
        plot_means(rows, bec_component_symbol(components), args.by_layer, args.plot_file)


if __name__ == "__main__":
    main()
