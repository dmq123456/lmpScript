#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Density of states of a three-component field, from its on-site correlation.

The calculation is the four lines of `dos_for_channel` below. There is no
q-grid and therefore no MPI: one transform per atom is cheap enough that the
whole thing runs in a fraction of the time the dump takes to read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from channels import parse_channels
from correlate import window_array
from dos import channel_correlation, finalise_curve, spectrum_from_correlation
from result import THZ_TO_MEV
from trajectory import load_trajectory


# ----------------------------------------------------------------------
# The calculation
# ----------------------------------------------------------------------
def dos_for_channel(traj, channel, window_vec, cfg):
    """Everything that happens for one requested channel."""
    weight = channel.weight(None)                          # W, q-independent here
    corr = channel_correlation(                            # C(tau), tau >= 0
        traj, weight,
        subtract_mean=cfg["subtract_mean"],
        window_vec=window_vec,
        corr_norm=cfg["corr_norm"],
    )
    freq, raw = spectrum_from_correlation(                 # D(w >= 0)
        corr, cfg["dt_fs"], clip=channel.clips(None),
    )
    return finalise_curve(                                 # trim, smooth, normalise
        freq, raw,
        freq_min_thz=cfg["freq_min_thz"], freq_max_thz=cfg["freq_max_thz"],
        smooth_sigma_thz=cfg["smooth_sigma_thz"], normalize=cfg["normalize"],
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Density of states from the on-site autocorrelation of a "
                    "three-component field. No q-grid is involved.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dumpfile", help="LAMMPS-like dump file (or .npz cache)")
    p.add_argument("--field-columns", nargs=3, metavar=("CX", "CY", "CZ"), default=None,
                   help="Dump column names holding the three field components")
    p.add_argument("--dt-fs", type=float, default=1.0, help="Time between saved frames, in fs")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="Storage dtype for the parsed trajectory; the transform is always float64")
    p.add_argument("--frame-start", type=int, default=None, help="First frame, inclusive")
    p.add_argument("--frame-stop", type=int, default=None, help="Stop before this frame")
    p.add_argument("--frame-step", type=int, default=None, help="Keep one frame every N")
    p.add_argument("--spin-threshold", type=float, default=0.0,
                   help="Drop atoms whose max_t |field| <= this value")
    p.add_argument("--cache-binary", action=argparse.BooleanOptionalAction, default=True,
                   help="Cache the parsed trajectory next to the dump")
    p.add_argument("--cache-file", default=None, help="Explicit cache file path")

    p.add_argument("--mass-weight", action="store_true",
                   help="Scale each atom's field by sqrt(mass) (phonon SED convention)")
    p.add_argument("--masses", type=float, nargs="+", default=None,
                   help="One mass per LAMMPS atom type, in type order")

    p.add_argument("--component", nargs="+", metavar="C", default=["1+5+9"],
                   help="Output channels, same grammar as sqw_spin_corr.py. L and T are "
                        "unavailable: they are built from the direction of q and the DOS "
                        "has no q. Default 1+5+9 is the trace")

    p.add_argument("--no-subtract-mean", action="store_true",
                   help="Keep the time average (leaves the elastic line in)")
    p.add_argument("--window", choices=["hann", "none"], default="hann",
                   help="Data-domain taper applied before correlating, to suppress leakage")
    p.add_argument("--corr-norm", choices=["biased", "unbiased"], default="biased",
                   help="Correlation denominator: Nt, or Nt-tau")

    p.add_argument("--freq-min-thz", type=float, default=None, help="Lowest frequency to keep")
    p.add_argument("--freq-max-thz", type=float, default=None, help="Highest frequency to keep")
    p.add_argument("--smooth-sigma-thz", type=float, default=0.0,
                   help="Gaussian smoothing width in THz; 0 disables")
    p.add_argument("--normalize", choices=["none", "max", "area"], default="max",
                   help="Normalisation applied to the final curve")

    p.add_argument("--save-npz", action="store_true", help="Write results to --output")
    p.add_argument("--output", default="dos_results.npz", help="Output .npz path")
    p.add_argument("--plot", action="store_true", help="Render the DOS curve")
    p.add_argument("--plot-file", default="dos.png", help="Figure output path")
    p.add_argument("--plot-raw", action="store_true",
                   help="Also draw the curve before smoothing and normalisation")
    p.add_argument("--mev", action="store_true", help="Label the axis in meV")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="Print dump-reading progress")
    p.add_argument("--progress-reports", type=int, default=20,
                   help="Approximate number of progress updates")
    return p


def plot_dos(freq_thz, curves, outfile, use_mev, plot_raw) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    x = freq_thz * THZ_TO_MEV if use_mev else freq_thz
    for label, raw, final in curves:
        line, = ax.plot(x, final, linewidth=2.0, label=label)
        if plot_raw and not np.allclose(raw, final):
            ax.plot(x, raw, color=line.get_color(), alpha=0.35, linewidth=1.0)

    ax.set_xlabel("Energy (meV)" if use_mev else "Frequency (THz)")
    ax.set_ylabel("DOS (arb. units)")
    ax.set_title("Density of states")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    png = f"{Path(outfile).with_suffix('')}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {png}")


def main(argv=None) -> None:
    args = build_parser(prog="sqw_dos.py").parse_args(argv)

    channels = parse_channels(args.component)
    for channel in channels:
        if channel.needs_qhat:
            raise ValueError(
                f"Channel {channel.label!r} uses L or T, whose weights are built from the "
                f"direction of q. The DOS is a q-integrated quantity and has no q, so only "
                f"constant weights are available: 1..9, xx..zz, x/y/z and their '+' groups."
            )

    # No supercell: without q there is no use for a reciprocal lattice.
    traj = load_trajectory(
        args.dumpfile,
        dtype=np.dtype(args.dtype),
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        frame_step=args.frame_step,
        cache_binary=args.cache_binary,
        cache_file=args.cache_file,
        spin_threshold=args.spin_threshold,
        field_columns=args.field_columns,
        progress=args.progress,
        progress_reports=args.progress_reports,
    )
    if args.mass_weight:
        if not args.masses:
            raise ValueError("--mass-weight requires --masses")
        traj.set_mass_weights(args.masses)
    elif args.masses:
        print("[INFO] --masses given without --mass-weight; ignored (magnon convention).")

    traj.print_lattice_info()
    print(f"[INFO] Frames: {traj.n_frames}   Atoms: {traj.n_atoms}")
    print(f"[INFO] Field columns: {traj.field_columns}")
    print(f"[INFO] Channels: {[c.label for c in channels]}")

    window_vec = window_array(traj.n_frames, args.window)
    cfg = {
        "subtract_mean": not args.no_subtract_mean,
        "corr_norm": args.corr_norm,
        "dt_fs": args.dt_fs,
        "freq_min_thz": args.freq_min_thz,
        "freq_max_thz": args.freq_max_thz,
        "smooth_sigma_thz": args.smooth_sigma_thz,
        "normalize": args.normalize,
    }

    payload: dict[str, object] = {
        "dt_fs": np.asarray(args.dt_fs, dtype=float),
        "corr_norm": np.asarray(args.corr_norm),
        "window": np.asarray(args.window),
        "normalize": np.asarray(args.normalize),
        "smooth_sigma_thz": np.asarray(args.smooth_sigma_thz, dtype=float),
        "components": np.asarray([c.label for c in channels]),
        "field_columns": np.asarray(traj.field_columns),
        "lattice": traj.lattice,
        "timesteps": traj.timesteps,
    }

    curves = []
    freq = None
    for channel in channels:
        freq, raw, final = dos_for_channel(traj, channel, window_vec, cfg)
        curves.append((channel.label, raw, final))
        tag = channel.label.replace("+", "p")
        payload[f"dos_{tag}"] = final
        payload[f"dos_raw_{tag}"] = raw
        print(f"[INFO] [{channel.label}] {freq.size} frequency points, "
              f"clip={'on' if channel.clips(None) else 'off'}")

    payload["freq_thz"] = freq
    payload["energy_meV"] = freq * THZ_TO_MEV

    if args.save_npz:
        np.savez(args.output, **payload)
        print(f"[INFO] Results saved to: {args.output}")

    if args.plot:
        plot_dos(freq, curves, args.plot_file, args.mev, args.plot_raw)


if __name__ == "__main__":
    main()
