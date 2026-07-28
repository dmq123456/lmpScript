#!/usr/bin/env python3
"""Plot the dynamic structure factor S(q, omega) at a fixed q from an sqw npz.

The npz is produced by sqw_spin.py or sqw_spin_corr.py with --save-npz; it
contains `sqw` of shape (nq, nfreq) and the frequency axis `freq_thz`.

Two modes (mirroring plot_corr_qt.py):
  * With --iq IQ : plot S(iq, omega).
  * Without --iq : list every q index with its fractional coordinate and
                   path distance, so you can pick an index to plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THZ_TO_MEV = 4.135667696


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="Path to sqw npz (needs sqw and freq_thz)")
    parser.add_argument(
        "--iq",
        type=int,
        default=None,
        help="q index to plot. If omitted, list all q indices and their coordinates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: sqw_q<iq>.png next to the npz).",
    )
    parser.add_argument(
        "--max-freq-thz",
        type=float,
        default=None,
        help="Upper limit of the frequency axis in THz (default: full range).",
    )
    parser.add_argument(
        "--mev",
        action="store_true",
        help="Use energy axis in meV instead of THz.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use a logarithmic intensity axis (reveals weak peaks and tails).",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize S by its maximum over the plotted range (default: off).",
    )
    parser.add_argument(
        "--mark-peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark and print the strongest peak in the plotted range (default: on).",
    )
    return parser


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def list_q_indices(data: dict[str, np.ndarray]) -> None:
    q_frac = np.asarray(data["q_frac"], dtype=float)
    nq = q_frac.shape[0]
    dist = np.asarray(data["q_path_distance"], dtype=float) if "q_path_distance" in data else None
    node_idx = set()
    if "q_node_indices" in data and data["q_node_indices"] is not None:
        node_idx = {int(i) for i in np.atleast_1d(data["q_node_indices"]).tolist()}

    print(f"# npz          : contains {nq} q points")
    if "sqw" in data:
        print(f"# sqw          : shape {data['sqw'].shape} (nq, nfreq)")
    if "freq_thz" in data:
        fz = np.asarray(data["freq_thz"], dtype=float)
        print(f"# freq_thz     : {fz.size} points, {fz.min():.4f} .. {fz.max():.4f} THz")
    print(f"# {'iq':>5}  {'qx':>10} {'qy':>10} {'qz':>10}  {'path_dist':>12}  node")
    for iq in range(nq):
        d = f"{dist[iq]:12.5f}" if dist is not None else f"{'':>12}"
        mark = "  <-- node" if iq in node_idx else ""
        print(f"  {iq:5d}  {q_frac[iq,0]:10.5f} {q_frac[iq,1]:10.5f} {q_frac[iq,2]:10.5f}  {d}{mark}")


def plot_sqw(data: dict[str, np.ndarray], iq: int, args: argparse.Namespace) -> Path:
    if "sqw" not in data or "freq_thz" not in data:
        raise KeyError("npz needs both 'sqw' and 'freq_thz'. Re-run with --save-npz.")
    sqw = np.asarray(data["sqw"], dtype=float)
    freq_thz = np.asarray(data["freq_thz"], dtype=float)
    nq, nfreq = sqw.shape
    if not (0 <= iq < nq):
        raise IndexError(f"--iq {iq} out of range [0, {nq - 1}]")

    x = freq_thz * THZ_TO_MEV if args.mev else freq_thz
    xlabel = "Energy (meV)" if args.mev else "Frequency (THz)"
    xmax_thz = args.max_freq_thz if args.max_freq_thz is not None else freq_thz.max()
    in_range = freq_thz <= xmax_thz

    s = sqw[iq].copy()
    if args.normalize:
        smax = s[in_range].max()
        if smax > 0.0:
            s = s / smax

    q_frac = np.asarray(data["q_frac"], dtype=float)[iq]
    dist = float(np.asarray(data["q_path_distance"], dtype=float)[iq]) if "q_path_distance" in data else float("nan")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plot_fn = ax.semilogy if args.logy else ax.plot
    plot_fn(x[in_range], np.maximum(s[in_range], 1e-12) if args.logy else s[in_range], lw=1.6, color="C0")

    peak_txt = ""
    if args.mark_peak:
        idx_range = np.where(in_range)[0]
        ipk = idx_range[np.argmax(s[in_range])]
        xpk = x[ipk]
        ax.axvline(xpk, color="C3", ls="--", lw=1.0, alpha=0.8)
        unit = "meV" if args.mev else "THz"
        peak_txt = f"   peak @ {xpk:.3f} {unit} ({freq_thz[ipk]:.3f} THz)"
        print(f"[INFO] Peak position: {xpk:.4f} {unit}  = {freq_thz[ipk]:.4f} THz")

    ax.set_xlabel(xlabel, fontsize=13)
    ylabel = "S(q,w) / max" if args.normalize else "S(q,w) (arb. units)"
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(x[in_range].min(), x[in_range].max())
    if not args.logy:
        ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=11)

    qtxt = f"[{q_frac[0]:.4f}, {q_frac[1]:.4f}, {q_frac[2]:.4f}]"
    ax.set_title(f"S(q, w)   iq={iq}   q_frac={qtxt}   path_dist={dist:.4f}{peak_txt}", fontsize=12)

    out = args.output if args.output is not None else args.npz.with_name(f"sqw_q{iq}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    data = load_npz(args.npz)

    if args.iq is None:
        list_q_indices(data)
        return

    out = plot_sqw(data, args.iq, args)
    print(f"[INFO] q index      : {args.iq}")
    print(f"[INFO] Output image : {out}")


if __name__ == "__main__":
    main()
