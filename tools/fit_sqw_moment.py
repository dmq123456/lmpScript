#!/usr/bin/env python3
"""Per-q single effective magnon mode from spectral moments of S(q, omega).

Physics assumption: theory says there is only ONE magnon branch per q (besides
the omega=0 quasi-elastic part). The multiple close peaks seen in a spin-MD
S(q,omega) are numerical/statistical splitting of that single branch. We
therefore collapse the whole cluster into ONE effective peak, whose width
absorbs the splitting (larger width, lower height, shorter effective lifetime).

Method of moments over the magnon band [f_lo, f_hi] (background-subtracted,
quasi-elastic region excluded):

    f0    = sum(f * S) / sum(S)                    (centroid, THz)
    sigma = sqrt( sum((f-f0)^2 * S) / sum(S) )     (rms width, THz)
    Gamma_eff = 2*sqrt(2 ln2) * sigma              (Gaussian-equivalent FWHM)
    tau       = 1 / (pi * Gamma_eff)               (ps; same convention as the DHO tool)

The variance automatically counts both each sub-peak's width and the spread
BETWEEN sub-peaks, i.e. the splitting shows up as broadening -- exactly the
"one broad mode" picture. No peak detection, no order selection.

Modes:
  * --iq IQ [IQ ...]  : per-q overlay plot (spectrum + band + f0 + width).
  * --batch [A B]     : fit all q (or [A,B)); write CSV + tau(q) plot.
                        Parallelizes over q under mpirun.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqw_mpi import (
    _progress_report_interval,
    _q_indices_for_rank,
    _report_q_progress,
    resolve_mpi_comm,
)

NEEDED_KEYS = ["sqw", "freq_thz", "q_frac", "q_vectors", "q_path_distance", "q_node_indices"]
GAUSS_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))  # sigma -> FWHM, ~2.3548


def estimate_noise(s: np.ndarray) -> float:
    """Robust noise sigma via the median absolute deviation."""
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    return 1.4826 * mad if mad > 0 else (float(np.std(s)) or 1.0)


def moment_fit_q(freq: np.ndarray, s: np.ndarray, args) -> dict:
    """Collapse S(q,omega) to a single effective mode via spectral moments."""
    df = float(freq[1] - freq[0])
    noise = estimate_noise(s)
    bg = float(np.median(s))
    s_bs = s - bg  # background-subtracted

    result = {
        "ok": False, "noise": noise, "df": df,
        "f0_THz": float("nan"), "sigma_THz": float("nan"),
        "gamma_fwhm_THz": float("nan"), "lifetime_ps": float("nan"),
        "band_lo_THz": float("nan"), "band_hi_THz": float("nan"),
        "peak_height": float("nan"), "snr": 0.0, "n_band": 0,
        "resolved": False, "near_qe": False, "reliable": False,
        "eff_amp": float("nan"),  # peak of the (spike-clipped) weights, for plotting
    }

    # Allowed region: above the quasi-elastic cut, within the fit range.
    allowed = freq >= args.qe_cut
    if not allowed.any():
        return result
    fa = freq[allowed]
    sa = s_bs[allowed]

    ipk = int(np.argmax(sa))
    f_peak = float(fa[ipk])
    speak = float(sa[ipk])
    snr = speak / noise if noise > 0 else 0.0
    result["peak_height"] = speak
    result["snr"] = snr
    if speak <= args.snr_min * noise:
        return result  # no significant magnon peak -> leave unreliable

    # Window of half-width `band_halfwidth` around the peak; within it, weight
    # each bin by its intensity ABOVE a soft noise floor (band_nsigma*noise).
    # The soft floor down-weights near-noise bins so tails/valleys don't
    # dominate, while the window caps how far a single branch's cluster spans.
    # This merges all sub-peaks in the window into ONE broad effective peak.
    win = np.abs(fa - f_peak) <= args.band_halfwidth
    w = np.clip(sa - args.band_nsigma * noise, 0.0, None) * win

    # Optional spike suppression: flatten the tall sharp core to `spike_clip`
    # times the typical (median) in-band level, so the intensity-weighted
    # moment reflects the broad pedestal instead of being dominated by a narrow
    # spike (e.g. a coherent k=0 line). A genuinely broad peak, whose bins never
    # exceed C*median, is left unchanged.
    if args.spike_clip is not None:
        pos = w[w > 0]
        if pos.size:
            cap = args.spike_clip * float(np.median(pos))
            if cap > 0:
                w = np.minimum(w, cap)

    wsum = float(w.sum())
    nz = np.nonzero(w > 0)[0]
    if wsum <= 0 or nz.size < 2:
        return result

    band_f = fa[nz[0]:nz[-1] + 1]  # reported extent of the effective peak
    f0 = float(np.sum(fa * w) / wsum)
    var = float(np.sum(w * (fa - f0) ** 2) / wsum)
    sigma = math.sqrt(max(var, 0.0))
    gamma = GAUSS_FWHM * sigma
    tau = 1.0 / (math.pi * gamma) if gamma > 0 else float("inf")

    resolved = gamma > args.resolve_bins * df
    near_qe = float(band_f[0]) <= args.qe_cut + df  # band touches the qe cut
    reliable = bool(resolved and snr > args.snr_min and not near_qe)

    result.update({
        "ok": True, "f0_THz": f0, "sigma_THz": sigma, "gamma_fwhm_THz": gamma,
        "lifetime_ps": tau, "band_lo_THz": float(band_f[0]), "band_hi_THz": float(band_f[-1]),
        "n_band": int(band_f.size), "resolved": resolved, "near_qe": near_qe,
        "reliable": reliable, "eff_amp": float(w.max()),
    })
    return result


def load_npz(npz_path: Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        wanted = data.files if keys is None else [k for k in keys if k in data.files]
        return {key: data[key] for key in wanted}


def restrict_range(freq, s, fmax):
    if fmax is None:
        return freq, s
    m = freq <= fmax
    return freq[m], s[m]


CSV_FIELDS = [
    "iq", "qx", "qy", "qz", "path_dist",
    "f0_THz", "sigma_THz", "gamma_fwhm_THz", "lifetime_ps",
    "band_lo_THz", "band_hi_THz", "peak_height", "snr", "n_band",
    "resolved", "near_qe", "reliable",
]


def row_from_result(iq, q_frac, dist, res) -> dict:
    return {
        "iq": iq, "qx": q_frac[0], "qy": q_frac[1], "qz": q_frac[2], "path_dist": dist,
        "f0_THz": res["f0_THz"], "sigma_THz": res["sigma_THz"],
        "gamma_fwhm_THz": res["gamma_fwhm_THz"], "lifetime_ps": res["lifetime_ps"],
        "band_lo_THz": res["band_lo_THz"], "band_hi_THz": res["band_hi_THz"],
        "peak_height": res["peak_height"], "snr": res["snr"], "n_band": res["n_band"],
        "resolved": int(res["resolved"]), "near_qe": int(res["near_qe"]),
        "reliable": int(res["reliable"]),
    }


def plot_single(freq, s, res, iq, q_frac, dist, args, out: Path):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True, sharex=True)
    f0, sigma = res["f0_THz"], res["sigma_THz"]
    for ax, logy in ((axs[0], False), (axs[1], True)):
        plot = ax.semilogy if logy else ax.plot
        y = np.maximum(s, 1e-12) if logy else s
        plot(freq, y, color="0.5", lw=1.0, label="S(q,w) data")
        if res["ok"]:
            ax.axvspan(res["band_lo_THz"], res["band_hi_THz"], color="C0", alpha=0.12,
                       label="band")
            # Gaussian-equivalent effective peak. Its height is the spike-clipped
            # peak level (eff_amp), so with --spike-clip it sits at the pedestal
            # instead of the raw spike -- i.e. what the moment actually "sees".
            bg = float(np.median(s))
            amp = res["eff_amp"]
            g = bg + amp * np.exp(-0.5 * ((freq - f0) / max(sigma, 1e-6)) ** 2)
            plot(freq, np.maximum(g, 1e-12) if logy else g, color="k", lw=1.6, ls="--",
                 label=f"eff. peak (FWHM={res['gamma_fwhm_THz']:.2f})")
        ax.set_ylabel("S(q,w)  " + ("(log)" if logy else "(arb.)"), fontsize=12)
        ax.tick_params(labelsize=10)
    axs[1].set_xlabel("Frequency (THz)", fontsize=12)
    axs[0].legend(fontsize=9)
    qtxt = f"[{q_frac[0]:.4f}, {q_frac[1]:.4f}, {q_frac[2]:.4f}]"
    flags = "reliable" if res["reliable"] else ("near_qe" if res["near_qe"] else
             ("UNRESOLVED" if not res["resolved"] else "lowSNR"))
    axs[0].set_title(
        f"S(q,w) moment fit   iq={iq}   q={qtxt}   dist={dist:.4f}\n"
        f"f0={f0:.3f} THz   FWHM_eff={res['gamma_fwhm_THz']:.3f} THz   "
        f"tau={res['lifetime_ps']:.3f} ps   SNR={res['snr']:.1f}   [{flags}]",
        fontsize=11,
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)


def print_result(iq, res):
    if not res["ok"]:
        print(f"[iq={iq}] no significant magnon peak (SNR={res['snr']:.1f})")
        return
    flags = []
    if not res["resolved"]:
        flags.append("UNRESOLVED")
    if res["near_qe"]:
        flags.append("near_qe")
    if res["reliable"]:
        flags.append("ok")
    print(f"[iq={iq}] f0={res['f0_THz']:.3f} THz  FWHM_eff={res['gamma_fwhm_THz']:.3f} THz  "
          f"sigma={res['sigma_THz']:.3f}  tau={res['lifetime_ps']:.3f} ps  "
          f"band=[{res['band_lo_THz']:.2f},{res['band_hi_THz']:.2f}]  SNR={res['snr']:.1f}  "
          f"{','.join(flags)}")


def run_batch(data, args, freq_full, q_indices, mpi_comm, rank, size):
    sqw = np.asarray(data["sqw"], dtype=float)
    nq = sqw.shape[0]
    q_frac = np.asarray(data["q_frac"], dtype=float)
    dist = np.asarray(data["q_path_distance"], dtype=float) if "q_path_distance" in data else np.arange(nq, dtype=float)

    q_indices = np.asarray(q_indices, dtype=int)
    n_sel = int(q_indices.size)
    local_idx = q_indices[rank::size]
    local_total = int(local_idx.size)
    max_local_steps = (n_sel + size - 1) // size if size > 0 else local_total
    report_interval = _progress_report_interval(max_local_steps, 20)
    start = time.perf_counter()

    local_rows = []
    for iloc in range(max_local_steps):
        if iloc < local_total:
            iq = int(local_idx[iloc])
            freq, s = restrict_range(freq_full, sqw[iq], args.max_freq_thz)
            res = moment_fit_q(freq, s, args)
            local_rows.append(row_from_result(iq, q_frac[iq], dist[iq], res))
        step = iloc + 1
        if args.progress and (step % report_interval == 0 or step == max_local_steps):
            _report_q_progress("moment fit", step, local_total, n_sel, start, mpi_comm, rank)

    if mpi_comm is None:
        rows = local_rows
    else:
        gathered = mpi_comm.gather(local_rows, root=0)
        if rank != 0:
            return
        rows = [r for part in gathered for r in part]

    rows.sort(key=lambda r: r["iq"])

    csv_path = args.csv if args.csv is not None else args.npz.with_name(args.npz.stem + "_moment.csv")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    n_rel = sum(1 for r in rows if r["reliable"])
    print(f"[INFO] q fitted    : {len(rows)}  (reliable: {n_rel})")
    print(f"[INFO] CSV written : {csv_path}")

    # tau(q): one point per q by construction.
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    rel = [r for r in rows if r["reliable"]]
    unrel = [r for r in rows if not r["reliable"] and np.isfinite(r["lifetime_ps"])]
    if unrel:
        ax.scatter([r["path_dist"] for r in unrel], [r["lifetime_ps"] for r in unrel],
                   s=12, c="0.7", marker="x", label="flagged")
    if rel:
        sc = ax.scatter([r["path_dist"] for r in rel], [r["lifetime_ps"] for r in rel],
                        s=[max(8, min(80, r["snr"] * 3)) for r in rel],
                        c=[r["f0_THz"] for r in rel], cmap="viridis", label="reliable")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("f0 (THz)")
    if "q_node_indices" in data and data["q_node_indices"] is not None:
        for ni in np.atleast_1d(data["q_node_indices"]).tolist():
            ni = int(ni)
            if 0 <= ni < len(dist):
                ax.axvline(dist[ni], color="k", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("q-path distance", fontsize=13)
    ax.set_ylabel(r"effective magnon lifetime $\tau$ (ps)", fontsize=13)
    ax.set_yscale("log")
    ax.set_title("Single effective magnon lifetime per q (spectral moments)", fontsize=13)
    ax.legend(fontsize=10)
    out = args.plot if args.plot is not None else args.npz.with_name(args.npz.stem + "_tau_q.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[INFO] tau(q) plot : {out}")

    # f0(q): effective magnon dispersion, one point per q.
    figf, axf = plt.subplots(figsize=(11, 5), constrained_layout=True)
    unrel_f = [r for r in rows if not r["reliable"] and np.isfinite(r["f0_THz"])]
    if unrel_f:
        axf.scatter([r["path_dist"] for r in unrel_f], [r["f0_THz"] for r in unrel_f],
                    s=10, c="0.75", marker="x", label="flagged")
    if rel:
        scf = axf.scatter([r["path_dist"] for r in rel], [r["f0_THz"] for r in rel],
                          s=[max(8, min(80, r["snr"] * 3)) for r in rel],
                          c=[r["lifetime_ps"] for r in rel], cmap="viridis", label="reliable")
        cbf = figf.colorbar(scf, ax=axf, pad=0.02)
        cbf.set_label(r"lifetime $\tau$ (ps)")
    if "q_node_indices" in data and data["q_node_indices"] is not None:
        for ni in np.atleast_1d(data["q_node_indices"]).tolist():
            ni = int(ni)
            if 0 <= ni < len(dist):
                axf.axvline(dist[ni], color="k", ls="--", lw=0.5, alpha=0.5)
    axf.set_xlabel("q-path distance", fontsize=13)
    axf.set_ylabel(r"effective peak $f_0$ (THz)", fontsize=13)
    axf.set_title("Effective magnon dispersion f0(q) (spectral moments)", fontsize=13)
    axf.legend(fontsize=10)
    out_f0 = args.plot_f0 if args.plot_f0 is not None else args.npz.with_name(args.npz.stem + "_f0_q.png")
    figf.savefig(out_f0, dpi=200)
    plt.close(figf)
    print(f"[INFO] f0(q) plot  : {out_f0}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="sqw npz (needs sqw, freq_thz)")
    parser.add_argument("--iq", type=int, nargs="+", default=None, metavar="IQ",
                        help="Inspection: fit one or more q, save an overlay plot for each.")
    parser.add_argument("--batch", type=int, nargs="*", default=None, metavar="START STOP",
                        help="Batch: fit many q -> CSV + tau(q) plot. '--batch' = all; '--batch A B' = [A,B). Parallel under mpirun.")
    parser.add_argument("--max-freq-thz", type=float, default=None, help="Only use spectrum up to this frequency (THz).")
    parser.add_argument("--qe-cut", type=float, default=0.8, help="Exclude f < this (THz) as quasi-elastic/zero-freq (default: 0.8).")
    parser.add_argument("--band-halfwidth", type=float, default=1.2, help="Half-width (THz) of the window around the peak over which the cluster is merged into one effective peak (default: 1.2).")
    parser.add_argument("--band-nsigma", type=float, default=3.0, help="Soft noise floor: bins are weighted by max(S - band_nsigma*noise, 0) (default: 3).")
    parser.add_argument("--spike-clip", type=float, default=None, metavar="C",
                        help="Suppress sharp spikes: clip in-band weights to C x their median before the moment, "
                             "so a narrow tall core (e.g. a coherent k=0 line) no longer dominates the width. "
                             "Broad peaks are unaffected. Off by default; try C=3-5.")
    parser.add_argument("--resolve-bins", type=float, default=2.0, help="'resolved' if FWHM_eff > this many freq bins (default: 2).")
    parser.add_argument("--snr-min", type=float, default=3.0, help="Min peak/noise for a reliable mode (default: 3).")
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path (batch).")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Batch: tau(q) plot path. Inspection: overlay path. For several --iq, "
                             "use an '{iq}' placeholder (e.g. tmp/fitsqwy{iq}.png) or give a directory.")
    parser.add_argument("--plot-f0", type=Path, default=None, help="f0(q) dispersion plot path (batch; default: <npz>_f0_q.png).")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Print batch progress (rank 0 under mpirun).")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.batch is not None and args.iq is not None:
        parser.error("--iq and --batch are mutually exclusive")
    if args.batch is None and args.iq is None:
        parser.error("specify --iq IQ [IQ ...] or --batch")

    mpi_comm, rank, size = resolve_mpi_comm()
    data = load_npz(args.npz, keys=NEEDED_KEYS)
    if "sqw" not in data or "freq_thz" not in data:
        raise KeyError("npz needs 'sqw' and 'freq_thz'. Re-run sqw with --save-npz.")
    freq_full = np.asarray(data["freq_thz"], dtype=float)
    nq = np.asarray(data["sqw"]).shape[0]

    if args.batch is not None:
        if len(args.batch) == 0:
            q_indices = np.arange(nq, dtype=int)
        elif len(args.batch) == 2:
            a, b = args.batch
            a = max(0, a); b = min(nq, b)
            if b <= a:
                parser.error(f"--batch START STOP must satisfy 0<=START<STOP<={nq}")
            q_indices = np.arange(a, b, dtype=int)
        else:
            parser.error("--batch takes no args (all q) or exactly START STOP")
        if rank == 0 and mpi_comm is not None:
            print(f"[INFO] MPI moment fit: ranks={size}", flush=True)
        if rank == 0:
            print(f"[INFO] Fitting {q_indices.size} q indices [{q_indices[0]}..{q_indices[-1]}]", flush=True)
        run_batch(data, args, freq_full, q_indices, mpi_comm, rank, size)
        return

    if rank != 0:
        return
    sqw = np.asarray(data["sqw"], dtype=float)
    q_frac_all = np.asarray(data["q_frac"], dtype=float)
    dist_all = np.asarray(data["q_path_distance"], dtype=float) if "q_path_distance" in data else np.arange(nq, dtype=float)
    def overlay_out(iq: int) -> Path:
        # --plot controls the overlay path. It may contain an "{iq}" placeholder
        # (formatted per q); be a bare directory (files named inside it); or, for
        # a single --iq, a plain file path. Without --plot, default next to npz.
        if args.plot is None:
            return args.npz.with_name(f"sqw_moment_q{iq}.png")
        spec = str(args.plot)
        if "{iq}" in spec:
            out = Path(spec.format(iq=iq))
        elif len(args.iq) == 1:
            out = args.plot
        else:  # multiple q, no placeholder -> treat --plot as an output directory
            out = args.plot / f"sqw_moment_q{iq}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    for iq in args.iq:
        if not (0 <= iq < nq):
            print(f"[WARN] --iq {iq} out of range [0, {nq - 1}], skipped", flush=True)
            continue
        freq, s = restrict_range(freq_full, sqw[iq], args.max_freq_thz)
        res = moment_fit_q(freq, s, args)
        print_result(iq, res)
        out = overlay_out(iq)
        plot_single(freq, s, res, iq, q_frac_all[iq], dist_all[iq], args, out)
        print(f"[INFO] Overlay plot : {out}")


if __name__ == "__main__":
    main()
