#!/usr/bin/env python3
"""Per-q DHO-sum fitting of S(q, omega) to extract magnon lifetimes.

For each q index independently (NO symmetry averaging), the constant-q cut
S(q, omega) is fitted with

    S(f) = c0 + [quasi-elastic Lorentzian at f=0]
              + sum_m  A_m * G_m * f0_m^2 / ((f^2 - f0_m^2)^2 + G_m^2 * f^2)

where f is the frequency in THz. Each term is a damped-harmonic-oscillator
(DHO) line. The number of DHO modes M_q is chosen automatically per q:
generous peak detection sets an upper bound, then BIC backward-elimination
prunes modes that are not justified.

Reported per mode:
  f0            peak (bare) frequency [THz]
  linewidth     G_m = FWHM in frequency [THz]
  lifetime      tau = 1 / (pi * G_m) [ps]     (tau = 2/Gamma_angular)
  damping_ratio G_m / (2 f0_m); > ~0.7 => strongly/over-damped
plus per-mode reliability flags (resolution / SNR / convergence).

Two modes:
  * --iq IQ [IQ ...] : inspection. Fit each listed q, print its modes, and save
                       an S(q,w)+fit overlay plot per q.
  * --batch [START STOP] : batch-fit. Fit many q, write a CSV of all modes and a
                       tau(q) plot. '--batch' alone = all q; '--batch START STOP'
                       = only q in [START, STOP). Parallelizes over q under mpirun.
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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqw_mpi import (
    _progress_report_interval,
    _q_indices_for_rank,
    _report_q_progress,
    resolve_mpi_comm,
)

# Keys actually needed for fitting; avoids loading the huge corr_plus array on
# every MPI rank.
NEEDED_KEYS = ["sqw", "freq_thz", "q_frac", "q_vectors", "q_path_distance", "q_node_indices"]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def dho(f: np.ndarray, amp: float, f0: float, gamma: float) -> np.ndarray:
    return amp * gamma * f0**2 / ((f**2 - f0**2) ** 2 + (gamma * f) ** 2)


def quasi_elastic(f: np.ndarray, b: float, gc: float) -> np.ndarray:
    return b * gc / (f**2 + gc**2)


def make_model(n_dho: int, use_qe: bool):
    """Return a curve_fit-compatible model f(x, *params).

    Parameter order: [c0] (+ [B, Gc] if use_qe) + [A, f0, G] * n_dho.
    """

    def model(f: np.ndarray, *params: float) -> np.ndarray:
        idx = 0
        out = np.full_like(f, params[idx], dtype=float)
        idx += 1
        if use_qe:
            out = out + quasi_elastic(f, params[idx], params[idx + 1])
            idx += 2
        for _ in range(n_dho):
            out = out + dho(f, params[idx], params[idx + 1], params[idx + 2])
            idx += 3
        return out

    return model


# --------------------------------------------------------------------------
# Fitting one q
# --------------------------------------------------------------------------
def estimate_noise(s: np.ndarray) -> float:
    """Robust noise sigma via the median absolute deviation."""
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    return 1.4826 * mad if mad > 0 else (np.std(s) or 1.0)


def initial_peaks(freq: np.ndarray, s: np.ndarray, sigma: float, n_sigma: float, max_peaks: int = 6):
    """Generous peak detection -> initial (f0, amp-height, gamma) guesses.

    At most `max_peaks` peaks are kept, ranked by prominence. This bounds the
    model size (and thus the fit time), which matters a lot on noisy real
    spectra where dozens of spurious local maxima cross the threshold and would
    otherwise blow up the per-q fit time.
    """
    df = float(freq[1] - freq[0])
    height = np.median(s) + n_sigma * sigma
    min_dist = max(1, int(round(0.15 / df)))  # >= ~0.15 THz apart
    peaks, props = find_peaks(s, height=height, prominence=n_sigma * sigma, distance=min_dist)
    if peaks.size == 0:
        return []
    if peaks.size > max_peaks:
        keep = np.argsort(props["prominences"])[::-1][:max_peaks]
        peaks = np.sort(peaks[keep])
    widths, _, _, _ = peak_widths(s, peaks, rel_height=0.5)
    guesses = []
    for p, w in zip(peaks, widths):
        f0 = float(freq[p])
        gamma = max(float(w) * df, 2.0 * df)  # FWHM in THz, floored at ~2 bins
        height_p = float(s[p])
        amp = height_p * gamma  # DHO peak height = amp/gamma
        guesses.append((f0, amp, gamma, height_p))
    return guesses


def bic(rss: float, n: int, k: int) -> float:
    if rss <= 0 or n <= 0:
        return -np.inf
    return n * math.log(rss / n) + k * math.log(n)


def _pack(c0, qe, dhos):
    p = [c0]
    if qe is not None:
        p += list(qe)
    for d in dhos:
        p += list(d)
    return p


def fit_with_order(freq, s, sigma, dho_guesses, use_qe, fmax):
    """Fit a fixed (n_dho, use_qe) model; return popt, rss, ok."""
    n_dho = len(dho_guesses)
    model = make_model(n_dho, use_qe)

    c0_0 = max(np.median(s), 1e-12)
    qe_0 = None
    if use_qe:
        qe_0 = (max(float(s[0]) - c0_0, sigma), max(0.3, 2.0 * (freq[1] - freq[0])))
    dhos_0 = [(g[1], g[0], g[2]) for g in dho_guesses]  # (amp, f0, gamma)
    p0 = _pack(c0_0, qe_0, dhos_0)

    # Bounds
    lo = [0.0]
    hi = [np.inf]
    if use_qe:
        lo += [0.0, 1e-6]
        hi += [np.inf, fmax]
    for (amp, f0, gamma) in dhos_0:
        lo += [0.0, max(1e-6, f0 - 1.0), 1e-4]
        hi += [np.inf, min(fmax, f0 + 1.0), 4.0 * fmax]
    try:
        popt, _ = curve_fit(model, freq, s, p0=p0, bounds=(lo, hi), maxfev=20000)
    except Exception:
        return None, np.inf, False
    resid = s - model(freq, *popt)
    rss = float(np.sum(resid**2))
    return popt, rss, True


def unpack(popt, n_dho, use_qe):
    idx = 1
    qe = None
    if use_qe:
        qe = (popt[idx], popt[idx + 1])
        idx += 2
    dhos = []
    for _ in range(n_dho):
        dhos.append((popt[idx], popt[idx + 1], popt[idx + 2]))  # amp, f0, gamma
        idx += 3
    return popt[0], qe, dhos


def fit_q(freq, s, args):
    """Full per-q fit with automatic order selection. Returns a result dict."""
    fmax = float(freq.max())
    sigma = estimate_noise(s)
    guesses = initial_peaks(freq, s, sigma, args.peak_nsigma, args.max_peaks)

    # Decide quasi-elastic inclusion: low-frequency excess not at a detected peak.
    df = float(freq[1] - freq[0])
    low_mask = freq < max(2.0, 3 * df)
    use_qe = bool(np.median(s[low_mask]) > np.median(s) + args.peak_nsigma * sigma)

    n = s.size

    # Start from all detected peaks, then BIC backward-elimination.
    best = None
    current = list(guesses)
    while True:
        popt, rss, ok = fit_with_order(freq, s, sigma, current, use_qe, fmax)
        if ok:
            k = 1 + (2 if use_qe else 0) + 3 * len(current)
            b = bic(rss, n, k)
            if best is None or b < best["bic"] - 1e-9:
                best = {"popt": popt, "rss": rss, "bic": b, "n_dho": len(current), "use_qe": use_qe}
        if len(current) == 0:
            break
        # Drop the weakest (smallest peak height amp/gamma) mode and retry.
        if not ok:
            current = current[:-1]
            continue
        _, _, dhos = unpack(popt, len(current), use_qe)
        heights = [d[0] / max(d[2], 1e-12) for d in dhos]  # amp/gamma
        weakest = int(np.argmin(heights))
        # Map fitted order back to guess list order for removal.
        current = [g for i, g in enumerate(current) if i != weakest]

    if best is None:
        return {"ok": False, "sigma": sigma, "modes": [], "n_dho": 0, "use_qe": use_qe, "r2": 0.0}

    c0, qe, dhos = unpack(best["popt"], best["n_dho"], best["use_qe"])
    ss_tot = float(np.sum((s - np.mean(s)) ** 2))
    r2 = 1.0 - best["rss"] / ss_tot if ss_tot > 0 else 0.0

    modes = []
    for (amp, f0, gamma) in sorted(dhos, key=lambda d: d[1]):
        gamma = abs(gamma)
        f0 = abs(f0)
        height = amp / max(gamma, 1e-12)
        area = amp  # proportional to integrated weight
        lifetime_ps = 1.0 / (math.pi * gamma) if gamma > 0 else np.inf
        damping_ratio = gamma / (2.0 * f0) if f0 > 0 else np.inf
        resolved = gamma > args.resolve_bins * df
        snr = height / sigma
        snr_ok = snr > args.snr_min
        overdamped = damping_ratio > 0.7
        reliable = bool(resolved and snr_ok)
        modes.append(
            {
                "f0_THz": f0,
                "linewidth_FWHM_THz": gamma,
                "lifetime_ps": lifetime_ps,
                "damping_ratio": damping_ratio,
                "amplitude": amp,
                "area": area,
                "peak_height": height,
                "snr": snr,
                "resolved": resolved,
                "overdamped": overdamped,
                "reliable": reliable,
            }
        )

    return {
        "ok": True,
        "sigma": sigma,
        "c0": c0,
        "qe": qe,
        "modes": modes,
        "n_dho": best["n_dho"],
        "use_qe": best["use_qe"],
        "popt": best["popt"],
        "r2": r2,
        "freq_resolution_THz": df,
    }


# --------------------------------------------------------------------------
# I/O and driving
# --------------------------------------------------------------------------
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


def plot_single_fit(freq, s, res, iq, qfrac, dist, out: Path):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True, sharex=True)
    total = make_model(res["n_dho"], res["use_qe"])(freq, *res["popt"]) if res["ok"] else None

    for ax, logy in ((axs[0], False), (axs[1], True)):
        plot = ax.semilogy if logy else ax.plot
        y = np.maximum(s, 1e-12) if logy else s
        plot(freq, y, color="0.5", lw=1.0, label="S(q,w) data")
        if res["ok"]:
            yt = np.maximum(total, 1e-12) if logy else total
            plot(freq, yt, color="k", lw=1.8, label="total fit")
            for i, m in enumerate(res["modes"]):
                comp = dho(freq, m["amplitude"], m["f0_THz"], m["linewidth_FWHM_THz"])
                yc = np.maximum(comp + res["c0"], 1e-12) if logy else comp
                ls = "--" if m["reliable"] else ":"
                plot(freq, yc, lw=1.2, ls=ls, label=f"mode {i}: f0={m['f0_THz']:.2f} THz")
        ax.set_ylabel("S(q,w)  " + ("(log)" if logy else "(arb.)"), fontsize=12)
        ax.tick_params(labelsize=10)
    axs[1].set_xlabel("Frequency (THz)", fontsize=12)
    axs[0].legend(fontsize=9)
    qtxt = f"[{qfrac[0]:.4f}, {qfrac[1]:.4f}, {qfrac[2]:.4f}]"
    r2 = res.get("r2", 0.0)
    axs[0].set_title(
        f"S(q,w) DHO fit   iq={iq}   q={qtxt}   dist={dist:.4f}   "
        f"n_dho={res['n_dho']}  qe={res['use_qe']}  R2={r2:.4f}",
        fontsize=11,
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)


def print_modes(iq, res):
    if not res["ok"] or not res["modes"]:
        print(f"[iq={iq}] no DHO mode found (n_dho={res['n_dho']}, qe={res['use_qe']})")
        return
    print(f"[iq={iq}] n_dho={res['n_dho']}  qe={res['use_qe']}  R2={res['r2']:.4f}  "
          f"freq_res={res['freq_resolution_THz']:.4f} THz")
    print(f"  {'mode':>4} {'f0(THz)':>9} {'FWHM(THz)':>10} {'tau(ps)':>10} "
          f"{'zeta':>7} {'SNR':>7}  flags")
    for i, m in enumerate(res["modes"]):
        flags = []
        if not m["resolved"]:
            flags.append("UNRESOLVED")
        if not (m["snr"] > 0):
            flags.append("NOSNR")
        if m["overdamped"]:
            flags.append("overdamped")
        if m["reliable"]:
            flags.append("ok")
        print(f"  {i:>4} {m['f0_THz']:>9.3f} {m['linewidth_FWHM_THz']:>10.3f} "
              f"{m['lifetime_ps']:>10.3f} {m['damping_ratio']:>7.3f} {m['snr']:>7.1f}  "
              f"{','.join(flags)}")


CSV_FIELDS = [
    "iq", "qx", "qy", "qz", "path_dist", "mode",
    "f0_THz", "linewidth_FWHM_THz", "lifetime_ps", "damping_ratio",
    "amplitude", "peak_height", "snr", "resolved", "overdamped", "reliable",
    "n_dho", "r2",
]


def representative_per_q(rows: list[dict], method: str) -> list[dict]:
    """One representative lifetime per q, from the *reliable* modes only.

    method='dominant': the mode with the largest spectral weight (amplitude A_m).
    method='weighted' : spectral-weight-weighted linewidth Gamma_eff=sum(A_m G_m)/sum(A_m),
                        tau_rep = 1/(pi*Gamma_eff), with a weighted-mean f0.
    q with no reliable mode -> lifetime NaN, reliable=0.
    """
    by_q: dict[int, list[dict]] = {}
    for r in rows:
        by_q.setdefault(int(r["iq"]), []).append(r)

    out = []
    for iq in sorted(by_q):
        modes = by_q[iq]
        rel = [m for m in modes if m["reliable"]]
        rec = {"iq": iq, "path_dist": modes[0]["path_dist"]}
        if not rel:
            rec.update({"lifetime_ps": float("nan"), "f0_THz": float("nan"), "reliable": 0})
            out.append(rec)
            continue
        w = np.array([m["amplitude"] for m in rel], dtype=float)
        f0 = np.array([m["f0_THz"] for m in rel], dtype=float)
        g = np.array([m["linewidth_FWHM_THz"] for m in rel], dtype=float)
        tau = np.array([m["lifetime_ps"] for m in rel], dtype=float)
        wsum = float(w.sum()) if w.sum() > 0 else 1.0
        if method == "weighted":
            g_rep = float(np.sum(w * g) / wsum)
            rec.update({
                "lifetime_ps": 1.0 / (math.pi * g_rep) if g_rep > 0 else float("inf"),
                "f0_THz": float(np.sum(w * f0) / wsum), "reliable": 1,
            })
        else:  # dominant
            j = int(np.argmax(w))
            rec.update({"lifetime_ps": float(tau[j]), "f0_THz": float(f0[j]), "reliable": 1})
        out.append(rec)
    return out


def run_batch(data, args, freq_full, q_indices, mpi_comm, rank, size):
    sqw = np.asarray(data["sqw"], dtype=float)
    nq = sqw.shape[0]
    q_frac = np.asarray(data["q_frac"], dtype=float)
    dist = np.asarray(data["q_path_distance"], dtype=float) if "q_path_distance" in data else np.arange(nq, dtype=float)

    q_indices = np.asarray(q_indices, dtype=int)
    n_sel = int(q_indices.size)
    # Each rank fits a round-robin slice of the SELECTED q subset; this spreads
    # the occasional slow (many-peak) q across ranks reasonably evenly.
    local_idx = q_indices[rank::size]
    local_total = int(local_idx.size)
    # All ranks iterate the same number of steps (padding to the global max) so
    # the collective progress reduction below is called the same number of
    # times on every rank -- otherwise ranks with fewer q deadlock.
    max_local_steps = (n_sel + size - 1) // size if size > 0 else local_total
    report_interval = _progress_report_interval(max_local_steps, 20)
    start = time.perf_counter()

    local_rows = []
    for iloc in range(max_local_steps):
        if iloc < local_total:
            iq = int(local_idx[iloc])
            freq, s = restrict_range(freq_full, sqw[iq], args.max_freq_thz)
            res = fit_q(freq, s, args)
            for i, m in enumerate(res["modes"]):
                local_rows.append({
                    "iq": iq, "qx": q_frac[iq, 0], "qy": q_frac[iq, 1], "qz": q_frac[iq, 2],
                    "path_dist": dist[iq], "mode": i,
                    "f0_THz": m["f0_THz"], "linewidth_FWHM_THz": m["linewidth_FWHM_THz"],
                    "lifetime_ps": m["lifetime_ps"], "damping_ratio": m["damping_ratio"],
                    "amplitude": m["amplitude"], "peak_height": m["peak_height"], "snr": m["snr"],
                    "resolved": int(m["resolved"]), "overdamped": int(m["overdamped"]),
                    "reliable": int(m["reliable"]), "n_dho": res["n_dho"], "r2": res["r2"],
                })
        step = iloc + 1
        if args.progress and (step % report_interval == 0 or step == max_local_steps):
            _report_q_progress("DHO fit", step, local_total, n_sel, start, mpi_comm, rank)

    if mpi_comm is None:
        rows = local_rows
    else:
        gathered = mpi_comm.gather(local_rows, root=0)
        if rank != 0:
            return
        rows = [r for part in gathered for r in part]

    rows.sort(key=lambda r: (r["iq"], r["mode"]))

    csv_path = args.csv if args.csv is not None else args.npz.with_name(args.npz.stem + "_dho.csv")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] Modes found : {len(rows)}")
    print(f"[INFO] CSV written : {csv_path}")

    # tau(q) summary plot
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    rel = [r for r in rows if r["reliable"]]
    unrel = [r for r in rows if not r["reliable"]]
    if unrel:
        ax.scatter([r["path_dist"] for r in unrel], [r["lifetime_ps"] for r in unrel],
                   s=12, c="0.7", marker="x", label="flagged (unreliable)")
    if rel:
        sc = ax.scatter([r["path_dist"] for r in rel], [r["lifetime_ps"] for r in rel],
                        s=[max(8, min(80, r["snr"] * 3)) for r in rel],
                        c=[r["f0_THz"] for r in rel], cmap="viridis", label="reliable")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("mode f0 (THz)")
    if "q_node_indices" in data and data["q_node_indices"] is not None:
        for ni in np.atleast_1d(data["q_node_indices"]).tolist():
            ni = int(ni)
            if 0 <= ni < len(dist):
                ax.axvline(dist[ni], color="k", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("q-path distance", fontsize=13)
    ax.set_ylabel(r"magnon lifetime $\tau$ (ps)", fontsize=13)
    ax.set_yscale("log")
    ax.set_title("Per-q DHO magnon lifetimes (marker size ~ SNR)", fontsize=13)
    ax.legend(fontsize=10)
    out = args.plot if args.plot is not None else args.npz.with_name(args.npz.stem + "_tau_q.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[INFO] tau(q) plot : {out}")

    # Second plot: ONE representative lifetime per q (from reliable modes).
    reps = representative_per_q(rows, args.representative)
    good = [s for s in reps if s["reliable"]]
    n_missing = sum(1 for s in reps if not s["reliable"])
    fig2, ax2 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    if good:
        sc2 = ax2.scatter([s["path_dist"] for s in good], [s["lifetime_ps"] for s in good],
                          s=28, c=[s["f0_THz"] for s in good], cmap="viridis")
        cb2 = fig2.colorbar(sc2, ax=ax2, pad=0.02)
        cb2.set_label("representative f0 (THz)")
    if "q_node_indices" in data and data["q_node_indices"] is not None:
        for ni in np.atleast_1d(data["q_node_indices"]).tolist():
            ni = int(ni)
            if 0 <= ni < len(dist):
                ax2.axvline(dist[ni], color="k", ls="--", lw=0.5, alpha=0.5)
    ax2.set_xlabel("q-path distance", fontsize=13)
    ax2.set_ylabel(r"representative $\tau$ (ps)", fontsize=13)
    ax2.set_yscale("log")
    ax2.set_title(f"Representative magnon lifetime per q ({args.representative}); "
                  f"{len(good)} q shown, {n_missing} without a reliable mode", fontsize=12)
    out_rep = args.plot_rep if args.plot_rep is not None else args.npz.with_name(args.npz.stem + "_tau_q_rep.png")
    fig2.savefig(out_rep, dpi=200)
    plt.close(fig2)
    print(f"[INFO] rep tau(q)  : {out_rep}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="sqw npz (needs sqw, freq_thz)")
    parser.add_argument("--iq", type=int, nargs="+", default=None, metavar="IQ",
                        help="Inspection mode: fit one or more q indices and save an overlay plot for each.")
    parser.add_argument("--batch", type=int, nargs="*", default=None, metavar="START STOP",
                        help="Batch mode: fit many q, write a CSV + tau(q) plot. '--batch' alone = all q; "
                             "'--batch START STOP' = only q in [START, STOP). Parallelizes over q under mpirun.")
    parser.add_argument("--max-freq-thz", type=float, default=None, help="Only fit up to this frequency (THz).")
    parser.add_argument("--peak-nsigma", type=float, default=4.0, help="Peak detection threshold in noise sigma (default: 4).")
    parser.add_argument("--max-peaks", type=int, default=6, help="Cap on detected peaks per q (top by prominence); bounds fit time on noisy spectra (default: 6).")
    parser.add_argument("--resolve-bins", type=float, default=2.0, help="A mode is 'resolved' if FWHM > this many freq bins (default: 2).")
    parser.add_argument("--snr-min", type=float, default=3.0, help="Minimum peak-height/noise for a mode to be reliable (default: 3).")
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path (batch mode).")
    parser.add_argument("--plot", type=Path, default=None, help="Per-mode tau(q) plot path (batch) or overlay path (--iq).")
    parser.add_argument("--plot-rep", type=Path, default=None, help="One-point-per-q representative tau(q) plot path (default: <npz>_tau_q_rep.png).")
    parser.add_argument("--representative", choices=["dominant", "weighted"], default="dominant",
                        help="How to pick the single lifetime per q for the representative plot: 'dominant' = largest spectral weight, 'weighted' = spectral-weight-weighted rate (default: dominant).")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Print batch progress with ETA (rank 0 under mpirun).")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.batch is not None and args.iq is not None:
        parser.error("--iq (inspection) and --batch are mutually exclusive")
    if args.batch is None and args.iq is None:
        parser.error("specify --iq IQ [IQ ...] to inspect, or --batch to batch-fit")

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
            start, stop = args.batch
            start = max(0, start)
            stop = min(nq, stop)
            if stop <= start:
                parser.error(f"--batch START STOP must satisfy 0<=START<STOP<={nq}")
            q_indices = np.arange(start, stop, dtype=int)
        else:
            parser.error("--batch takes no args (all q) or exactly START STOP")
        if rank == 0 and mpi_comm is not None:
            print(f"[INFO] MPI DHO fit enabled: ranks={size}", flush=True)
        if rank == 0:
            print(f"[INFO] Fitting {q_indices.size} q indices "
                  f"[{q_indices[0]}..{q_indices[-1]}]", flush=True)
        run_batch(data, args, freq_full, q_indices, mpi_comm, rank, size)
        return

    # Inspection mode: rank 0 only, one overlay plot per requested q.
    if rank != 0:
        return
    sqw = np.asarray(data["sqw"], dtype=float)
    q_frac_all = np.asarray(data["q_frac"], dtype=float)
    dist_all = np.asarray(data["q_path_distance"], dtype=float) if "q_path_distance" in data else np.arange(nq, dtype=float)
    single = len(args.iq) == 1
    for iq in args.iq:
        if not (0 <= iq < nq):
            print(f"[WARN] --iq {iq} out of range [0, {nq - 1}], skipped", flush=True)
            continue
        freq, s = restrict_range(freq_full, sqw[iq], args.max_freq_thz)
        res = fit_q(freq, s, args)
        print_modes(iq, res)
        out = args.plot if (args.plot is not None and single) else args.npz.with_name(f"sqw_dho_q{iq}.png")
        plot_single_fit(freq, s, res, iq, q_frac_all[iq], dist_all[iq], out)
        print(f"[INFO] Overlay plot : {out}")


if __name__ == "__main__":
    main()
