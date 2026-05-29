#!/usr/bin/env python3
"""
Read an sqw_results.npz file and inspect S(q,w).

Usage:
  python read_npz.py                          # summary + harmonic scan
  python read_npz.py --q-index 300            # S(q,w) at a specific q-point
  python read_npz.py --freq 4.0               # S(q) slice at a given freq (THz)
  python read_npz.py --plot                   # quick re-plot
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

FILE = Path(__file__).resolve().parent / "sqw_results.npz"


def load(filepath: Path) -> dict[str, np.ndarray]:
    """Return arrays from an sqw .npz file."""
    data = dict(np.load(filepath, allow_pickle=True))
    # convert 0-d arrays back to scalars where appropriate
    for k in ("components", "projection", "freq_mode"):
        if k in data and data[k].ndim == 0:
            data[k] = str(data[k])
    return data


def summary(data: dict) -> None:
    """Print a one-page summary of the dataset."""
    sqw = data["sqw"]
    freq = data["freq_thz"]
    q_dist = data["q_path_distance"]
    labels = data.get("q_node_indices")
    label_names = None  # not stored in this file version

    nq, nf = sqw.shape
    print(f"S(q,w)  shape : {nq} q-points  x  {nf} frequencies")
    print(f"q range       : {q_dist[0]:.3f} -> {q_dist[-1]:.3f} (cumulative distance)")
    print(f"freq range    : {freq[0]:.4f} -> {freq[-1]:.4f} THz")
    print(f"freq step     : {freq[1] - freq[0]:.5f} THz")
    print(f"components    : {data.get('components', '?')}")
    print(f"projection    : {data.get('projection', '?')}")
    print(f"dt            : {float(data['dt_fs'])} fs")
    i_max = int(np.argmax(sqw.max(axis=1)))
    print(f"sqw max       : {sqw.max():.2f}  (q_idx={i_max}, q_dist={q_dist[i_max]:.4f})")
    print(f"sqw > 0.1*max : {(sqw > 0.1*sqw.max()).sum()} / {sqw.size} pixels")

    if labels is not None:
        print(f"\nHigh-symmetry nodes (q-path distance):")
        for i, idx in enumerate(labels):
            name = label_names[i] if label_names else f"node{i}"
            print(f"  {name}: q_dist = {q_dist[idx]:.4f}")


def scan_harmonics(data: dict, base_freq: float = 4.0, n_harm: int = 6) -> None:
    """Print intensity at harmonic multiples across the q-path."""
    sqw = data["sqw"]
    freq = data["freq_thz"]
    q_dist = data["q_path_distance"]

    print(f"\nHarmonic intensities at multiples of {base_freq} THz:")
    print(f"{'n':<4} {'freq (THz)':<12} {'mean':<12} {'max':<12} {'q_idx':<8} {'q_dist':<10}")
    print("-" * 60)

    for n in range(n_harm):
        f_target = n * base_freq
        if f_target > freq[-1]:
            break
        i_freq = np.argmin(np.abs(freq - f_target))
        f_actual = freq[i_freq]
        row = sqw[:, i_freq]
        i_max = int(np.argmax(row))
        print(
            f"{n:<4} {f_actual:<12.4f} {row.mean():<12.2f} "
            f"{row.max():<12.2f} {i_max:<8} {q_dist[i_max]:<10.4f}"
        )


def show_q_slice(data: dict, q_index: int) -> None:
    """Print S(q,w) vs frequency at a single q-point."""
    sqw = data["sqw"]
    freq = data["freq_thz"]
    q_dist = data["q_path_distance"]

    if q_index < 0 or q_index >= sqw.shape[0]:
        print(f"q_index out of range [0, {sqw.shape[0] - 1}]")
        return

    row = sqw[q_index]
    print(f"\nS(q,w) at q_index={q_index}, q_dist={q_dist[q_index]:.4f}")
    print(f"max = {row.max():.4f} at {freq[row.argmax()]:.4f} THz")
    print(f"\n{'freq (THz)':<12} {'S(q,w)':<12}")
    print("-" * 24)

    # show top peaks
    from scipy.signal import find_peaks

    peaks, props = find_peaks(row, prominence=row.max() * 0.01, distance=5)
    if len(peaks) == 0:
        # fallback: show non-zero entries
        nz = np.where(row > row.max() * 0.01)[0]
        for i in nz[:20]:
            print(f"{freq[i]:<12.4f} {row[i]:<12.4f}")
    else:
        order = np.argsort(row[peaks])[::-1]
        for idx in order[:20]:
            i = peaks[idx]
            print(f"{freq[i]:<12.4f} {row[i]:<12.4f}")


def show_freq_slice(data: dict, freq_thz: float) -> None:
    """Print S(q) at a fixed frequency across the q-path."""
    sqw = data["sqw"]
    freq = data["freq_thz"]
    q_dist = data["q_path_distance"]

    i_freq = np.argmin(np.abs(freq - freq_thz))
    f_actual = freq[i_freq]
    col = sqw[:, i_freq]

    print(f"\nS(q) at f = {f_actual:.4f} THz:")
    print(f"mean = {col.mean():.4f}, max = {col.max():.4f}, "
          f"fraction non-zero = {(col > 0).mean():.3f}")
    print(f"\nTop q-points:")
    order = np.argsort(col)[::-1]
    for rank, idx in enumerate(order[:10]):
        if col[idx] <= 0:
            break
        print(f"  q_idx={idx:4d}, q_dist={q_dist[idx]:.4f}, S={col[idx]:.4f}")


def plot_freq_slice(
    data: dict,
    freq_thz: float,
    outfile: Optional[str] = None,
    cbar_min: Optional[float] = None,
    cbar_max: Optional[float] = None,
) -> None:
    """Plot S(q) along the q-path at a fixed frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sqw = data["sqw"]
    freq = data["freq_thz"]
    q_dist = data["q_path_distance"]
    node_idx = data.get("q_node_indices")
    components = data.get("components", "?")

    i_freq = np.argmin(np.abs(freq - freq_thz))
    f_actual = freq[i_freq]
    col = sqw[:, i_freq]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.plot(q_dist, col, linewidth=0.8, color="steelblue")
    ax.fill_between(q_dist, 0, col, alpha=0.15, color="steelblue")
    ax.set_xlabel("q-path distance")
    ax.set_ylabel("S(q, ω)")
    ax.set_title(f"S(q) at ω = {f_actual:.4f} THz  (components={components})")

    if cbar_min is not None or cbar_max is not None:
        ax.set_ylim(bottom=cbar_min or 0, top=cbar_max)

    if node_idx is not None:
        for xpos in q_dist[node_idx]:
            ax.axvline(x=xpos, color="gray", linestyle="--", linewidth=0.6)

    fig.tight_layout()
    path = outfile or f"sqw_slice_{f_actual:.2f}THz.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {path}")


def plot_q_slice(
    data: dict,
    q_index: int,
    outfile: Optional[str] = None,
    max_freq_thz: Optional[float] = None,
    cbar_min: Optional[float] = None,
    cbar_max: Optional[float] = None,
) -> None:
    """Plot S(q,ω) vs ω at a fixed q-point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sqw = data["sqw"]
    freq = data["freq_thz"]
    q_dist = data["q_path_distance"]
    components = data.get("components", "?")

    if q_index < 0 or q_index >= sqw.shape[0]:
        print(f"q_index out of range [0, {sqw.shape[0] - 1}]")
        return

    row = sqw[q_index]

    if max_freq_thz is not None:
        mask = freq <= max_freq_thz
        freq_plot = freq[mask]
        row_plot = row[mask]
    else:
        freq_plot = freq
        row_plot = row

    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.plot(row_plot, freq_plot, linewidth=0.8, color="steelblue")
    ax.fill_betweenx(freq_plot, 0, row_plot, alpha=0.12, color="steelblue")
    ax.set_xlabel("Intensity (arb. units)")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(
        f"S(q,ω) at q_idx={q_index}, q_dist={q_dist[q_index]:.4f}  "
        f"(components={components})"
    )
    if cbar_min is not None or cbar_max is not None:
        ax.set_xlim(left=cbar_min or 0, right=cbar_max)
    if row_plot.max() > 0:
        ax.set_ylim(bottom=0, top=freq_plot[-1])

    fig.tight_layout()
    path = outfile or f"sqw_q{q_index}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {path}")


def quick_plot(
    data: dict,
    outfile: Optional[str] = None,
    max_freq_thz: Optional[float] = None,
    cbar_min: Optional[float] = None,
    cbar_max: Optional[float] = None,
) -> None:
    """Minimal pcolormesh re-plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sqw = data["sqw"]
    freq = data["freq_thz"]
    q_dist = data["q_path_distance"]
    node_idx = data.get("q_node_indices")

    if max_freq_thz is not None:
        mask = freq <= max_freq_thz
    else:
        mask = np.ones(len(freq), dtype=bool)
    sqw_plot = sqw[:, mask]
    freq_plot = freq[mask]

    q_edges = np.empty(len(q_dist) + 1)
    mid = 0.5 * (q_dist[1:] + q_dist[:-1])
    q_edges[1:-1] = mid
    q_edges[0] = q_dist[0] - 0.5 * (q_dist[1] - q_dist[0])
    q_edges[-1] = q_dist[-1] + 0.5 * (q_dist[-1] - q_dist[-2])

    f_edges = np.empty(len(freq_plot) + 1)
    f_mid = 0.5 * (freq_plot[1:] + freq_plot[:-1])
    f_edges[1:-1] = f_mid
    f_edges[0] = freq_plot[0] - 0.5 * (freq_plot[1] - freq_plot[0])
    f_edges[-1] = freq_plot[-1] + 0.5 * (freq_plot[-1] - freq_plot[-2])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    mesh_kw = {"shading": "auto"}
    if cbar_min is not None:
        mesh_kw["vmin"] = cbar_min
    if cbar_max is not None:
        mesh_kw["vmax"] = cbar_max
    im = ax.pcolormesh(q_edges, f_edges, sqw_plot.T, **mesh_kw)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Intensity (arb. units)")
    ax.set_xlabel("q-path distance")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("S(q,w)")

    if node_idx is not None:
        for xpos in q_dist[node_idx]:
            ax.axvline(x=xpos, color="w", linestyle="--", linewidth=0.6)

    fig.tight_layout()
    path = outfile or "sqw_replot.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Read and inspect sqw_results.npz")
    parser.add_argument("--file", type=str, default=str(FILE), help="Path to .npz file")
    parser.add_argument("--q-index", type=int, default=None, help="Show S(q,w) at a specific q index")
    parser.add_argument("--freq", type=float, default=None, help="Show S(q) slice at a given frequency (THz)")
    parser.add_argument("--plot", action="store_true", help="Quick re-plot")
    parser.add_argument("--plot-file", type=str, default=None, help="Output filename for --plot")
    parser.add_argument("--max-freq-thz", type=float, default=None, help="Max frequency for plot (THz)")
    parser.add_argument("--cbar-min", type=float, default=None, help="Colorbar min")
    parser.add_argument("--cbar-max", type=float, default=None, help="Colorbar max")
    args = parser.parse_args()

    data = load(Path(args.file))
    summary(data)
    scan_harmonics(data, base_freq=4.0)

    if args.q_index is not None:
        show_q_slice(data, args.q_index)
        if args.plot:
            plot_q_slice(data, args.q_index, args.plot_file,
                         args.max_freq_thz, args.cbar_min, args.cbar_max)

    if args.freq is not None:
        show_freq_slice(data, args.freq)
        if args.plot:
            plot_freq_slice(data, args.freq, args.plot_file,
                            args.cbar_min, args.cbar_max)
    elif args.plot and args.q_index is None:
        quick_plot(data, args.plot_file, args.max_freq_thz,
                   args.cbar_min, args.cbar_max)


if __name__ == "__main__":
    main()
