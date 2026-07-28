#!/usr/bin/env python3
"""Plot the time-correlation C(q, tau) at a fixed q from an sqw_spin_corr npz.

The npz must have been written with --save-npz --save-corr-plus, so that it
contains the complex one-sided correlation `corr_plus` of shape (nq, nt, 3).

Two modes:
  * With --iq IQ : plot C(iq, tau) (Re / Im / |C|, and ln|C|).
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

COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Upper envelope of a real oscillating signal via the analytic signal.

    Equivalent to |Hilbert(x)|. For x(t) = A e^{-gamma t} cos(w t + phi) this
    recovers the smooth decay A e^{-gamma t}, so |C| no longer dips to zero at
    every half period. Implemented with numpy FFT to avoid a scipy dependency.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * h))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="Path to sqw_corr_results.npz (needs corr_plus)")
    parser.add_argument(
        "--iq",
        type=int,
        default=None,
        help="q index to plot. If omitted, list all q indices and their coordinates.",
    )
    parser.add_argument(
        "--component",
        choices=["x", "y", "z", "all"],
        default=None,
        help="Field component to plot (default: components stored in npz, else x).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: corr_q<iq>.png next to the npz).",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each component by |C(tau=0)| (default: on).",
    )
    parser.add_argument(
        "--logabs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a semilog envelope-vs-tau panel to eyeball the exponential decay (default: on).",
    )
    parser.add_argument(
        "--envelope",
        choices=["hilbert", "none"],
        default="hilbert",
        help=(
            "Envelope shown/used: 'hilbert' = smooth analytic-signal upper envelope "
            "(correct for real cos-type C that crosses zero); 'none' = raw |C| "
            "(default: hilbert)."
        ),
    )
    parser.add_argument(
        "--max-tau-ps",
        type=float,
        default=None,
        help="Limit the tau axis to this many ps (default: full range).",
    )
    return parser


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _as_str_list(value: object) -> list[str]:
    arr = np.atleast_1d(value)
    return [str(v) for v in arr.tolist()]


def resolve_components(data: dict[str, np.ndarray], requested: str | None) -> list[str]:
    if requested == "all":
        return ["x", "y", "z"]
    if requested is not None:
        return [requested]
    # Auto: use the components the S(q,w) was built from, if available and single-char.
    if "components" in data:
        stored = "".join(_as_str_list(data["components"])).lower()
        comps = [c for c in ["x", "y", "z"] if c in stored]
        if len(comps) == 1:
            return comps
    return ["x"]


def list_q_indices(data: dict[str, np.ndarray]) -> None:
    q_frac = np.asarray(data["q_frac"], dtype=float)
    nq = q_frac.shape[0]
    dist = np.asarray(data["q_path_distance"], dtype=float) if "q_path_distance" in data else None
    node_idx = set()
    if "q_node_indices" in data and data["q_node_indices"] is not None:
        node_idx = {int(i) for i in np.atleast_1d(data["q_node_indices"]).tolist()}

    print(f"# npz          : contains {nq} q points")
    if "corr_plus" in data:
        cp = data["corr_plus"]
        print(f"# corr_plus    : shape {cp.shape} (nq, nt, 3), dtype {cp.dtype}")
    dt_fs = float(np.atleast_1d(data["dt_fs"])[0]) if "dt_fs" in data else None
    if dt_fs is not None:
        print(f"# dt_fs        : {dt_fs}")
    print(f"# {'iq':>5}  {'qx':>10} {'qy':>10} {'qz':>10}  {'path_dist':>12}  node")
    for iq in range(nq):
        d = f"{dist[iq]:12.5f}" if dist is not None else f"{'':>12}"
        mark = "  <-- node" if iq in node_idx else ""
        print(f"  {iq:5d}  {q_frac[iq,0]:10.5f} {q_frac[iq,1]:10.5f} {q_frac[iq,2]:10.5f}  {d}{mark}")


def plot_corr(data: dict[str, np.ndarray], iq: int, args: argparse.Namespace) -> Path:
    if "corr_plus" not in data:
        raise KeyError(
            "npz has no 'corr_plus'. Re-run sqw_spin_corr.py with --save-npz --save-corr-plus."
        )
    corr_plus = np.asarray(data["corr_plus"])
    nq, nt, _ = corr_plus.shape
    if not (0 <= iq < nq):
        raise IndexError(f"--iq {iq} out of range [0, {nq - 1}]")

    dt_fs = float(np.atleast_1d(data["dt_fs"])[0]) if "dt_fs" in data else 1.0
    tau_ps = np.arange(nt) * dt_fs / 1000.0

    q_frac = np.asarray(data["q_frac"], dtype=float)[iq]
    dist = float(np.asarray(data["q_path_distance"], dtype=float)[iq]) if "q_path_distance" in data else float("nan")
    corr_norm = str(np.atleast_1d(data["corr_norm"])[0]) if "corr_norm" in data else "?"

    components = resolve_components(data, args.component)
    comp_indices = [COMPONENT_INDEX[c] for c in components]

    npanels = 2 if args.logabs else 1
    fig, axs = plt.subplots(npanels, 1, figsize=(9, 4.2 * npanels), squeeze=False, constrained_layout=True)
    ax = axs[0, 0]

    def envelope_of(c: np.ndarray) -> tuple[np.ndarray, str]:
        if args.envelope == "hilbert":
            return analytic_envelope(c.real), "envelope (Hilbert)"
        return np.abs(c), "|C|"

    single = len(components) == 1
    for comp, ci in zip(components, comp_indices):
        c = corr_plus[iq, :, ci].astype(complex)
        scale = abs(c[0]) if (args.normalize and abs(c[0]) > 0.0) else 1.0
        c = c / scale
        env, env_label = envelope_of(c)
        if single:
            ax.plot(tau_ps, c.real, lw=1.0, alpha=0.8, label=r"$\mathrm{Re}\,C$")
            ax.plot(tau_ps, c.imag, lw=1.0, alpha=0.8, label=r"$\mathrm{Im}\,C$")
            ax.plot(tau_ps, env, lw=2.0, color="k", label=env_label)
        else:
            ax.plot(tau_ps, env, lw=1.6, label=rf"$C_{comp}$ {env_label}")

    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel(r"$\tau$ (ps)", fontsize=13)
    ylab = r"$C(q,\tau)/|C(q,0)|$" if args.normalize else r"$C(q,\tau)$"
    ax.set_ylabel(ylab, fontsize=13)
    ax.legend(fontsize=11, ncol=3)
    ax.tick_params(labelsize=11)
    if args.max_tau_ps is not None:
        ax.set_xlim(0.0, args.max_tau_ps)

    if args.logabs:
        ax2 = axs[1, 0]
        for comp, ci in zip(components, comp_indices):
            c = corr_plus[iq, :, ci].astype(complex)
            scale = abs(c[0]) if (args.normalize and abs(c[0]) > 0.0) else 1.0
            env, _ = envelope_of(c / scale)
            ax2.semilogy(tau_ps, np.maximum(env, 1e-12), lw=1.6, label=rf"$C_{comp}$")
        ax2.set_xlabel(r"$\tau$ (ps)", fontsize=13)
        ylab2 = "envelope (log)" if args.envelope == "hilbert" else r"$|C(q,\tau)|$ (log)"
        ax2.set_ylabel(ylab2, fontsize=13)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.tick_params(labelsize=11)
        if len(components) > 1:
            ax2.legend(fontsize=11)
        if args.max_tau_ps is not None:
            ax2.set_xlim(0.0, args.max_tau_ps)

    qtxt = f"[{q_frac[0]:.4f}, {q_frac[1]:.4f}, {q_frac[2]:.4f}]"
    fig.suptitle(
        f"C(q, tau)   iq={iq}   q_frac={qtxt}   path_dist={dist:.4f}\n"
        f"components={','.join(components)}   corr_norm={corr_norm}   nt={nt}   dt={dt_fs:g} fs",
        fontsize=12,
    )

    out = args.output if args.output is not None else args.npz.with_name(f"corr_q{iq}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    data = load_npz(args.npz)

    if args.iq is None:
        list_q_indices(data)
        return

    out = plot_corr(data, args.iq, args)
    print(f"[INFO] q index      : {args.iq}")
    print(f"[INFO] Output image : {out}")


if __name__ == "__main__":
    main()
