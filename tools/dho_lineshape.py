#!/usr/bin/env python3
"""Plot the damped-harmonic-oscillator (DHO) lineshape

    S(omega) = (1/pi) * Gamma * omega0^2 / ((omega^2 - omega0^2)^2 + Gamma^2 * omega^2)

for several damping values Gamma, at fixed bare frequency omega0.

The prefactor Gamma*omega0^2/pi normalizes each curve to unit area, so the
curves show how increasing damping lowers and broadens the peak while the
integrated weight stays fixed. The oscillation frequency (pole real part) is
    omega_q = sqrt(omega0^2 - (Gamma/2)^2),
which is marked for each curve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def dho_lineshape(omega: np.ndarray, omega0: float, gamma_damp: float) -> np.ndarray:
    """DHO spectral function, normalized to unit area.

    gamma_damp is Gamma in the equation of motion (the width parameter that
    multiplies omega in the denominator). The envelope decay rate is Gamma/2.
    """
    denom = (omega**2 - omega0**2) ** 2 + (gamma_damp * omega) ** 2
    return (gamma_damp * omega0**2 / np.pi) / denom


def lorentzian_approx(omega: np.ndarray, omega0: float, gamma_damp: float) -> np.ndarray:
    """Weak-damping Lorentzian approximation of the DHO near the +omega0 peak.

    Obtained from the DHO by omega^2 - omega0^2 ~ 2*omega0*(omega - omega0) and
    Gamma^2 * omega^2 ~ Gamma^2 * omega0^2 around the peak:

        S ~ (Gamma / (4 pi)) / ((omega - omega0)^2 + (Gamma/2)^2)

    It is centered exactly at omega0 (no damping shift) with HWHM = Gamma/2, and
    carries the same area as the +omega0 half of the DHO. It coincides with the
    DHO for Gamma << omega0 and deviates as damping grows.
    """
    return (gamma_damp / (4.0 * np.pi)) / ((omega - omega0) ** 2 + (gamma_damp / 2.0) ** 2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omega0", type=float, default=6.0, help="Bare frequency omega0 (default: 6.0)")
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.3, 0.8, 1.6, 3.0, 5.0],
        help="Damping values Gamma to plot (default: 0.3 0.8 1.6 3.0 5.0)",
    )
    parser.add_argument("--omega-max", type=float, default=15.0, help="Max omega on the x-axis (default: 15.0)")
    parser.add_argument("--num", type=int, default=2000, help="Number of omega samples (default: 2000)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dho_lineshape.png",
        help="Output image path (default: dho_lineshape.png next to this script)",
    )
    parser.add_argument(
        "--mark-omega-q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark omega_q = sqrt(omega0^2 - (Gamma/2)^2) for each curve (default: on)",
    )
    parser.add_argument(
        "--lorentzian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the weak-damping Lorentzian approximation (dashed) for each Gamma (default: on)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    omega = np.linspace(0.0, args.omega_max, args.num)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(args.gammas)))

    for gamma_damp, color in zip(args.gammas, colors):
        s = dho_lineshape(omega, args.omega0, gamma_damp)
        ax.plot(omega, s, color=color, lw=2.0, label=rf"$\Gamma={gamma_damp:g}$")

        if args.lorentzian:
            s_lor = lorentzian_approx(omega, args.omega0, gamma_damp)
            ax.plot(omega, s_lor, color=color, lw=1.5, ls="--", alpha=0.9)

        if args.mark_omega_q:
            arg = args.omega0**2 - (gamma_damp / 2.0) ** 2
            if arg > 0.0:  # underdamped: a real oscillation frequency exists
                omega_q = np.sqrt(arg)
                ax.axvline(omega_q, color=color, ls=":", lw=1.0, alpha=0.7)

    ax.axvline(args.omega0, color="k", ls="--", lw=1.0, alpha=0.5, label=rf"$\omega_0={args.omega0:g}$")

    if args.lorentzian:
        # Style-only legend entries clarifying solid vs. dashed.
        ax.plot([], [], color="0.3", lw=2.0, ls="-", label="DHO (exact)")
        ax.plot([], [], color="0.3", lw=1.5, ls="--", label="Lorentzian approx.")

    ax.set_xlabel(r"$\omega$", fontsize=14)
    ax.set_ylabel(r"$S(q,\omega)$  (unit area)", fontsize=14)
    ax.set_title("DHO lineshape vs. damping $\\Gamma$", fontsize=15)
    ax.set_xlim(0.0, args.omega_max)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"[INFO] omega0        : {args.omega0}")
    print(f"[INFO] Gamma values  : {args.gammas}")
    print(f"[INFO] Output image  : {args.output}")


if __name__ == "__main__":
    main()
