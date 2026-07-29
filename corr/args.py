#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface for the correlation-route S(q,w).

Only the options this route actually uses are defined here. In particular there
is no --projection / --components pair any more: every output is requested
through --component, whose weight-matrix model subsumes both. See channels.py
for the token grammar and MIGRATION.md for the old-to-new mapping.
"""

from __future__ import annotations

import argparse

DEFAULT_COMPONENT = ["1+5+9"]


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Compute a q-resolved dynamic structure factor S(q,w) from "
            "time-correlation functions of a three-component field."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- input -------------------------------------------------------
    parser.add_argument("dumpfile", help="LAMMPS-like dump file (or .npz cache)")
    parser.add_argument("qfile", help="Text file listing the q-path nodes")
    parser.add_argument(
        "--field-columns",
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help="Dump column names holding the three field components",
    )
    parser.add_argument(
        "--supercell", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=(20, 20, 1),
        help="Supercell expansion relative to the primitive cell",
    )
    parser.add_argument("--dt-fs", type=float, default=1.0,
                        help="Time between consecutive saved frames, in fs")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="Storage dtype for the parsed trajectory")
    parser.add_argument("--frame-start", type=int, default=None, help="First frame, inclusive")
    parser.add_argument("--frame-stop", type=int, default=None, help="Stop before this frame")
    parser.add_argument("--frame-step", type=int, default=None, help="Keep one frame every N")
    parser.add_argument("--spin-threshold", type=float, default=0.0,
                        help="Drop atoms whose max_t |field| <= this value")
    parser.add_argument("--cache-binary", action=argparse.BooleanOptionalAction, default=True,
                        help="Cache the parsed trajectory next to the dump")
    parser.add_argument("--cache-file", default=None, help="Explicit cache file path")

    # ---- q-space -----------------------------------------------------
    parser.add_argument("--points-per-segment", type=int, default=101,
                        help="Interpolated q-points per path segment")
    parser.add_argument(
        "--bz-folded", type=int, nargs=3, metavar=("FX", "FY", "FZ"), default=(1, 1, 1),
        help="Interpret the q file in a Brillouin zone folded by these factors; "
             "unfolded branches are reduced by max intensity at each frequency",
    )
    parser.add_argument(
        "--translation-repeats", type=int, nargs=3, metavar=("N1", "N2", "N3"), default=(1, 1, 1),
        help="Finite repeats of the loaded cell in the structure-factor sum",
    )
    parser.add_argument("--use-instantaneous-pos", action="store_true",
                        help="Use per-frame positions instead of freezing the first frame")

    # ---- weighting ---------------------------------------------------
    parser.add_argument("--mass-weight", action="store_true",
                        help="Scale each atom's field by sqrt(mass) (phonon SED convention)")
    parser.add_argument("--masses", type=float, nargs="+", default=None,
                        help="One mass per LAMMPS atom type, in type order")

    # ---- channels ----------------------------------------------------
    parser.add_argument(
        "--component", nargs="+", metavar="C", default=DEFAULT_COMPONENT,
        help="Output channels. Separate tokens are separate outputs; terms joined "
             "by '+' are summed into one. Terms: 1..9, xx..zz, x/y/z, L, T. "
             "Example: --component 1+2 3 gives S^xx+S^xy and S^xz. "
             "The default 1+5+9 is the trace S^xx+S^yy+S^zz",
    )

    # ---- time-correlation --------------------------------------------
    parser.add_argument("--no-subtract-mean", action="store_true",
                        help="Keep the time average (leaves the elastic line in)")
    parser.add_argument("--window", choices=["hann", "none"], default="hann",
                        help="Data-domain taper applied to s(q,t) before correlating. "
                             "Suppresses spectral leakage from the finite record; "
                             "'none' leaves a rectangular record, which is exact only "
                             "when every mode completes a whole number of cycles")
    parser.add_argument("--corr-norm", choices=["biased", "unbiased"], default="biased",
                        help="Correlation denominator: Nt, or Nt-tau")
    parser.add_argument("--save-corr-plus", action="store_true",
                        help="Also store the one-sided C^{ab}(q,tau) in the npz")

    # ---- output ------------------------------------------------------
    parser.add_argument("--save-npz", action="store_true", help="Write results to --output")
    parser.add_argument("--output", default="sqw_corr_results.npz", help="Output .npz path")
    parser.add_argument("--plot", action="store_true", help="Render an intensity map")
    parser.add_argument("--plot-file", default="sqw_corr.png", help="Plot output path")
    parser.add_argument("--mev", action="store_true", help="Label the energy axis in meV")
    parser.add_argument("--max-freq-thz", type=float, default=None, help="Frequency cutoff for the plot")
    parser.add_argument("--cbar-min", type=float, default=None, help="Colorbar minimum")
    parser.add_argument("--cbar-max", type=float, default=None, help="Colorbar maximum")

    # ---- runtime -----------------------------------------------------
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Print q-loop progress")
    parser.add_argument("--progress-reports", type=int, default=20,
                        help="Approximate number of progress updates")

    return parser


__all__ = ["build_parser", "DEFAULT_COMPONENT"]
