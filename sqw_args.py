#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse


def default_path_output_for_method(method: str) -> str:
    if method == "corr":
        return "sqw_corr_results.npz"
    return "sqw_results.npz"


def _default_path_description(default_method: str | None) -> str:
    if default_method == "periodogram":
        return "Compute a q-resolved phonon spectrum from a LAMMPS-like dump file."
    if default_method == "corr":
        return "Compute a q-resolved phonon spectrum from time-correlation functions of a three-component field."
    return "Compute a q-resolved phonon spectrum along a q-path using either a periodogram route or correlation-based spectrum estimation."


def _add_dumpfile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dumpfile", help="Input dump file")


def _add_qfile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "qfile",
        help="q file in primitive-cell fractional reciprocal coordinates; either direct q points or labeled path nodes",
    )


def _add_method_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=["periodogram", "corr"],
        default="periodogram",
        help="Spectrum estimator: periodogram from s(q,t) or correlation-based FFT (default: periodogram)",
    )


def _add_supercell_arg(parser: argparse.ArgumentParser, help_text: str | None = None) -> None:
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(20, 20, 1),
        help=(
            "Supercell expansion relative to the primitive cell, e.g. --supercell 20 20 1"
            if help_text is None
            else help_text
        ),
    )


def _add_dt_fs_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dt-fs",
        type=float,
        default=1.0,
        help="Physical time interval between consecutive saved frames in fs (default: 1.0)",
    )


def _add_translation_repeats_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--translation-repeats",
        type=int,
        nargs=3,
        metavar=("NR1", "NR2", "NR3"),
        default=(1, 1, 1),
        help="Finite repeats of the loaded simulation cell in the structure-factor sum (default: 1 1 1)",
    )


def _add_dtype_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Internal numeric dtype for parsed trajectories (default: float32)",
    )


def _add_frame_selection_args(
    parser: argparse.ArgumentParser,
    start_help: str = "First frame index to keep, inclusive (default: 0)",
    stop_help: str = "Stop before this frame index, exclusive (default: keep through last frame)",
    step_help: str = "Keep one frame every N selected frames (default: 1)",
) -> None:
    parser.add_argument(
        "--frame-start",
        type=int,
        default=None,
        help=start_help,
    )
    parser.add_argument(
        "--frame-stop",
        type=int,
        default=None,
        help=stop_help,
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help=step_help,
    )


def _add_components_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--components",
        type=str,
        default="xyz",
        help=help_text,
    )


def _add_field_columns_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--field-columns",
        nargs=3,
        metavar=("COLX", "COLY", "COLZ"),
        default=("vx", "vy", "vz"),
        help="Dump columns used as the three-component field, e.g. vx vy vz (default: vx vy vz)",
    )


def _add_projection_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--projection",
        choices=["cartesian", "longitudinal", "transverse"],
        default="cartesian",
        help="q-space projection of the three-component field (default: cartesian)",
    )


def _add_use_instantaneous_pos_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--use-instantaneous-pos",
        action="store_true",
        help=help_text,
    )


def _add_no_subtract_mean_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--no-subtract-mean",
        action="store_true",
        help=help_text,
    )


def _add_window_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--window",
        choices=["hann", "none"],
        default="hann",
        help=help_text,
    )


def _add_frequency_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--freq-mode",
        choices=["fft", "direct"],
        default="fft",
        help="Time-frequency transform for periodogram route: FFT or direct FT (default: fft)",
    )
    parser.add_argument(
        "--freq-min-thz",
        type=float,
        default=0.0,
        help="Minimum sampled frequency for --freq-mode direct in THz (default: 0.0)",
    )
    parser.add_argument(
        "--freq-max-thz",
        type=float,
        default=None,
        help="Maximum sampled frequency for --freq-mode direct in THz (required for direct mode)",
    )
    parser.add_argument(
        "--freq-step-thz",
        type=float,
        default=None,
        help="Frequency spacing for --freq-mode direct in THz (required for direct mode)",
    )


def _add_progress_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print root-rank q-loop progress during the calculation (default: enabled)",
    )
    parser.add_argument(
        "--progress-reports",
        type=int,
        default=20,
        help="Approximate number of q-loop progress updates to print (default: 20)",
    )


def _add_cache_args(
    parser: argparse.ArgumentParser,
    cache_help: str | None = None,
    cache_file_help: str | None = None,
) -> None:
    parser.add_argument(
        "--cache-binary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-read/write binary cache for dump parsing "
            "(default: enabled; cache file is <dumpfile>.sqwcache.npz)"
            if cache_help is None
            else cache_help
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help=(
            "Custom binary cache path (default: <dumpfile>.sqwcache.npz)"
            if cache_file_help is None
            else cache_file_help
        ),
    )


def _add_spin_threshold_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spin-threshold",
        type=float,
        default=0.0,
        help="Ignore atoms with max_t |field_i(t)| <= threshold (default: 0.0)",
    )


def _add_corr_args(
    parser: argparse.ArgumentParser,
    corr_norm_help: str,
    include_save_corr_plus: bool,
) -> None:
    parser.add_argument(
        "--corr-norm",
        choices=["biased", "unbiased"],
        default="biased",
        help=corr_norm_help,
    )
    if include_save_corr_plus:
        parser.add_argument(
            "--save-corr-plus",
            action="store_true",
            help="Also save corr_plus(q,t) into npz (can be memory-heavy)",
        )


def _add_save_npz_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help=help_text,
    )


def _add_output_arg(
    parser: argparse.ArgumentParser,
    default: str | None,
    help_text: str,
) -> None:
    parser.add_argument(
        "--output",
        type=str,
        default=default,
        help=help_text,
    )


def _add_plot_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--plot",
        action="store_true",
        help=help_text,
    )


def _add_plot_file_arg(parser: argparse.ArgumentParser, default: str | None, help_text: str) -> None:
    parser.add_argument(
        "--plot-file",
        type=str,
        default=default,
        help=help_text,
    )


def _add_mev_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mev",
        action="store_true",
        help="Plot energy axis in meV instead of THz",
    )


def _add_qpath_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-freq-thz",
        type=float,
        default=None,
        help="Maximum frequency to show in plot (THz)",
    )
    parser.add_argument(
        "--cbar-min",
        type=float,
        default=None,
        help="Minimum colorbar value for S(q,w) plot (default: data minimum)",
    )
    parser.add_argument(
        "--cbar-max",
        type=float,
        default=None,
        help="Maximum colorbar value for S(q,w) plot (default: data maximum)",
    )
    _add_mev_arg(parser)


def _add_bz_grid_args(
    parser: argparse.ArgumentParser,
    nh_help: str,
    nk_help: str,
    frac_limit_help: str,
) -> None:
    parser.add_argument("--nh", type=int, default=121, help=nh_help)
    parser.add_argument("--nk", type=int, default=121, help=nk_help)
    parser.add_argument(
        "--frac-limit",
        type=float,
        default=1.5,
        help=frac_limit_help,
    )


def _add_freq_range_args(
    parser: argparse.ArgumentParser,
    min_help: str,
    max_help: str,
) -> None:
    parser.add_argument("--freq-min-thz", type=float, default=None, help=min_help)
    parser.add_argument("--freq-max-thz", type=float, default=None, help=max_help)


def build_path_arg_parser(
    default_method: str | None = None,
    prog: str | None = None,
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=_default_path_description(default_method) if description is None else description,
    )
    _add_dumpfile_arg(parser)
    _add_qfile_arg(parser)
    if default_method is None:
        _add_method_arg(parser)
    parser.add_argument(
        "--points-per-segment",
        type=int,
        default=41,
        help="Interpolation points per labeled path segment (default: 41)",
    )
    _add_supercell_arg(parser)
    _add_dt_fs_arg(parser)
    _add_translation_repeats_arg(parser)
    _add_dtype_arg(parser)
    _add_frame_selection_args(parser)
    _add_field_columns_arg(parser)
    _add_projection_arg(parser)
    _add_components_arg(
        parser,
        "Cartesian components to include when --projection cartesian: xyz/all, xy, x, y, z (default: xyz)",
    )
    _add_use_instantaneous_pos_arg(
        parser,
        "Use instantaneous positions in exp(i q·r_j(t)); default uses first-frame positions",
    )

    if default_method == "periodogram":
        no_subtract_mean_help = "Do not subtract time average before FFT/direct FT"
        window_help = "Window for FFT or direct FT (default: hann)"
    elif default_method == "corr":
        no_subtract_mean_help = "Do not subtract the time average before building the correlation function"
        window_help = "Window applied to the correlation function before FFT (default: hann)"
    else:
        no_subtract_mean_help = "Do not subtract the time average before periodogram FFT/direct FT or correlation"
        window_help = "Window for periodogram FFT/direct FT or correlation FFT (default: hann)"

    _add_no_subtract_mean_arg(parser, no_subtract_mean_help)
    _add_window_arg(parser, window_help)
    if default_method != "corr":
        _add_frequency_sampling_args(parser)
    _add_progress_args(parser)
    _add_cache_args(parser)
    _add_spin_threshold_arg(parser)
    _add_save_npz_arg(parser, "Write result arrays to --output .npz (default: disabled)")

    include_corr_options = default_method is None or default_method == "corr"
    if include_corr_options:
        _add_corr_args(
            parser,
            corr_norm_help=(
                "Correlation normalization: fixed 1/Nt or lag-dependent 1/(Nt-tau) (default: biased)"
            ),
            include_save_corr_plus=True,
        )

    if default_method is None:
        output_help = (
            "Output .npz file path used when --save-npz is set "
            "(default: sqw_results.npz for periodogram, sqw_corr_results.npz for corr)"
        )
        output_default = None
    else:
        output_help = (
            f"Output .npz file path used when --save-npz is set "
            f"(default: {default_path_output_for_method(default_method)})"
        )
        output_default = default_path_output_for_method(default_method)
    _add_output_arg(parser, output_default, output_help)
    _add_plot_arg(parser, "Plot the S(q,w) intensity map")
    _add_plot_file_arg(parser, None, "Plot filename prefix; default is sqw_results")
    _add_qpath_plot_args(parser)
    return parser


def build_bz_map_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute S(q,w) on a 2D first-Brillouin-zone grid and plot the "
            "omega-maximum and omega-average intensity maps."
        )
    )
    _add_dumpfile_arg(parser)
    _add_method_arg(parser)
    _add_supercell_arg(parser)
    _add_dt_fs_arg(parser)
    _add_translation_repeats_arg(parser)
    _add_dtype_arg(parser)
    _add_frame_selection_args(parser)
    _add_field_columns_arg(parser)
    _add_projection_arg(parser)
    _add_components_arg(
        parser,
        "Cartesian components to include when --projection cartesian: xyz/all, xy, x, y, z (default: xyz)",
    )
    _add_use_instantaneous_pos_arg(
        parser,
        "Use instantaneous positions in exp(-i q·r_j(t)); default uses first-frame positions",
    )
    _add_no_subtract_mean_arg(parser, "Do not subtract the time average before FFT/correlation")
    _add_window_arg(parser, "Window for FFT or correlation FFT (default: hann)")
    _add_progress_args(parser)
    parser.add_argument(
        "--plot-mode",
        choices=["gouraud", "flat", "scatter"],
        default="gouraud",
        help="2D BZ plot rendering mode (default: gouraud)",
    )
    _add_corr_args(
        parser,
        corr_norm_help="Correlation normalization used when --method corr (default: biased)",
        include_save_corr_plus=False,
    )
    _add_cache_args(parser)
    _add_spin_threshold_arg(parser)
    _add_bz_grid_args(
        parser,
        nh_help="Number of q-grid points along primitive b1 direction (default: 121)",
        nk_help="Number of q-grid points along primitive b2 direction (default: 121)",
        frac_limit_help="Half-width of the sampled rectangle in primitive reciprocal fractional units (default: 1.5)",
    )
    _add_freq_range_args(
        parser,
        min_help="Minimum frequency included in the max/mean statistics (default: 0)",
        max_help="Maximum frequency included in the max/mean statistics (default: all positive frequencies)",
    )
    _add_save_npz_arg(parser, "Write BZ grid, masks, and summary maps to --output .npz")
    _add_output_arg(
        parser,
        "sqw_bz_map.npz",
        "Output .npz file path used when --save-npz is set (default: sqw_bz_map.npz)",
    )
    _add_plot_arg(parser, "Plot the omega-maximum and omega-average maps on the 2D BZ grid")
    _add_plot_file_arg(parser, "sqw_bz_map", "Plot filename prefix (default: sqw_bz_map)")
    return parser


def build_magnon_dos_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute magnon DOS using two routes: "
            "(1) q-averaged S(q,w) on 2D first BZ, "
            "(2) Fourier transform of spin autocorrelation."
        )
    )
    _add_dumpfile_arg(parser)
    parser.add_argument(
        "--dos-method",
        choices=["qavg", "autocorr", "both"],
        default="both",
        help="Which DOS definition to run (default: both)",
    )
    parser.add_argument(
        "--sqw-method",
        choices=["periodogram", "corr"],
        default="periodogram",
        help="S(q,w) estimator used by qavg route (default: periodogram)",
    )
    _add_supercell_arg(
        parser,
        help_text="Supercell expansion relative to primitive cell, e.g. --supercell 20 20 1",
    )
    _add_dt_fs_arg(parser)
    _add_translation_repeats_arg(parser)
    _add_dtype_arg(parser)
    _add_frame_selection_args(
        parser,
        start_help="First frame index kept, inclusive",
        stop_help="Stop before this frame index",
        step_help="Keep one frame every N selected frames",
    )
    _add_field_columns_arg(parser)
    _add_projection_arg(parser)
    _add_components_arg(
        parser,
        "Cartesian components to include when --projection cartesian: xyz/all, xy, x, y, z (default: xyz)",
    )
    _add_use_instantaneous_pos_arg(
        parser,
        "Use instantaneous positions in exp(-i q·r_j(t)); default uses first-frame positions",
    )
    _add_no_subtract_mean_arg(parser, "Do not subtract time average before FFT/correlation")
    _add_window_arg(parser, "Window for FFT/correlation spectrum (default: hann)")
    _add_corr_args(
        parser,
        corr_norm_help="Normalization used in correlation-based routes (default: biased)",
        include_save_corr_plus=False,
    )
    _add_cache_args(
        parser,
        cache_help="Auto-read/write binary cache for dump parsing (default: enabled)",
        cache_file_help="Custom cache path",
    )
    _add_spin_threshold_arg(parser)
    _add_bz_grid_args(
        parser,
        nh_help="q grid size along primitive b1 (default: 121)",
        nk_help="q grid size along primitive b2 (default: 121)",
        frac_limit_help="Half-width of sampled rectangle in primitive reciprocal fractional units",
    )
    _add_freq_range_args(
        parser,
        min_help="Minimum frequency to keep",
        max_help="Maximum frequency to keep",
    )
    parser.add_argument(
        "--smooth-sigma-thz",
        type=float,
        default=0.0,
        help="Gaussian smoothing width in THz (default: 0 means no smoothing)",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "max", "area"],
        default="max",
        help="DOS normalization mode (default: max)",
    )
    _add_save_npz_arg(parser, "Write output arrays to --output .npz")
    _add_output_arg(parser, "magnon_dos.npz", "Output npz path")
    _add_plot_arg(parser, "Plot DOS curves")
    parser.add_argument(
        "--plot-raw",
        action="store_true",
        help="Also plot raw DOS curves before smoothing/normalization",
    )
    _add_plot_file_arg(parser, "magnon_dos", "Plot filename prefix")
    _add_mev_arg(parser)
    return parser


__all__ = [
    "build_bz_map_arg_parser",
    "build_magnon_dos_arg_parser",
    "build_path_arg_parser",
    "default_path_output_for_method",
]
