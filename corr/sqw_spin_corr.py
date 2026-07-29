#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S(q,w) from time-correlation functions of a three-component field.

The whole calculation is the seven lines of `sqw_at_one_q` below, one per stage
of the derivation:

    S_j^a(t)  ->  s^a(q,t)  ->  C^{ab}(q,tau)  ->  S^{ab}(q,w)  ->  channels

Parallelism does not appear in that chain: the kernel is serial and handles a
single q-point, and `map_over_q` is the only thing that knows about MPI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from args import build_parser
from channels import channel_value, parse_channels
from correlate import (
    apply_data_window,
    hermitian_extend,
    subtract_time_mean,
    time_correlation_tensor,
    window_array,
)
from driver import map_over_q
from fields import spatial_fourier_transform, validate_translation_repeats
from geometry import load_q_file
from mpi import resolve_mpi_comm
from plot import plot_sqw
from result import SQWResult
from spectrum import frequency_grid, tensor_spectrum
from trajectory import load_trajectory


# ----------------------------------------------------------------------
# The calculation
# ----------------------------------------------------------------------
def sqw_at_one_q(traj, q, channels, window_vec, pos_mask, opts):
    """Everything that happens at a single q-point."""
    s_qt = spatial_fourier_transform(                    # s^a(q,t)
        traj, q,
        use_instantaneous_pos=opts["use_instantaneous_pos"],
        translation_repeats=opts["translation_repeats"],
    )
    if opts["subtract_mean"]:
        s_qt = subtract_time_mean(s_qt)                  # drop the elastic line
    s_qt = apply_data_window(s_qt, window_vec)           # taper the record ends
    c_tau = time_correlation_tensor(                     # C^{ab}(q,tau), tau >= 0
        s_qt, corr_norm=opts["corr_norm"],
    )
    c_full = hermitian_extend(c_tau)                     # tau in [-(Nt-1), Nt-1]
    s_ab = tensor_spectrum(c_full, pos_mask)             # S^{ab}(q,w >= 0)

    q_norm = float(np.linalg.norm(q))
    q_hat = None if q_norm <= 1.0e-15 else q / q_norm
    spectra = np.stack([channel_value(s_ab, ch, q_hat) for ch in channels])

    return (spectra, c_tau) if opts["save_corr_plus"] else spectra


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main(argv=None) -> None:
    args = build_parser(prog="sqw_spin_corr.py").parse_args(argv)

    mpi_comm, rank, _ = resolve_mpi_comm()
    is_root = rank == 0

    channels = parse_channels(args.component)
    repeats = validate_translation_repeats(args.translation_repeats)

    # ---- input ------------------------------------------------------
    traj = load_trajectory(
        args.dumpfile,
        supercell=np.asarray(args.supercell, dtype=int),
        use_instantaneous_pos=args.use_instantaneous_pos,
        dtype=np.dtype(args.dtype),
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        frame_step=args.frame_step,
        cache_binary=args.cache_binary,
        cache_file=args.cache_file,
        spin_threshold=args.spin_threshold,
        field_columns=args.field_columns,
        progress=args.progress and is_root,
        progress_reports=args.progress_reports,
    )
    if args.mass_weight:
        if not args.masses:
            raise ValueError("--mass-weight requires --masses")
        traj.set_mass_weights(args.masses)
    elif args.masses and is_root:
        print("[INFO] --masses given without --mass-weight; ignored (magnon convention).")

    if is_root:
        traj.print_lattice_info()
        print(f"[INFO] Frames: {traj.n_frames}   Atoms: {traj.n_atoms}")
        print(f"[INFO] Field columns: {traj.field_columns}")
        print(f"[INFO] Channels: {[c.label for c in channels]}")
        if args.spin_threshold > 0.0:
            print(f"[INFO] Field threshold: {args.spin_threshold:g}")

    # ---- q-path -----------------------------------------------------
    q_frac, q_node_indices, q_node_labels = load_q_file(
        args.qfile, args.points_per_segment
    )
    q_cart = traj.q_frac_to_cart(q_frac)
    if is_root:
        print(f"[INFO] q-points: {q_cart.shape[0]}")

    # ---- spectra ----------------------------------------------------
    n_corr = 2 * traj.n_frames - 1
    window_vec = window_array(traj.n_frames, args.window)   # data window: length Nt
    freq_thz, pos_mask = frequency_grid(args.dt_fs, n_corr)

    opts = {
        "use_instantaneous_pos": args.use_instantaneous_pos,
        "translation_repeats": repeats,
        "subtract_mean": not args.no_subtract_mean,
        "corr_norm": args.corr_norm,
        "save_corr_plus": args.save_corr_plus,
    }

    if is_root:
        print("[INFO] Computing S(q,w) from C(q,tau)...")

    result = map_over_q(
        q_cart,
        lambda iq, qv: sqw_at_one_q(traj, qv, channels, window_vec, pos_mask, opts),
        out_width=freq_thz.size,
        n_channels=len(channels),
        aux_shape=(traj.n_frames, 3, 3) if args.save_corr_plus else None,
        mpi_comm=mpi_comm,
        progress=args.progress,
        progress_reports=args.progress_reports,
        label="S(q,w) from C(q,tau)",
    )
    corr_plus = None
    if args.save_corr_plus:
        sqw_all, corr_plus = result
    else:
        sqw_all = result

    if not is_root:
        return

    # ---- output -----------------------------------------------------
    multi = len(channels) > 1
    out_path = Path(args.output)
    plot_path = Path(args.plot_file)

    for ic, channel in enumerate(channels):
        tag = _safe_tag(channel.label)
        result_c = SQWResult(
            timesteps=traj.timesteps,
            q_frac=q_frac,
            q_vectors=q_cart,
            freq_thz=freq_thz,
            sqw=sqw_all[ic],
            lattice=traj.lattice,
            reciprocal_lattice=traj.reciprocal,
            dt_fs=args.dt_fs,
            components=channel.label,
            field_columns=np.asarray(traj.field_columns),
            translation_repeats=repeats,
            q_node_indices=q_node_indices,
            q_node_labels=q_node_labels,
            corr_plus=corr_plus if args.save_corr_plus else None,
            corr_norm=args.corr_norm,
        )

        if args.save_npz:
            path = _suffixed(out_path, tag) if multi else out_path
            np.savez(path, **result_c.to_npz_dict())
            print(f"[INFO] [{channel.label}] saved: {path}")

        if args.plot:
            path = _suffixed(plot_path, tag) if multi else plot_path
            plot_sqw(
                q_vectors=q_cart,
                freq_thz=freq_thz,
                sqw=sqw_all[ic],
                outfile=str(path),
                max_freq_thz=args.max_freq_thz,
                cbar_min=args.cbar_min,
                cbar_max=args.cbar_max,
                use_meV=args.mev,
                title=f"S(q,w)  component={channel.label}",
                q_node_indices=q_node_indices,
                q_node_labels=q_node_labels,
            )
            print(f"[INFO] [{channel.label}] plot: {path}")


def _safe_tag(label: str) -> str:
    """'+' is legal in filenames but awkward in shells; keep it readable."""
    return label.replace("+", "p")


def _suffixed(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


if __name__ == "__main__":
    main()
