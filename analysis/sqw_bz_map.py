#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqw_args import build_bz_map_arg_parser
from sqw_core import (
    SpinStructureFactorCalculator,
    cart_to_frac_q,
    frac_to_cart_q,
    generate_bz_points,
    primitive_lattice_from_supercell,
    reciprocal_lattice_from_real,
    resolve_mpi_comm,
)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_bz_map_arg_parser()


def summarize_over_frequency(
    freq_thz: np.ndarray,
    sqw: np.ndarray,
    freq_min_thz: float | None,
    freq_max_thz: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(freq_thz.shape, dtype=bool)
    if freq_min_thz is not None:
        mask &= freq_thz >= float(freq_min_thz)
    if freq_max_thz is not None:
        mask &= freq_thz <= float(freq_max_thz)
    if not np.any(mask):
        raise ValueError("No frequency points remain after applying freq_min_thz/freq_max_thz.")

    sqw_sel = sqw[:, mask]
    sqw_max = np.max(sqw_sel, axis=1)
    sqw_mean = np.mean(sqw_sel, axis=1)
    return mask, sqw_max, sqw_mean


def to_grid_map(values_inside: np.ndarray, inside_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, np.nan, dtype=float)
    grid.reshape(-1)[inside_mask] = values_inside
    return grid


def plot_bz_maps(
    qx_inside: np.ndarray,
    qy_inside: np.ndarray,
    max_values: np.ndarray,
    mean_values: np.ndarray,
    outfile: str,
    title: str,
    plot_mode: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
    triang = mtri.Triangulation(qx_inside, qy_inside)
    panels = [
        (max_values, "max_omega S(q,omega)", "magma"),
        (mean_values, "mean_omega S(q,omega)", "viridis"),
    ]

    for ax, (values, label, cmap) in zip(axes, panels):
        if plot_mode == "scatter":
            mesh = ax.scatter(qx_inside, qy_inside, c=values, cmap=cmap, s=18, edgecolors="none")
        else:
            mesh = ax.tripcolor(triang, values, shading=plot_mode, cmap=cmap)
            ax.triplot(triang, color="w", linewidth=0.15, alpha=0.2)
        ax.plot(0.0, 0.0, marker="o", markersize=4.5, color="cyan", markeredgecolor="k")
        ax.annotate(
            r"$\Gamma$",
            xy=(0.0, 0.0),
            xytext=(6, 6),
            textcoords="offset points",
            color="cyan",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(r"$q_x$ (1/Angstrom)")
        ax.set_ylabel(r"$q_y$ (1/Angstrom)")
        ax.set_aspect("equal")
        ax.set_title(label)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Intensity (arb. units)")

    fig.suptitle(title)
    prefix = str(Path(outfile).with_suffix(""))
    png_file = f"{prefix}.png"
    eps_file = f"{prefix}.eps"
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(eps_file, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {png_file}")
    print(f"[INFO] Plot saved to: {eps_file}")


def main() -> None:
    args = build_arg_parser().parse_args()
    mpi_comm, mpi_rank, mpi_size = resolve_mpi_comm()
    is_root = mpi_rank == 0

    if is_root:
        if mpi_comm is not None:
            print(f"[INFO] MPI q-point parallel enabled: ranks={mpi_size}")
        print("[INFO] Reading dump file...")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    calc = SpinStructureFactorCalculator.from_dump(
        args.dumpfile,
        keep_all_positions=args.use_instantaneous_pos,
        dtype=dtype,
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
    if is_root:
        calc.print_lattice_info()

    primitive_lattice = primitive_lattice_from_supercell(
        calc.lattice,
        np.asarray(args.supercell, dtype=int),
    )
    primitive_reciprocal = reciprocal_lattice_from_real(primitive_lattice)

    if is_root:
        print("[INFO] Primitive-cell lattice vectors inferred from --supercell (rows):")
        for idx, vec in enumerate(primitive_lattice, start=1):
            print(f"[INFO]   a{idx}^prim = ({vec[0]: .10f}, {vec[1]: .10f}, {vec[2]: .10f})")
        print("[INFO] Primitive-cell reciprocal lattice vectors used for 2D BZ grid (rows):")
        for idx, vec in enumerate(primitive_reciprocal, start=1):
            print(f"[INFO]   b{idx}^prim = ({vec[0]: .10f}, {vec[1]: .10f}, {vec[2]: .10f})")

    q_frac_prim_all, q_cart_all, inside_mask = generate_bz_points(
        reciprocal_lattice=primitive_reciprocal,
        nh=args.nh,
        nk=args.nk,
        frac_limit=args.frac_limit,
    )
    q_frac_prim_inside = q_frac_prim_all[inside_mask]
    q_frac_engine_inside = cart_to_frac_q(q_cart_all[inside_mask], calc.reciprocal_lattice)

    if is_root:
        print(f"[INFO] Frames: {calc.spins.shape[0]}")
        print(f"[INFO] Atoms : {calc.positions.shape[1]}")
        if args.spin_threshold > 0.0:
            print(f"[INFO] Field threshold filter enabled: {args.spin_threshold:g}")
        print(f"[INFO] Field columns: {tuple(args.field_columns)}")
        print(f"[INFO] Projection mode: {args.projection}")
        print(f"[INFO] Translation repeats used in sum: {tuple(args.translation_repeats)}")
        print(f"[INFO] 2D q-grid: {args.nh} x {args.nk}")
        print(f"[INFO] First-BZ q-points kept: {q_frac_prim_inside.shape[0]}")
        print(f"[INFO] Using dt_fs = {args.dt_fs} fs between saved frames")
        print(f"[INFO] Computing S(q,w) using method = {args.method}")
        if args.progress:
            print(f"[INFO] Progress reporting enabled: target ~{args.progress_reports} updates")

    if args.method == "periodogram":
        result = calc.compute_periodogram(
            q_frac=q_frac_engine_inside,
            dt_fs=args.dt_fs,
            components=args.components,
            projection=args.projection,
            use_instantaneous_pos=args.use_instantaneous_pos,
            translation_repeats=tuple(args.translation_repeats),
            subtract_mean=(not args.no_subtract_mean),
            window=args.window,
            progress=args.progress,
            progress_reports=args.progress_reports,
            mpi_comm=mpi_comm,
        )
    else:
        result = calc.compute_correlation_spectrum(
            q_frac=q_frac_engine_inside,
            dt_fs=args.dt_fs,
            components=args.components,
            projection=args.projection,
            use_instantaneous_pos=args.use_instantaneous_pos,
            translation_repeats=tuple(args.translation_repeats),
            subtract_mean=(not args.no_subtract_mean),
            window=args.window,
            corr_norm=args.corr_norm,
            return_corr_plus=False,
            progress=args.progress,
            progress_reports=args.progress_reports,
            mpi_comm=mpi_comm,
        )

    if not is_root:
        return

    freq_mask, sqw_max, sqw_mean = summarize_over_frequency(
        freq_thz=result.freq_thz,
        sqw=result.sqw,
        freq_min_thz=args.freq_min_thz,
        freq_max_thz=args.freq_max_thz,
    )

    inside_grid = inside_mask.reshape(args.nh, args.nk)
    max_grid = to_grid_map(sqw_max, inside_mask, (args.nh, args.nk))
    mean_grid = to_grid_map(sqw_mean, inside_mask, (args.nh, args.nk))
    q_cart_grid = q_cart_all.reshape(args.nh, args.nk, 3)
    qx_grid = q_cart_grid[:, :, 0]
    qy_grid = q_cart_grid[:, :, 1]
    qx_inside = q_cart_all[inside_mask, 0]
    qy_inside = q_cart_all[inside_mask, 1]

    if args.save_npz:
        np.savez(
            args.output,
            qx_grid=qx_grid,
            qy_grid=qy_grid,
            inside_mask=inside_grid,
            q_frac_prim_all=q_frac_prim_all,
            q_frac_prim_inside=q_frac_prim_inside,
            q_frac_engine_inside=q_frac_engine_inside,
            q_cart_inside=q_cart_all[inside_mask],
            qx_inside=qx_inside,
            qy_inside=qy_inside,
            freq_thz=result.freq_thz,
            freq_mask=freq_mask,
            sqw=result.sqw,
            sqw_max=sqw_max,
            sqw_mean=sqw_mean,
            sqw_max_grid=max_grid,
            sqw_mean_grid=mean_grid,
            primitive_lattice=primitive_lattice,
            primitive_reciprocal=primitive_reciprocal,
            supercell=np.asarray(args.supercell, dtype=int),
            translation_repeats=np.asarray(args.translation_repeats, dtype=int),
            method=np.asarray(args.method),
            components=np.asarray(args.components),
            field_columns=np.asarray(args.field_columns),
            projection=np.asarray(args.projection),
            dt_fs=np.asarray(args.dt_fs, dtype=float),
        )
        print(f"[INFO] Results saved to: {args.output}")

    if args.plot:
        freq_desc = []
        if args.freq_min_thz is not None:
            freq_desc.append(f"omega >= {args.freq_min_thz:g} THz")
        if args.freq_max_thz is not None:
            freq_desc.append(f"omega <= {args.freq_max_thz:g} THz")
        freq_text = ", ".join(freq_desc) if freq_desc else "all positive omega"
        plot_bz_maps(
            qx_inside=qx_inside,
            qy_inside=qy_inside,
            max_values=sqw_max,
            mean_values=sqw_mean,
            outfile=args.plot_file,
            title=f"2D BZ S(q,omega) summaries, method={args.method}, {freq_text}",
            plot_mode=args.plot_mode,
        )


if __name__ == "__main__":
    main()
