#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from sqw_args import build_path_arg_parser, default_path_output_for_method
from sqw_core import (
    THZ_TO_MEV,
    SpinStructureFactorCalculator,
    cart_to_frac_q,
    frac_to_cart_q,
    load_q_file,
    primitive_lattice_from_supercell,
    q_path_distance,
    reciprocal_lattice_from_real,
    resolve_mpi_comm,
)


def axis_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("axis_edges expects a non-empty 1D array")
    if values.size == 1:
        delta = 0.5
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)

    edges = np.empty(values.size + 1, dtype=float)
    midpoints = 0.5 * (values[1:] + values[:-1])
    edges[1:-1] = midpoints
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def plot_sqw(
    q_vectors: np.ndarray,
    freq_thz: np.ndarray,
    sqw: np.ndarray,
    outfile: str | None = None,
    max_freq_thz: float | None = None,
    cbar_min: float | None = None,
    cbar_max: float | None = None,
    use_meV: bool = False,
    title: str | None = None,
    q_node_indices: np.ndarray | None = None,
    q_node_labels: List[str] | None = None,
    cmap: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_freq_thz is not None:
        mask = freq_thz <= max_freq_thz
        freq_plot = freq_thz[mask]
        sqw_plot = sqw[:, mask]
    else:
        freq_plot = freq_thz
        sqw_plot = sqw

    q_path = q_path_distance(q_vectors)
    y = freq_plot * THZ_TO_MEV if use_meV else freq_plot
    ylabel = "Energy (meV)" if use_meV else "Frequency (THz)"

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    q_edges = axis_edges(q_path)
    y_edges = axis_edges(y)
    mesh_kwargs = {
        "shading": "auto",
        "vmin": cbar_min,
        "vmax": cbar_max,
    }
    if cmap is not None:
        mesh_kwargs["cmap"] = cmap
    im = ax.pcolormesh(q_edges, y_edges, sqw_plot.T, **mesh_kwargs)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Intensity (arb. units)")

    ax.set_xlabel("q-path distance")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if q_node_indices is not None:
        tick_pos = q_path[q_node_indices]
        for xpos in tick_pos:
            ax.axvline(x=xpos, color="w", linestyle="--", linewidth=0.6)
        if q_node_labels is not None:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(q_node_labels)

    fig.tight_layout()
    prefix = "sqw_results" if outfile is None else str(Path(outfile).with_suffix(""))
    png_file = f"{prefix}.png"
    eps_file = f"{prefix}.eps"
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(eps_file, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {png_file}")
    print(f"[INFO] Plot saved to: {eps_file}")


def build_arg_parser(
    default_method: str | None = None,
    prog: str | None = None,
    description: str | None = None,
) -> argparse.ArgumentParser:
    return build_path_arg_parser(
        default_method=default_method,
        prog=prog,
        description=description,
    )


def _resolve_method(args: argparse.Namespace, default_method: str | None) -> str:
    return default_method if default_method is not None else args.method


def _result_title(method: str, components: str, projection: str) -> str:
    if projection == "cartesian":
        detail = f"components={components}"
    else:
        detail = f"projection={projection}"
    if method == "corr":
        return f"S(q,ω) from C(q,τ), {detail}"
    return f"S(q,ω), {detail}"


def _plot_cmap(method: str) -> str | None:
    if method == "corr":
        return "YlOrRd"
    return None


def main(
    default_method: str | None = None,
    prog: str | None = None,
    description: str | None = None,
) -> None:
    args = build_arg_parser(
        default_method=default_method,
        prog=prog,
        description=description,
    ).parse_args()
    method = _resolve_method(args, default_method)
    freq_mode = getattr(args, "freq_mode", "fft")
    freq_min_thz = getattr(args, "freq_min_thz", 0.0)
    freq_max_thz = getattr(args, "freq_max_thz", None)
    freq_step_thz = getattr(args, "freq_step_thz", None)

    if method != "corr" and getattr(args, "save_corr_plus", False):
        raise ValueError("--save-corr-plus is only available when method='corr'")
    if method == "corr" and freq_mode != "fft":
        raise ValueError("--freq-mode direct is only available when method='periodogram'")
    if freq_mode == "direct":
        if freq_max_thz is None:
            raise ValueError("--freq-max-thz is required when --freq-mode direct")
        if freq_step_thz is None:
            raise ValueError("--freq-step-thz is required when --freq-mode direct")

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
        print("[INFO] Primitive-cell reciprocal lattice vectors used to interpret qfile (rows):")
        for idx, vec in enumerate(primitive_reciprocal, start=1):
            print(f"[INFO]   b{idx}^prim = ({vec[0]: .10f}, {vec[1]: .10f}, {vec[2]: .10f})")

    q_frac_prim, q_node_indices, q_node_labels = load_q_file(args.qfile, args.points_per_segment)
    q_cart = frac_to_cart_q(q_frac_prim, primitive_reciprocal)
    q_frac_engine = cart_to_frac_q(q_cart, calc.reciprocal_lattice)

    if is_root:
        print(f"[INFO] Frames: {calc.spins.shape[0]}")
        print(f"[INFO] Atoms : {calc.positions.shape[1]}")
        if args.spin_threshold > 0.0:
            print(f"[INFO] Field threshold filter enabled: {args.spin_threshold:g}")
        print(f"[INFO] Field columns: {tuple(args.field_columns)}")
        print(f"[INFO] Projection mode: {args.projection}")
        print(f"[INFO] q-points: {q_frac_engine.shape[0]}")
        if calc.timesteps.size >= 2:
            dstep = np.diff(calc.timesteps)
            if not np.all(dstep == dstep[0]):
                print("[WARN] Dump timestep spacing is not uniform.")
            else:
                print(f"[INFO] Dump timestep interval = {dstep[0]} (raw MD step units)")
        print(f"[INFO] Translation repeats used in sum: {tuple(args.translation_repeats)}")
        print(f"[INFO] Using dt_fs = {args.dt_fs} fs between saved frames")
        if method == "corr":
            print("[INFO] Computing S(q,w) from C(q,τ)...")
        else:
            print("[INFO] Computing S(q,w) from s(q,t)...")
            print(f"[INFO] Frequency sampling mode: {freq_mode}")
            if freq_mode == "direct":
                print(
                    f"[INFO] Direct FT grid (THz): start={freq_min_thz:g}, "
                    f"stop={freq_max_thz:g}, step={freq_step_thz:g}"
                )
        if args.progress:
            print(f"[INFO] Progress reporting enabled: target ~{args.progress_reports} updates")

    if method == "corr":
        result = calc.compute_correlation_spectrum(
            q_frac=q_frac_engine,
            dt_fs=args.dt_fs,
            components=args.components,
            projection=args.projection,
            use_instantaneous_pos=args.use_instantaneous_pos,
            translation_repeats=tuple(args.translation_repeats),
            subtract_mean=(not args.no_subtract_mean),
            window=args.window,
            corr_norm=args.corr_norm,
            return_corr_plus=args.save_corr_plus,
            progress=args.progress,
            progress_reports=args.progress_reports,
            mpi_comm=mpi_comm,
            q_node_indices=q_node_indices,
            q_node_labels=q_node_labels,
        )
    else:
        result = calc.compute_periodogram(
            q_frac=q_frac_engine,
            dt_fs=args.dt_fs,
            components=args.components,
            projection=args.projection,
            use_instantaneous_pos=args.use_instantaneous_pos,
            translation_repeats=tuple(args.translation_repeats),
            subtract_mean=(not args.no_subtract_mean),
            window=args.window,
            freq_mode=freq_mode,
            freq_min_thz=freq_min_thz,
            freq_max_thz=freq_max_thz,
            freq_step_thz=freq_step_thz,
            progress=args.progress,
            progress_reports=args.progress_reports,
            mpi_comm=mpi_comm,
            q_node_indices=q_node_indices,
            q_node_labels=q_node_labels,
        )

    if not is_root:
        return

    output_file = args.output if args.output is not None else default_path_output_for_method(method)
    if args.save_npz:
        np.savez(output_file, **result.to_npz_dict())
        print(f"[INFO] Results saved to: {output_file}")
    else:
        print("[INFO] --save-npz is not set; skipping .npz output.")

    if args.plot:
        plot_sqw(
            q_vectors=result.q_vectors,
            freq_thz=result.freq_thz,
            sqw=result.sqw,
            outfile=args.plot_file,
            max_freq_thz=args.max_freq_thz,
            cbar_min=args.cbar_min,
            cbar_max=args.cbar_max,
            use_meV=args.mev,
            title=_result_title(method, args.components, args.projection),
            q_node_indices=q_node_indices,
            q_node_labels=q_node_labels,
            cmap=_plot_cmap(method),
        )


if __name__ == "__main__":
    main()
