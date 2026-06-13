#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the frequency-integrated dynamic structure factor on a 2D BZ grid.

    S_int(q) = ∫ S(q,ω) dω   over [freq_min_thz, freq_max_thz]

Usage:
  python analysis/sqw_integrated.py dump.lammpstrj                  \
      --supercell 20 20 1 --dt-fs 2.0                                \
      --field-columns c_outsp[1] c_outsp[2] c_outsp[3]               \
      --components x --freq-min-thz 0 --freq-max-thz 40              \
      --nh 81 --nk 81 --plot --output result.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqw_args import build_integrated_arg_parser
from sqw_core import (
    SpinStructureFactorCalculator,
    cart_to_frac_q,
    generate_bz_points,
    primitive_lattice_from_supercell,
    reciprocal_lattice_from_real,
    resolve_mpi_comm,
)


def integrate_over_frequency(
    freq_thz: np.ndarray,
    sqw: np.ndarray,
    freq_min_thz: float,
    freq_max_thz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq_mask, sqw_integrated) — trapezoidal integral over ω."""
    mask = np.ones(freq_thz.shape, dtype=bool)
    if freq_min_thz > 0:
        mask &= freq_thz >= freq_min_thz
    if freq_max_thz is not None:
        mask &= freq_thz <= freq_max_thz
    if not np.any(mask):
        raise ValueError("No frequency points in integration range.")

    freq_sel = freq_thz[mask]
    sqw_sel = sqw[:, mask]
    dw = freq_sel[1] - freq_sel[0]  # uniform grid
    sqw_int = np.trapz(sqw_sel, dx=dw, axis=1) if sqw_sel.shape[1] > 1 else sqw_sel[:, 0] * dw
    return mask, sqw_int


def to_grid(values_inside: np.ndarray, inside_mask: np.ndarray,
            shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, np.nan, dtype=float)
    grid.reshape(-1)[inside_mask] = values_inside
    return grid


def plot_integrated_map(
    qx_inside: np.ndarray,
    qy_inside: np.ndarray,
    sqw_int: np.ndarray,
    outfile: str,
    freq_range: str,
    components: str = "xyz",
    projection: str = "cartesian",
    cbar_min: Optional[float] = None,
    cbar_max: Optional[float] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    triang = mtri.Triangulation(qx_inside, qy_inside)

    mesh_kw = {"cmap": "viridis", "shading": "gouraud"}
    if cbar_min is not None:
        mesh_kw["vmin"] = cbar_min
    if cbar_max is not None:
        mesh_kw["vmax"] = cbar_max

    # mesh = ax.tripcolor(triang, sqw_int, **mesh_kw)
    mesh = ax.tripcolor(triang, np.log10(np.maximum(sqw_int, 1e-12)), **mesh_kw)
    ax.triplot(triang, color="w", linewidth=0.1, alpha=0.15)

    ax.plot(0, 0, "o", markersize=5, color="cyan", markeredgecolor="k")
    ax.annotate(r"$\Gamma$", xy=(0, 0), xytext=(6, 6),
                textcoords="offset points", color="cyan",
                fontsize=11, fontweight="bold")
    ax.set_xlabel(r"$q_x$ (1/Å)")
    ax.set_ylabel(r"$q_y$ (1/Å)")
    ax.set_aspect("equal")

    if projection == "cartesian":
        detail = f"components={components}"
    else:
        detail = f"projection={projection}"
    # ax.set_title(f"∫ S(q,ω) dω   {freq_range}   ({detail})")
    # ax.set_title(f"∫ S(q,ω) dω  ({detail})")
    ax.set_title(f"log10{{∫ S(q,ω) dω}}  ({detail})")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Intensity (arb. units)")
    fig.tight_layout()

    png_file = str(Path(outfile).with_suffix("")) + ".png"
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {png_file}")


# ---------------------------------------------------------------------------
def main() -> None:
    args = build_integrated_arg_parser().parse_args()
    mpi_comm, mpi_rank, mpi_size = resolve_mpi_comm()
    is_root = mpi_rank == 0

    if is_root:
        if mpi_comm is not None:
            print(f"[INFO] MPI: ranks={mpi_size}")
        print("[INFO] Reading dump file ...")

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
    if args.mass_weight:
        if not args.masses:
            raise ValueError("--mass-weight requires --masses (one atomic mass per atom type)")
        calc.set_mass_weights(args.masses)
        if is_root:
            print(
                "[INFO] Mass weighting enabled (phonon convention): field scaled by "
                f"sqrt(mass) per atom; masses by type = {tuple(args.masses)}"
            )
    elif is_root and args.masses:
        print("[INFO] --masses provided but --mass-weight not set; masses ignored (magnon convention).")
    if is_root:
        calc.print_lattice_info()

    prim_lattice = primitive_lattice_from_supercell(
        calc.lattice, np.asarray(args.supercell, dtype=int))
    prim_recip = reciprocal_lattice_from_real(prim_lattice)

    q_frac_all, q_cart_all, inside_mask = generate_bz_points(
        reciprocal_lattice=prim_recip,
        nh=args.nh, nk=args.nk,
        frac_limit=args.frac_limit,
    )
    q_frac_inside = q_frac_all[inside_mask]
    q_engine_inside = cart_to_frac_q(q_cart_all[inside_mask], calc.reciprocal_lattice)

    if is_root:
        print(f"[INFO] Frames: {calc.spins.shape[0]},  Atoms: {calc.positions.shape[1]}")
        print(f"[INFO] BZ grid: {args.nh}x{args.nk},  "
              f"inside 1st BZ: {q_frac_inside.shape[0]} q-points")
        print(f"[INFO] Frequency integration: [{args.freq_min_thz}, "
              f"{args.freq_max_thz or 'Nyquist'}] THz")
        print(f"[INFO] Components: {args.components},  projection: {args.projection}")

    if args.method == "corr":
        result = calc.compute_correlation_spectrum(
            q_frac=q_engine_inside,
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
    else:
        result = calc.compute_periodogram(
            q_frac=q_engine_inside,
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

    if not is_root:
        return

    freq_max = args.freq_max_thz if args.freq_max_thz is not None else result.freq_thz[-1]
    freq_mask, sqw_int = integrate_over_frequency(
        freq_thz=result.freq_thz,
        sqw=result.sqw,
        freq_min_thz=args.freq_min_thz,
        freq_max_thz=args.freq_max_thz,
    )

    freq_str = f"[{args.freq_min_thz}, {freq_max:.1f}] THz"
    print(f"[INFO] Integrated {freq_str}:  "
          f"min={sqw_int.min():.4f},  max={sqw_int.max():.4f},  "
          f"mean={sqw_int.mean():.4f}")

    qx_inside = q_cart_all[inside_mask, 0]
    qy_inside = q_cart_all[inside_mask, 1]

    if args.save_npz:
        int_grid = to_grid(sqw_int, inside_mask, (args.nh, args.nk))
        np.savez(
            args.output,
            q_frac_all=q_frac_all,
            q_frac_inside=q_frac_inside,
            inside_mask=inside_mask,
            qx_inside=qx_inside,
            qy_inside=qy_inside,
            sqw_int=sqw_int,
            sqw_int_grid=int_grid,
            freq_thz=result.freq_thz,
            freq_mask=freq_mask,
            freq_int_range=np.array([args.freq_min_thz, freq_max]),
            prim_lattice=prim_lattice,
            prim_reciprocal=prim_recip,
            supercell=np.asarray(args.supercell, dtype=int),
            components=np.asarray(args.components),
            projection=np.asarray(args.projection),
        )
        print(f"[INFO] Saved to: {args.output}")

    if args.plot:
        plot_integrated_map(
            qx_inside, qy_inside, sqw_int,
            outfile=args.plot_file,
            freq_range=freq_str,
            components=args.components,
            projection=args.projection,
            cbar_min=args.cbar_min,
            cbar_max=args.cbar_max,
        )


if __name__ == "__main__":
    main()
