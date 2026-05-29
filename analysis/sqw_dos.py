#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqw_args import build_dos_arg_parser
from sqw_core import (
    THZ_TO_MEV,
    SpinStructureFactorCalculator,
    cart_to_frac_q,
    component_indices,
    generate_bz_points,
    primitive_lattice_from_supercell,
    reciprocal_lattice_from_real,
    resolve_mpi_comm,
)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_dos_arg_parser()


def _window_array(size: int, window: str) -> np.ndarray:
    if window == "hann":
        return np.hanning(size)
    if window == "none":
        return np.ones(size)
    raise ValueError("window must be 'hann' or 'none'")


def select_frequency_range(
    freq_thz: np.ndarray,
    values: np.ndarray,
    freq_min_thz: float | None,
    freq_max_thz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(freq_thz.shape, dtype=bool)
    if freq_min_thz is not None:
        mask &= freq_thz >= float(freq_min_thz)
    if freq_max_thz is not None:
        mask &= freq_thz <= float(freq_max_thz)
    if not np.any(mask):
        raise ValueError("No frequency points remain after applying frequency limits.")
    return freq_thz[mask], values[mask]


def gaussian_smooth_1d(values: np.ndarray, sigma_points: float) -> np.ndarray:
    if sigma_points <= 0.0:
        return values
    half = int(np.ceil(4.0 * sigma_points))
    if half < 1:
        return values
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_points) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def normalize_dos(freq_thz: np.ndarray, dos: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return dos
    if mode == "max":
        vmax = float(np.max(dos))
        return dos / vmax if vmax > 0.0 else dos
    if mode == "area":
        area = float(np.trapz(dos, x=freq_thz))
        return dos / area if area > 0.0 else dos
    raise ValueError(f"Unsupported normalization mode: {mode}")


def finalize_dos_curve(
    freq_thz: np.ndarray,
    dos_raw: np.ndarray,
    freq_min_thz: float | None,
    freq_max_thz: float | None,
    smooth_sigma_thz: float,
    normalize: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq_sel, dos_sel_raw = select_frequency_range(freq_thz, dos_raw, freq_min_thz, freq_max_thz)
    if freq_sel.size >= 2 and smooth_sigma_thz > 0.0:
        df = float(np.median(np.diff(freq_sel)))
        sigma_points = smooth_sigma_thz / max(df, 1.0e-15)
    else:
        sigma_points = 0.0
    dos_smoothed = gaussian_smooth_1d(dos_sel_raw, sigma_points)
    dos_final = normalize_dos(freq_sel, dos_smoothed, normalize)
    return freq_sel, dos_sel_raw, dos_final


def compute_dos_qavg(
    calc: SpinStructureFactorCalculator,
    args: argparse.Namespace,
    mpi_comm: object | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    primitive_lattice = primitive_lattice_from_supercell(calc.lattice, np.asarray(args.supercell, dtype=int))
    primitive_reciprocal = reciprocal_lattice_from_real(primitive_lattice)

    q_frac_prim_all, q_cart_all, inside_mask = generate_bz_points(
        reciprocal_lattice=primitive_reciprocal,
        nh=args.nh,
        nk=args.nk,
        frac_limit=args.frac_limit,
    )
    q_frac_prim_inside = q_frac_prim_all[inside_mask]
    q_frac_engine_inside = cart_to_frac_q(q_cart_all[inside_mask], calc.reciprocal_lattice)

    if args.sqw_method == "periodogram":
        result = calc.compute_periodogram(
            q_frac=q_frac_engine_inside,
            dt_fs=args.dt_fs,
            components=args.components,
            projection=args.projection,
            use_instantaneous_pos=args.use_instantaneous_pos,
            translation_repeats=tuple(args.translation_repeats),
            subtract_mean=(not args.no_subtract_mean),
            window=args.window,
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
            mpi_comm=mpi_comm,
        )

    # Under MPI, non-root ranks hold an empty sqw array by design.
    # Avoid np.mean(empty, axis=0) warnings on those ranks.
    if result.sqw.shape[0] == 0:
        dos_raw = np.zeros(result.freq_thz.shape, dtype=np.float64)
    else:
        dos_raw = np.mean(result.sqw, axis=0)
    return result.freq_thz, dos_raw, q_frac_prim_inside, q_frac_engine_inside, q_cart_all[inside_mask], inside_mask


def _build_scalar_spin_autocorr(
    spins: np.ndarray,
    comp_idx: list[int],
    subtract_mean: bool,
    corr_norm: str,
) -> np.ndarray:
    data = np.asarray(spins[:, :, comp_idx], dtype=np.float64)
    if subtract_mean:
        data = data - np.mean(data, axis=0, keepdims=True)

    nt, natoms, ncomp = data.shape
    nch = natoms * ncomp
    data2d = data.reshape(nt, nch)

    # FFT-based autocorrelation (Wiener-Khinchin), O(Nt log Nt) instead of O(Nt^2).
    nfft = int(1 << (2 * nt - 1).bit_length())
    fft_all = np.fft.fft(data2d, n=nfft, axis=0)
    acf = np.fft.ifft(fft_all * np.conjugate(fft_all), axis=0).real
    corr_sum_lag = np.sum(acf[:nt, :], axis=1)

    denom_base = float(nch)
    if corr_norm == "biased":
        return corr_sum_lag / (nt * denom_base)
    if corr_norm == "unbiased":
        lag_norm = (nt - np.arange(nt, dtype=np.float64)) * denom_base
        return corr_sum_lag / lag_norm
    raise ValueError("corr_norm must be 'biased' or 'unbiased'")


def compute_dos_autocorr(
    calc: SpinStructureFactorCalculator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comp_idx = component_indices(args.components)
    corr_plus = _build_scalar_spin_autocorr(
        spins=calc.spins,
        comp_idx=comp_idx,
        subtract_mean=(not args.no_subtract_mean),
        corr_norm=args.corr_norm,
    )

    corr_full = np.concatenate([corr_plus[1:][::-1], corr_plus])
    window_vec = _window_array(corr_full.size, args.window)
    corr_full = corr_full * window_vec

    dt_s = args.dt_fs * 1.0e-15
    freq_hz_all = np.fft.fftfreq(corr_full.size, d=dt_s)
    pos_mask = freq_hz_all >= 0.0
    freq_thz = freq_hz_all[pos_mask] / 1.0e12

    fft_all = np.fft.fft(np.fft.ifftshift(corr_full), axis=0)
    dos_raw = np.real(fft_all[pos_mask])
    dos_raw = np.maximum(dos_raw, 0.0)
    return freq_thz, dos_raw, corr_plus


def plot_dos_curves(
    curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    outfile: str,
    use_mev: bool,
    plot_raw: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

    mirror = len(curves) == 2

    for i, (label, freq_thz, raw, final) in enumerate(curves):
        color = colors[i % len(colors)]
        x = freq_thz * THZ_TO_MEV if use_mev else freq_thz
        sign = 1.0 if i == 0 else -1.0
        y_raw = sign * raw
        y_final = sign * final

        if plot_raw and not np.allclose(raw, final):
            ax.plot(x, y_raw, color=color, alpha=0.35, linewidth=1.0)
        ax.plot(x, y_final, color=color, linewidth=2.0, label=label)

    if mirror:
        ax.axhline(y=0, color="k", linewidth=0.6)
        ymax = max(abs(v) for _, _, _, final in curves for v in (final.max(),))
        for sign in (-1.0, 1.0):
            ax.fill_between(
                curves[0][1], 0, sign * ymax,
                alpha=0.04, color="gray",
            )

    ax.set_xlabel("Energy (meV)" if use_mev else "Frequency (THz)")
    ax.set_ylabel("DOS (arb. units)")
    ax.set_title("DOS")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()
    prefix = str(Path(outfile).with_suffix(""))
    png_file = f"{prefix}.png"
    # eps_file = f"{prefix}.eps"
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    # fig.savefig(eps_file, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {png_file}")
    # print(f"[INFO] Plot saved to: {eps_file}")


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
        print(f"[INFO] Frames: {calc.spins.shape[0]}")
        print(f"[INFO] Atoms : {calc.positions.shape[1]}")
        if args.spin_threshold > 0.0:
            print(f"[INFO] Field threshold filter enabled: {args.spin_threshold:g}")
        print(f"[INFO] Field columns: {tuple(args.field_columns)}")
        print(f"[INFO] Projection mode (qavg route): {args.projection}")
        print(f"[INFO] Using dt_fs = {args.dt_fs} fs between saved frames")

    run_qavg = args.dos_method in ("qavg", "both")
    run_autocorr = args.dos_method in ("autocorr", "both")

    qavg_payload = None
    autocorr_payload = None

    if run_qavg:
        if is_root:
            print(f"[INFO] qavg route: 2D q-grid {args.nh} x {args.nk}, sqw_method={args.sqw_method}")
        freq_qavg, dos_qavg_raw, q_frac_prim_inside, q_frac_engine_inside, q_cart_inside, inside_mask = compute_dos_qavg(
            calc=calc,
            args=args,
            mpi_comm=mpi_comm,
        )
        if is_root:
            freq_qavg_sel, dos_qavg_raw_sel, dos_qavg = finalize_dos_curve(
                freq_qavg,
                dos_qavg_raw,
                args.freq_min_thz,
                args.freq_max_thz,
                args.smooth_sigma_thz,
                args.normalize,
            )
            qavg_payload = {
                "freq_raw": freq_qavg,
                "dos_raw": dos_qavg_raw,
                "freq": freq_qavg_sel,
                "dos_raw_sel": dos_qavg_raw_sel,
                "dos": dos_qavg,
                "q_frac_prim_inside": q_frac_prim_inside,
                "q_frac_engine_inside": q_frac_engine_inside,
                "q_cart_inside": q_cart_inside,
                "inside_mask": inside_mask.reshape(args.nh, args.nk),
            }

    if run_autocorr and is_root:
        print(f"[INFO] autocorr route: components={args.components}, corr_norm={args.corr_norm}")
        if args.projection != "cartesian":
            print("[INFO] autocorr route ignores --projection and uses Cartesian components only")
        freq_auto, dos_auto_raw, corr_plus = compute_dos_autocorr(calc=calc, args=args)
        freq_auto_sel, dos_auto_raw_sel, dos_auto = finalize_dos_curve(
            freq_auto,
            dos_auto_raw,
            args.freq_min_thz,
            args.freq_max_thz,
            args.smooth_sigma_thz,
            args.normalize,
        )
        autocorr_payload = {
            "freq_raw": freq_auto,
            "dos_raw": dos_auto_raw,
            "freq": freq_auto_sel,
            "dos_raw_sel": dos_auto_raw_sel,
            "dos": dos_auto,
            "corr_plus": corr_plus,
        }

    if not is_root:
        return

    if args.save_npz:
        payload: dict[str, object] = {
            "dos_method": np.asarray(args.dos_method),
            "sqw_method": np.asarray(args.sqw_method),
            "components": np.asarray(args.components),
            "corr_norm": np.asarray(args.corr_norm),
            "dt_fs": np.asarray(args.dt_fs, dtype=float),
            "supercell": np.asarray(args.supercell, dtype=int),
            "normalize": np.asarray(args.normalize),
            "smooth_sigma_thz": np.asarray(args.smooth_sigma_thz, dtype=float),
        }
        if qavg_payload is not None:
            payload.update(
                {
                    "qavg_freq_thz_raw": qavg_payload["freq_raw"],
                    "qavg_dos_raw": qavg_payload["dos_raw"],
                    "qavg_freq_thz": qavg_payload["freq"],
                    "qavg_energy_meV": qavg_payload["freq"] * THZ_TO_MEV,
                    "qavg_dos_raw_sel": qavg_payload["dos_raw_sel"],
                    "qavg_dos": qavg_payload["dos"],
                    "qavg_q_frac_prim_inside": qavg_payload["q_frac_prim_inside"],
                    "qavg_q_frac_engine_inside": qavg_payload["q_frac_engine_inside"],
                    "qavg_q_cart_inside": qavg_payload["q_cart_inside"],
                    "qavg_inside_mask": qavg_payload["inside_mask"],
                }
            )
        if autocorr_payload is not None:
            payload.update(
                {
                    "autocorr_freq_thz_raw": autocorr_payload["freq_raw"],
                    "autocorr_dos_raw": autocorr_payload["dos_raw"],
                    "autocorr_freq_thz": autocorr_payload["freq"],
                    "autocorr_energy_meV": autocorr_payload["freq"] * THZ_TO_MEV,
                    "autocorr_dos_raw_sel": autocorr_payload["dos_raw_sel"],
                    "autocorr_dos": autocorr_payload["dos"],
                    "autocorr_corr_plus": autocorr_payload["corr_plus"],
                }
            )

        np.savez(args.output, **payload)
        print(f"[INFO] Results saved to: {args.output}")

    if args.plot:
        curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        if qavg_payload is not None:
            sign = "+" if run_autocorr else ""
            curves.append(
                (f"{sign}qavg", qavg_payload["freq"], qavg_payload["dos_raw_sel"], qavg_payload["dos"])
            )
        if autocorr_payload is not None:
            sign = "−" if run_qavg else ""  # minus sign
            curves.append(
                (f"{sign}autocorr", autocorr_payload["freq"], autocorr_payload["dos_raw_sel"], autocorr_payload["dos"])
            )
        if curves:
            plot_dos_curves(curves=curves, outfile=args.plot_file, use_mev=args.mev, plot_raw=args.plot_raw)


if __name__ == "__main__":
    main()
