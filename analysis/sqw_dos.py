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
    component_indices,
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


def _build_scalar_spin_autocorr(
    spins: np.ndarray,
    comp_idx: list[int],
    subtract_mean: bool,
    corr_norm: str,
    atom_weights: np.ndarray | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    # The Wiener-Khinchin autocorrelation zero-pads the time series to
    # nfft ~ 2*nframes. Doing all atom*component channels in one FFT would allocate
    # an (nfft, natoms*ncomp) array. Since the DOS sums over independent channels,
    # we process one Cartesian component at a time and accumulate; the FFT buffer is
    # then (nfft, natoms) instead of (nfft, natoms*ncomp), i.e. ncomp times smaller.
    # rfft/irfft (real transform) further halves work vs the full complex FFT.
    real_dtype = np.dtype(dtype)
    if real_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        real_dtype = np.dtype(np.float64)

    nt, natoms, _ = spins.shape
    ncomp = len(comp_idx)
    nch = natoms * ncomp

    # Per-atom sqrt(mass) weight (phonon convention): the autocorrelation is
    # quadratic in the field, so scaling each atom's field by sqrt(m_i) yields the
    # standard mass-weighted m_i factor on its VACF contribution. None (magnon)
    # leaves the DOS unchanged.
    w = None if atom_weights is None else np.asarray(atom_weights, dtype=real_dtype)

    nfft = int(1 << (2 * nt - 1).bit_length())

    corr_sum_lag = np.zeros(nt, dtype=np.float64)
    for c in comp_idx:
        chunk = np.asarray(spins[:, :, c], dtype=real_dtype)  # (nt, natoms)
        if subtract_mean:
            chunk = chunk - chunk.mean(axis=0, keepdims=True)
        if w is not None:
            chunk = chunk * w[None, :]
        spec = np.fft.rfft(chunk, n=nfft, axis=0)
        power = spec.real ** 2 + spec.imag ** 2
        acf = np.fft.irfft(power, n=nfft, axis=0)  # (nfft, natoms), real
        corr_sum_lag += np.sum(acf[:nt, :], axis=1, dtype=np.float64)

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
        atom_weights=calc.atom_weights,
        dtype=calc.real_dtype,
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


def plot_dos_curve(
    freq_thz: np.ndarray,
    raw: np.ndarray,
    final: np.ndarray,
    outfile: str,
    use_mev: bool,
    plot_raw: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    x = freq_thz * THZ_TO_MEV if use_mev else freq_thz

    if plot_raw and not np.allclose(raw, final):
        ax.plot(x, raw, color="#1f77b4", alpha=0.35, linewidth=1.0)
    ax.plot(x, final, color="#d62728", linewidth=2.0, label="autocorr")

    ax.set_xlabel("Energy (meV)" if use_mev else "Frequency (THz)")
    ax.set_ylabel("DOS (arb. units)")
    ax.set_title("DOS")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()
    prefix = str(Path(outfile).with_suffix(""))
    png_file = f"{prefix}.png"
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {png_file}")


def main() -> None:
    args = build_arg_parser().parse_args()
    mpi_comm, mpi_rank, mpi_size = resolve_mpi_comm()
    is_root = mpi_rank == 0

    if is_root:
        if mpi_comm is not None and mpi_size > 1:
            print(
                f"[INFO] Note: the autocorr DOS runs on the root rank only; the other "
                f"{mpi_size - 1} rank(s) are idle. Running with a single process is recommended."
            )
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
        print(f"[INFO] Frames: {calc.spins.shape[0]}")
        print(f"[INFO] Atoms : {calc.positions.shape[1]}")
        if args.spin_threshold > 0.0:
            print(f"[INFO] Field threshold filter enabled: {args.spin_threshold:g}")
        print(f"[INFO] Field columns: {tuple(args.field_columns)}")
        print(f"[INFO] Using dt_fs = {args.dt_fs} fs between saved frames")

    # Only the autocorrelation (velocity/spin VACF) DOS is computed, on root only.
    if not is_root:
        return

    print(f"[INFO] autocorr route: components={args.components}, corr_norm={args.corr_norm}")
    if args.projection != "cartesian":
        print("[INFO] autocorr route ignores --projection and uses Cartesian components only")

    freq_auto, dos_auto_raw, corr_plus = compute_dos_autocorr(calc=calc, args=args)
    freq_sel, dos_raw_sel, dos = finalize_dos_curve(
        freq_auto,
        dos_auto_raw,
        args.freq_min_thz,
        args.freq_max_thz,
        args.smooth_sigma_thz,
        args.normalize,
    )

    if args.save_npz:
        payload: dict[str, object] = {
            "components": np.asarray(args.components),
            "corr_norm": np.asarray(args.corr_norm),
            "dt_fs": np.asarray(args.dt_fs, dtype=float),
            "supercell": np.asarray(args.supercell, dtype=int),
            "normalize": np.asarray(args.normalize),
            "smooth_sigma_thz": np.asarray(args.smooth_sigma_thz, dtype=float),
            "autocorr_freq_thz_raw": freq_auto,
            "autocorr_dos_raw": dos_auto_raw,
            "autocorr_freq_thz": freq_sel,
            "autocorr_energy_meV": freq_sel * THZ_TO_MEV,
            "autocorr_dos_raw_sel": dos_raw_sel,
            "autocorr_dos": dos,
            "autocorr_corr_plus": corr_plus,
        }
        np.savez(args.output, **payload)
        print(f"[INFO] Results saved to: {args.output}")

    if args.plot:
        plot_dos_curve(
            freq_thz=freq_sel,
            raw=dos_raw_sel,
            final=dos,
            outfile=args.plot_file,
            use_mev=args.mev,
            plot_raw=args.plot_raw,
        )


if __name__ == "__main__":
    main()
