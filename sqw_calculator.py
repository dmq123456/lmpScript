#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from sqw_geometry import (
    cart_to_frac_q,
    component_indices,
    frac_to_cart_q,
    reciprocal_lattice_from_real,
)
from sqw_io import (
    apply_frame_selection_to_loaded_arrays,
    binary_cache_path,
    finalize_loaded_arrays,
    load_binary_arrays,
    load_binary_cache_if_valid,
    load_spin_dump,
    normalize_field_columns,
    normalize_frame_selection,
    parse_text_dump,
    write_binary_cache,
)
from sqw_mpi import (
    _mpi_rank_size,
    _progress_report_interval,
    _q_indices_for_rank,
    _report_q_progress,
)
from sqw_result import SQWResult


class SpinStructureFactorCalculator:
    def __init__(
        self,
        timesteps: np.ndarray,
        positions: np.ndarray,
        spins: np.ndarray,
        lattice: np.ndarray,
        field_columns: tuple[str, str, str] | None = None,
    ) -> None:
        self.timesteps = timesteps
        self.positions = positions
        self.spins = spins
        self.lattice = lattice
        self.field_columns = field_columns
        self.reciprocal_lattice = reciprocal_lattice_from_real(lattice)
        self.real_dtype = np.float32 if self.spins.dtype == np.float32 else np.float64
        self.complex_dtype = np.complex64 if self.real_dtype == np.float32 else np.complex128

    @staticmethod
    def _binary_cache_path(source_path: Path) -> Path:
        return binary_cache_path(source_path)

    @staticmethod
    def _load_binary_arrays(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        return load_binary_arrays(cache_path)

    @classmethod
    def _load_binary_cache_if_valid(
        cls,
        cache_path: Path,
        source_path: Path,
        keep_all_positions: bool,
        dtype: np.dtype,
        frame_start: int,
        frame_stop: int | None,
        frame_step: int,
        spin_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        return load_binary_cache_if_valid(
            cache_path=cache_path,
            source_path=source_path,
            keep_all_positions=keep_all_positions,
            dtype=dtype,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
            spin_threshold=spin_threshold,
        )

    @staticmethod
    def _write_binary_cache(
        cache_path: Path,
        source_path: Path,
        timesteps: np.ndarray,
        positions: np.ndarray,
        spins: np.ndarray,
        lattice: np.ndarray,
        keep_all_positions: bool,
        dtype: np.dtype,
        frame_start: int,
        frame_stop: int | None,
        frame_step: int,
        spin_threshold: float,
    ) -> None:
        write_binary_cache(
            cache_path=cache_path,
            source_path=source_path,
            timesteps=timesteps,
            positions=positions,
            spins=spins,
            lattice=lattice,
            keep_all_positions=keep_all_positions,
            dtype=dtype,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
            spin_threshold=spin_threshold,
        )

    @staticmethod
    def _apply_frame_selection_to_loaded_arrays(
        timesteps_arr: np.ndarray,
        positions_arr: np.ndarray,
        spins_arr: np.ndarray,
        frame_start: int,
        frame_stop: int | None,
        frame_step: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return apply_frame_selection_to_loaded_arrays(
            timesteps_arr=timesteps_arr,
            positions_arr=positions_arr,
            spins_arr=spins_arr,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
        )

    @staticmethod
    def _finalize_loaded_arrays(
        timesteps_arr: np.ndarray,
        positions_arr: np.ndarray,
        spins_arr: np.ndarray,
        lattice_arr: np.ndarray,
        keep_all_positions: bool,
        dtype: np.dtype,
        spin_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return finalize_loaded_arrays(
            timesteps_arr=timesteps_arr,
            positions_arr=positions_arr,
            spins_arr=spins_arr,
            lattice_arr=lattice_arr,
            keep_all_positions=keep_all_positions,
            dtype=dtype,
            spin_threshold=spin_threshold,
        )

    @staticmethod
    def _parse_text_dump(
        filename: str | Path,
        keep_all_positions: bool,
        dtype: np.dtype,
        frame_start: int,
        frame_stop: int | None,
        frame_step: int,
        progress: bool = False,
        progress_reports: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return parse_text_dump(
            filename=filename,
            keep_all_positions=keep_all_positions,
            dtype=dtype,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
            progress=progress,
            progress_reports=progress_reports,
        )

    @classmethod
    def from_dump(
        cls,
        filename: str | Path,
        keep_all_positions: bool = False,
        dtype: np.dtype = np.float32,
        frame_start: int | None = None,
        frame_stop: int | None = None,
        frame_step: int | None = None,
        cache_binary: bool = True,
        cache_file: str | Path | None = None,
        spin_threshold: float = 0.0,
        field_columns: tuple[str, str, str] | list[str] | None = None,
        progress: bool = False,
        progress_reports: int = 20,
    ) -> "SpinStructureFactorCalculator":
        normalized_field_columns = normalize_field_columns(field_columns)
        normalize_frame_selection(
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
        )
        timesteps_arr, positions_arr, spins_arr, lattice_arr = load_spin_dump(
            filename=filename,
            keep_all_positions=keep_all_positions,
            dtype=dtype,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
            cache_binary=cache_binary,
            cache_file=cache_file,
            spin_threshold=spin_threshold,
            field_columns=normalized_field_columns,
            progress=progress,
            progress_reports=progress_reports,
        )
        return cls(
            timesteps=timesteps_arr,
            positions=positions_arr,
            spins=spins_arr,
            lattice=lattice_arr,
            field_columns=normalized_field_columns,
        )

    def q_frac_to_cart(self, q_frac: np.ndarray) -> np.ndarray:
        return frac_to_cart_q(q_frac, self.reciprocal_lattice)

    def q_cart_to_frac(self, q_cart: np.ndarray) -> np.ndarray:
        return cart_to_frac_q(q_cart, self.reciprocal_lattice)

    def print_lattice_info(self, prefix: str = "[INFO]") -> None:
        print(f"{prefix} First-frame real-space lattice vectors (rows):")
        for idx, vec in enumerate(self.lattice, start=1):
            print(f"{prefix}   a{idx} = ({vec[0]: .10f}, {vec[1]: .10f}, {vec[2]: .10f})")
        print(f"{prefix} First-frame reciprocal lattice vectors (rows):")
        for idx, vec in enumerate(self.reciprocal_lattice, start=1):
            print(f"{prefix}   b{idx} = ({vec[0]: .10f}, {vec[1]: .10f}, {vec[2]: .10f})")

    def _build_sqt_single_q(
        self,
        q: np.ndarray,
        use_instantaneous_pos: bool,
        translation_repeats: np.ndarray,
        projection: str,
    ) -> np.ndarray:
        nt, natoms, _ = self.spins.shape
        out = np.empty((nt, 3), dtype=self.complex_dtype)
        norm = 1.0 / math.sqrt(natoms)
        translation_scale = self._translation_lattice_sum(q, translation_repeats)
        projection = projection.lower().strip()
        q_norm = float(np.linalg.norm(q))
        qhat = None if q_norm <= 1.0e-15 else (q / q_norm).astype(self.real_dtype, copy=False)

        def project_frame(field_frame: np.ndarray) -> np.ndarray:
            if projection == "cartesian":
                return field_frame
            if projection == "longitudinal":
                projected = np.zeros_like(field_frame)
                if qhat is not None:
                    projected[..., 0] = np.tensordot(field_frame, qhat, axes=([-1], [0]))
                return projected
            if projection == "transverse":
                if qhat is None:
                    return field_frame
                parallel = np.tensordot(field_frame, qhat, axes=([-1], [0]))[..., None] * qhat
                return field_frame - parallel
            raise ValueError("projection must be 'cartesian', 'longitudinal', or 'transverse'")

        if use_instantaneous_pos:
            if self.positions.shape[0] != nt:
                raise ValueError(
                    "Instantaneous positions requested but full position history is not available. "
                    "Re-read dump with keep_all_positions=True."
                )
            for it in range(nt):
                # Use the exp(+i q·r) Fourier convention so a mode exp(i(q·r-wt))
                # appears at the same signed q in S(q,w).
                phase = np.exp(1j * (self.positions[it] @ q)).astype(self.complex_dtype, copy=False)
                field_frame = project_frame(self.spins[it])
                for alpha in range(3):
                    out[it, alpha] = norm * np.dot(field_frame[:, alpha], phase)
        else:
            r0 = self.positions[0]
            phase = np.exp(1j * (r0 @ q)).astype(self.complex_dtype, copy=False)
            projected = project_frame(self.spins)
            for alpha in range(3):
                out[:, alpha] = norm * (projected[:, :, alpha] @ phase)
        out *= translation_scale
        return out

    @staticmethod
    def _resolve_component_indices(components: str, projection: str) -> List[int]:
        projection = projection.lower().strip()
        if projection == "cartesian":
            return component_indices(components)
        if projection == "longitudinal":
            return [0]
        if projection == "transverse":
            return [0, 1, 2]
        raise ValueError("projection must be 'cartesian', 'longitudinal', or 'transverse'")

    @staticmethod
    def _validate_translation_repeats(translation_repeats: Tuple[int, int, int] | np.ndarray) -> np.ndarray:
        repeats = np.asarray(translation_repeats, dtype=np.int64)
        if repeats.shape != (3,) or np.any(repeats < 1):
            raise ValueError("translation_repeats must be three positive integers")
        return repeats

    def _translation_lattice_sum(self, q: np.ndarray, translation_repeats: np.ndarray) -> np.complexfloating:
        repeats = self._validate_translation_repeats(translation_repeats)
        if np.all(repeats == 1):
            return self.complex_dtype(1.0 + 0.0j)

        lattice_sum = np.complex128(1.0 + 0.0j)
        for axis, nrep in enumerate(repeats):
            theta = float(np.dot(self.lattice[axis], q))
            axis_sum = np.exp(1j * theta * np.arange(int(nrep), dtype=np.float64)).sum(dtype=np.complex128)
            lattice_sum *= axis_sum

        # Keep the same 1/sqrt(N) normalization convention after explicitly
        # repeating the loaded simulation cell prod(repeats) times.
        lattice_sum /= math.sqrt(float(np.prod(repeats)))
        return self.complex_dtype(lattice_sum)

    @staticmethod
    def _window_array(size: int, window: str) -> np.ndarray:
        if window.lower() == "hann":
            return np.hanning(size)
        if window.lower() == "none":
            return np.ones(size)
        raise ValueError("window must be 'hann' or 'none'")

    @staticmethod
    def _fft_frequency_grid(dt_fs: float, nt: int) -> tuple[np.ndarray, np.ndarray]:
        dt_s = dt_fs * 1.0e-15
        freq_hz_all = np.fft.fftfreq(nt, d=dt_s)
        pos_mask = freq_hz_all >= 0.0
        return freq_hz_all[pos_mask] / 1.0e12, freq_hz_all[pos_mask]

    @staticmethod
    def _direct_frequency_grid(
        dt_fs: float,
        freq_min_thz: float,
        freq_max_thz: float | None,
        freq_step_thz: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if freq_max_thz is None:
            raise ValueError("--freq-max-thz is required when --freq-mode direct")
        if freq_step_thz is None:
            raise ValueError("--freq-step-thz is required when --freq-mode direct")
        if freq_min_thz < 0.0:
            raise ValueError("--freq-min-thz must be >= 0")
        if freq_max_thz < freq_min_thz:
            raise ValueError("--freq-max-thz must be >= --freq-min-thz")
        if freq_step_thz <= 0.0:
            raise ValueError("--freq-step-thz must be > 0")

        nyquist_thz = 0.5 / (dt_fs * 1.0e-15) / 1.0e12
        if freq_max_thz > nyquist_thz + 1.0e-12:
            raise ValueError(
                f"--freq-max-thz={freq_max_thz:g} exceeds Nyquist frequency {nyquist_thz:g} THz"
            )

        freq_thz = np.arange(
            freq_min_thz,
            freq_max_thz + 0.5 * freq_step_thz,
            freq_step_thz,
            dtype=float,
        )
        if freq_thz.size == 0:
            raise ValueError("Direct FT frequency grid is empty")
        return freq_thz, freq_thz * 1.0e12

    def _compute_sqw_from_sqt_series(
        self,
        sqt_q: np.ndarray,
        pos_mask: np.ndarray,
        comp_idx: List[int],
        subtract_mean: bool,
        window_vec: np.ndarray,
    ) -> np.ndarray:
        nt = sqt_q.shape[0]
        data = sqt_q.copy()
        if subtract_mean:
            data -= data.mean(axis=0, keepdims=True)
        data *= window_vec[:, None]

        intensity = np.zeros(int(np.count_nonzero(pos_mask)), dtype=np.float64)
        for alpha in comp_idx:
            fftv_all = np.fft.fft(data[:, alpha], axis=0)
            intensity += np.abs(fftv_all[pos_mask]) ** 2
        return intensity / nt

    def _compute_sqw_from_sqt_series_direct(
        self,
        sqt_q: np.ndarray,
        freq_hz: np.ndarray,
        comp_idx: List[int],
        subtract_mean: bool,
        window_vec: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        nt = sqt_q.shape[0]
        data = sqt_q.copy()
        if subtract_mean:
            data -= data.mean(axis=0, keepdims=True)
        data *= window_vec[:, None]

        intensity = np.zeros(freq_hz.size, dtype=np.float64)
        if freq_hz.size == 0:
            return intensity

        times_s = dt_s * np.arange(nt, dtype=np.float64)
        max_phase_bytes = 64 * 1024 * 1024
        bytes_per_freq = max(nt, 1) * np.dtype(np.complex128).itemsize
        block_size = max(1, min(freq_hz.size, max_phase_bytes // max(bytes_per_freq, 1)))

        for start in range(0, freq_hz.size, block_size):
            stop = min(start + block_size, freq_hz.size)
            phase = np.exp(
                -2j * np.pi * freq_hz[start:stop, None] * times_s[None, :]
            ).astype(np.complex128, copy=False)
            for alpha in comp_idx:
                amplitudes = phase @ data[:, alpha].astype(np.complex128, copy=False)
                intensity[start:stop] += np.abs(amplitudes) ** 2
        return intensity / nt

    @staticmethod
    def _compute_time_correlation_single_q(
        sqt_q: np.ndarray,
        subtract_mean: bool,
        corr_norm: str,
    ) -> np.ndarray:
        data = sqt_q.copy()
        if subtract_mean:
            data -= data.mean(axis=0, keepdims=True)

        nt = data.shape[0]
        corr_plus = np.zeros((nt, 3), dtype=data.dtype)
        for tau in range(nt):
            left = data[tau:, :]
            right = np.conjugate(data[: nt - tau, :])
            corr_sum = np.sum(left * right, axis=0)
            if corr_norm == "biased":
                corr_plus[tau, :] = corr_sum / nt
            elif corr_norm == "unbiased":
                corr_plus[tau, :] = corr_sum / (nt - tau)
            else:
                raise ValueError("corr_norm must be 'biased' or 'unbiased'")
        return corr_plus

    @staticmethod
    def _build_full_correlation(corr_plus: np.ndarray) -> np.ndarray:
        neg = np.conjugate(corr_plus[1:, :][::-1, :])
        return np.concatenate([neg, corr_plus], axis=0)

    @staticmethod
    def _compute_sqw_from_corr_series(
        corr_plus_q: np.ndarray,
        pos_mask: np.ndarray,
        comp_idx: List[int],
        window_vec: np.ndarray,
    ) -> np.ndarray:
        corr_full = SpinStructureFactorCalculator._build_full_correlation(corr_plus_q)
        corr_full *= window_vec[:, None]
        corr_for_fft = np.fft.ifftshift(corr_full, axes=0)
        fft_all = np.fft.fft(corr_for_fft, axis=0)
        sqw = np.real(fft_all[pos_mask, :][:, comp_idx].sum(axis=1))
        return np.maximum(sqw, 0.0)

    def _compute_sqw_from_sqt(
        self,
        q_vectors: np.ndarray,
        dt_fs: float,
        components: str,
        projection: str,
        use_instantaneous_pos: bool,
        translation_repeats: np.ndarray,
        subtract_mean: bool,
        window: str,
        freq_mode: str,
        freq_min_thz: float,
        freq_max_thz: float | None,
        freq_step_thz: float | None,
        progress: bool,
        progress_reports: int,
        mpi_comm: object | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nq = q_vectors.shape[0]
        nt = self.spins.shape[0]
        comp_idx = self._resolve_component_indices(components, projection)
        window_vec = self._window_array(nt, window).astype(self.real_dtype, copy=False)

        dt_s = dt_fs * 1.0e-15
        pos_mask = None
        if freq_mode == "fft":
            freq_thz, freq_hz = self._fft_frequency_grid(dt_fs=dt_fs, nt=nt)
            freq_hz_all = np.fft.fftfreq(nt, d=dt_s)
            pos_mask = freq_hz_all >= 0.0
        elif freq_mode == "direct":
            freq_thz, freq_hz = self._direct_frequency_grid(
                dt_fs=dt_fs,
                freq_min_thz=freq_min_thz,
                freq_max_thz=freq_max_thz,
                freq_step_thz=freq_step_thz,
            )
        else:
            raise ValueError("freq_mode must be 'fft' or 'direct'")
        mpi_rank, mpi_size = _mpi_rank_size(mpi_comm)
        if mpi_size == 1 or mpi_rank == 0:
            sqw = np.zeros((nq, freq_thz.size), dtype=np.float64)
        else:
            sqw = np.empty((0, freq_thz.size), dtype=np.float64)

        local_indices = _q_indices_for_rank(nq, mpi_rank, mpi_size)
        local_sqw = np.zeros((local_indices.size, freq_thz.size), dtype=np.float64)
        max_local_steps = (nq + mpi_size - 1) // mpi_size if nq > 0 else 0
        report_interval = _progress_report_interval(max_local_steps, progress_reports)
        progress_start = time.perf_counter()

        for iloc in range(max_local_steps):
            if iloc < local_indices.size:
                iq = local_indices[iloc]
                sqt_q = self._build_sqt_single_q(
                    q_vectors[int(iq)],
                    use_instantaneous_pos=use_instantaneous_pos,
                    translation_repeats=translation_repeats,
                    projection=projection,
                )
                if freq_mode == "fft":
                    local_sqw[iloc] = self._compute_sqw_from_sqt_series(
                        sqt_q=sqt_q,
                        pos_mask=pos_mask,
                        comp_idx=comp_idx,
                        subtract_mean=subtract_mean,
                        window_vec=window_vec,
                    )
                else:
                    local_sqw[iloc] = self._compute_sqw_from_sqt_series_direct(
                        sqt_q=sqt_q,
                        freq_hz=freq_hz,
                        comp_idx=comp_idx,
                        subtract_mean=subtract_mean,
                        window_vec=window_vec,
                        dt_s=dt_s,
                    )

            step = iloc + 1
            if progress and (step == max_local_steps or step % report_interval == 0):
                _report_q_progress(
                    label="S(q,w)",
                    step=step,
                    local_total=int(local_indices.size),
                    global_total=int(nq),
                    start_time=progress_start,
                    mpi_comm=mpi_comm,
                    mpi_rank=mpi_rank,
                )

        if mpi_size == 1:
            sqw[local_indices] = local_sqw
        else:
            gathered = mpi_comm.gather((local_indices, local_sqw), root=0)
            if mpi_rank == 0:
                for gathered_indices, gathered_sqw in gathered:
                    sqw[gathered_indices] = gathered_sqw

        return freq_thz, sqw

    def _compute_sqw_from_corr(
        self,
        q_vectors: np.ndarray,
        dt_fs: float,
        components: str,
        projection: str,
        use_instantaneous_pos: bool,
        translation_repeats: np.ndarray,
        subtract_mean: bool,
        window: str,
        corr_norm: str,
        return_corr_plus: bool,
        progress: bool,
        progress_reports: int,
        mpi_comm: object | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        nq = q_vectors.shape[0]
        nt = self.spins.shape[0]
        comp_idx = self._resolve_component_indices(components, projection)

        n_corr = 2 * nt - 1
        window_vec = self._window_array(n_corr, window).astype(self.real_dtype, copy=False)
        dt_s = dt_fs * 1.0e-15
        freq_hz_all = np.fft.fftfreq(n_corr, d=dt_s)
        pos_mask = freq_hz_all >= 0.0
        freq_thz = freq_hz_all[pos_mask] / 1.0e12

        mpi_rank, mpi_size = _mpi_rank_size(mpi_comm)
        if mpi_size == 1 or mpi_rank == 0:
            sqw = np.zeros((nq, freq_thz.size), dtype=np.float64)
        else:
            sqw = np.empty((0, freq_thz.size), dtype=np.float64)

        corr_plus_all = None
        if return_corr_plus and (mpi_size == 1 or mpi_rank == 0):
            corr_plus_all = np.zeros((nq, nt, 3), dtype=self.complex_dtype)
        local_indices = _q_indices_for_rank(nq, mpi_rank, mpi_size)
        local_sqw = np.zeros((local_indices.size, freq_thz.size), dtype=np.float64)
        local_corr_plus = None
        if return_corr_plus:
            local_corr_plus = np.zeros((local_indices.size, nt, 3), dtype=self.complex_dtype)
        max_local_steps = (nq + mpi_size - 1) // mpi_size if nq > 0 else 0
        report_interval = _progress_report_interval(max_local_steps, progress_reports)
        progress_start = time.perf_counter()

        for iloc in range(max_local_steps):
            if iloc < local_indices.size:
                iq = local_indices[iloc]
                sqt_q = self._build_sqt_single_q(
                    q_vectors[int(iq)],
                    use_instantaneous_pos=use_instantaneous_pos,
                    translation_repeats=translation_repeats,
                    projection=projection,
                )
                corr_plus_q = self._compute_time_correlation_single_q(
                    sqt_q=sqt_q,
                    subtract_mean=subtract_mean,
                    corr_norm=corr_norm,
                )
                local_sqw[iloc] = self._compute_sqw_from_corr_series(
                    corr_plus_q=corr_plus_q,
                    pos_mask=pos_mask,
                    comp_idx=comp_idx,
                    window_vec=window_vec,
                )
                if return_corr_plus:
                    local_corr_plus[iloc] = corr_plus_q

            step = iloc + 1
            if progress and (step == max_local_steps or step % report_interval == 0):
                _report_q_progress(
                    label="S(q,w) from C(q,tau)",
                    step=step,
                    local_total=int(local_indices.size),
                    global_total=int(nq),
                    start_time=progress_start,
                    mpi_comm=mpi_comm,
                    mpi_rank=mpi_rank,
                )

        if mpi_size == 1:
            sqw[local_indices] = local_sqw
            if return_corr_plus:
                corr_plus_all[local_indices] = local_corr_plus
        else:
            if return_corr_plus:
                gathered = mpi_comm.gather((local_indices, local_sqw, local_corr_plus), root=0)
                if mpi_rank == 0:
                    for gathered_indices, gathered_sqw, gathered_corr in gathered:
                        sqw[gathered_indices] = gathered_sqw
                        corr_plus_all[gathered_indices] = gathered_corr
            else:
                gathered = mpi_comm.gather((local_indices, local_sqw), root=0)
                if mpi_rank == 0:
                    for gathered_indices, gathered_sqw in gathered:
                        sqw[gathered_indices] = gathered_sqw

        return freq_thz, sqw, corr_plus_all

    def compute_periodogram(
        self,
        q_frac: np.ndarray,
        dt_fs: float,
        components: str = "xyz",
        projection: str = "cartesian",
        use_instantaneous_pos: bool = False,
        translation_repeats: Tuple[int, int, int] = (1, 1, 1),
        subtract_mean: bool = True,
        window: str = "hann",
        freq_mode: str = "fft",
        freq_min_thz: float = 0.0,
        freq_max_thz: float | None = None,
        freq_step_thz: float | None = None,
        progress: bool = True,
        progress_reports: int = 20,
        q_node_indices: np.ndarray | None = None,
        q_node_labels: List[str] | None = None,
        mpi_comm: object | None = None,
    ) -> SQWResult:
        q_vectors = self.q_frac_to_cart(q_frac)
        repeats = self._validate_translation_repeats(translation_repeats)
        freq_thz, sqw = self._compute_sqw_from_sqt(
            q_vectors=q_vectors,
            dt_fs=dt_fs,
            components=components,
            projection=projection,
            use_instantaneous_pos=use_instantaneous_pos,
            translation_repeats=repeats,
            subtract_mean=subtract_mean,
            window=window,
            freq_mode=freq_mode,
            freq_min_thz=freq_min_thz,
            freq_max_thz=freq_max_thz,
            freq_step_thz=freq_step_thz,
            progress=progress,
            progress_reports=progress_reports,
            mpi_comm=mpi_comm,
        )
        return SQWResult(
            timesteps=self.timesteps,
            q_frac=q_frac,
            q_vectors=q_vectors,
            freq_thz=freq_thz,
            sqw=sqw,
            lattice=self.lattice,
            reciprocal_lattice=self.reciprocal_lattice,
            dt_fs=dt_fs,
            components=components,
            field_columns=np.asarray(self.field_columns) if self.field_columns is not None else None,
            projection=projection,
            freq_mode=freq_mode,
            translation_repeats=repeats,
            q_node_indices=q_node_indices,
            q_node_labels=q_node_labels,
        )

    def compute_correlation_spectrum(
        self,
        q_frac: np.ndarray,
        dt_fs: float,
        components: str = "xyz",
        projection: str = "cartesian",
        use_instantaneous_pos: bool = False,
        translation_repeats: Tuple[int, int, int] = (1, 1, 1),
        subtract_mean: bool = True,
        window: str = "hann",
        corr_norm: str = "biased",
        return_corr_plus: bool = False,
        progress: bool = True,
        progress_reports: int = 20,
        q_node_indices: np.ndarray | None = None,
        q_node_labels: List[str] | None = None,
        mpi_comm: object | None = None,
    ) -> SQWResult:
        q_vectors = self.q_frac_to_cart(q_frac)
        repeats = self._validate_translation_repeats(translation_repeats)
        freq_thz, sqw, corr_plus = self._compute_sqw_from_corr(
            q_vectors=q_vectors,
            dt_fs=dt_fs,
            components=components,
            projection=projection,
            use_instantaneous_pos=use_instantaneous_pos,
            translation_repeats=repeats,
            subtract_mean=subtract_mean,
            window=window,
            corr_norm=corr_norm,
            return_corr_plus=return_corr_plus,
            progress=progress,
            progress_reports=progress_reports,
            mpi_comm=mpi_comm,
        )
        return SQWResult(
            timesteps=self.timesteps,
            q_frac=q_frac,
            q_vectors=q_vectors,
            freq_thz=freq_thz,
            sqw=sqw,
            lattice=self.lattice,
            reciprocal_lattice=self.reciprocal_lattice,
            dt_fs=dt_fs,
            components=components,
            field_columns=np.asarray(self.field_columns) if self.field_columns is not None else None,
            projection=projection,
            freq_mode="fft",
            translation_repeats=repeats,
            q_node_indices=q_node_indices,
            q_node_labels=q_node_labels,
            corr_plus=corr_plus,
            corr_norm=corr_norm,
        )


__all__ = ["SpinStructureFactorCalculator"]
