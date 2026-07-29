#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Density of states from the on-site time-correlation of a three-component field.

    D(w) = FT[ sum_j < x_j(t+tau) x_j(t) > ]

where x_j is a scalar time series obtained by projecting the field on atom j
onto one direction in spin space. No q enters anywhere: summing the correlation
over every lattice site is, by Parseval, the same as integrating S(q,w) over
the whole Brillouin zone, so the DOS costs one transform per atom rather than a
spectrum on a q-mesh.

Which direction (or directions) to project onto is what a channel selects. A
channel is a real 3x3 weight matrix W, and the quantity it names is

    sum_ab W_ab C^{ab},    C^{ab}(tau) = sum_j < S_j^a(t+tau) S_j^b(t) > .

Rather than build the nine cross-correlations and contract them, this module
diagonalises W = sum_i lambda_i v_i v_i^T and computes

    sum_i lambda_i * [ autocorrelation of (v_i . S_j) ] ,

which is the same number. Two things follow. Only rank(W) transforms are
needed instead of up to nine, and each one is an ordinary autocorrelation of a
real scalar, so exactly one spectrum is ever resident in memory. For the
default trace, W = I, the eigenvectors are the Cartesian axes and this reduces
to the obvious thing: autocorrelate S^x, S^y, S^z and add.

The decomposition requires W to be symmetric. That is not a real restriction
here: a non-symmetric W names a signed quantity that is not a spectral density,
and a density of states is by definition non-negative.
"""

from __future__ import annotations

import numpy as np


def channel_directions(weight: np.ndarray, tol: float = 1.0e-12):
    """Diagonalise a channel weight into (lambda_i, v_i) pairs.

    Zero eigenvalues are dropped, so a rank-1 channel such as `1` returns a
    single direction and costs a single transform.
    """
    w = np.asarray(weight, dtype=np.float64)
    if not np.allclose(w, w.T, atol=tol):
        raise ValueError(
            "This channel's weight matrix is not symmetric, so it names a signed "
            "quantity rather than a spectral density. A density of states is "
            "non-negative by definition; use a symmetric channel such as 1, 1+5+9, "
            "or a diagonal group."
        )
    eigenvalues, eigenvectors = np.linalg.eigh(w)
    return [(float(l), eigenvectors[:, i])
            for i, l in enumerate(eigenvalues) if abs(l) > tol]


def onsite_autocorrelation(
    traj,
    direction: np.ndarray,
    *,
    subtract_mean: bool,
    window_vec: np.ndarray,
) -> np.ndarray:
    """sum_j < x_j(t+tau) x_j(t) > for x_j = direction . S_j, tau = 0..Nt-1.

    Evaluated through the Wiener-Khinchin identity with the series zero-padded
    past 2*Nt, which makes the circular correlation of the padded signal equal
    the linear correlation of the original -- an exact identity, not an
    approximation. The projected field is real, so the half-spectrum transforms
    rfft/irfft do this at half the cost of a complex pair.

    Everything runs in float64 regardless of the storage dtype of the
    trajectory: the accumulation runs over Nt terms and would otherwise eat
    into float32's seven digits.
    """
    nt = traj.n_frames
    x = np.asarray(traj.spins, dtype=np.float64) @ np.asarray(direction, dtype=np.float64)
    if subtract_mean:
        x = x - x.mean(axis=0, keepdims=True)
    if traj.weights is not None:
        x = x * traj.weights[None, :]
    x = x * window_vec[:, None]

    n_pad = 1 << (2 * nt - 1).bit_length()
    spectrum = np.fft.rfft(x, n=n_pad, axis=0)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    return np.fft.irfft(power, n=n_pad, axis=0)[:nt].sum(axis=1)


def channel_correlation(
    traj,
    weight: np.ndarray,
    *,
    subtract_mean: bool,
    window_vec: np.ndarray,
    corr_norm: str,
) -> np.ndarray:
    """The one-sided correlation behind one channel, C(tau), tau = 0..Nt-1.

    corr_norm selects the denominator: `biased` divides by a constant Nt and
    keeps the estimate a genuine power spectrum, `unbiased` divides by the
    Nt-tau terms that actually contributed and removes the triangular bias at
    the cost of positive semidefiniteness.
    """
    if corr_norm not in ("biased", "unbiased"):
        raise ValueError("corr_norm must be 'biased' or 'unbiased'")

    nt, natoms = traj.n_frames, traj.n_atoms
    total = np.zeros(nt, dtype=np.float64)
    for eigenvalue, direction in channel_directions(weight):
        total += eigenvalue * onsite_autocorrelation(
            traj, direction, subtract_mean=subtract_mean, window_vec=window_vec
        )

    if corr_norm == "biased":
        return total / float(nt * natoms)
    return total / ((nt - np.arange(nt, dtype=np.float64)) * natoms)


def spectrum_from_correlation(
    corr_plus: np.ndarray,
    dt_fs: float,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend C(tau) to negative lags and transform. Returns (freq_thz, D)."""
    # C(-tau) = C(tau): projecting a real field on a real direction leaves a
    # real, even correlation, so the extension is a plain mirror.
    full = np.concatenate([corr_plus[1:][::-1], corr_plus])

    freq_hz = np.fft.fftfreq(full.size, d=dt_fs * 1.0e-15)
    positive = freq_hz >= 0.0
    values = np.real(np.fft.fft(np.fft.ifftshift(full)))[positive]
    if clip:
        values = np.maximum(values, 0.0)
    return freq_hz[positive] / 1.0e12, values


# ----------------------------------------------------------------------
# Presentation of the curve
# ----------------------------------------------------------------------
def select_frequency_range(freq_thz, values, freq_min_thz, freq_max_thz):
    mask = np.ones(freq_thz.shape, dtype=bool)
    if freq_min_thz is not None:
        mask &= freq_thz >= float(freq_min_thz)
    if freq_max_thz is not None:
        mask &= freq_thz <= float(freq_max_thz)
    if not np.any(mask):
        raise ValueError("No frequency points remain after applying the frequency limits.")
    return freq_thz[mask], values[mask]


def gaussian_smooth(values: np.ndarray, sigma_points: float) -> np.ndarray:
    if sigma_points <= 0.0:
        return values
    half = int(np.ceil(4.0 * sigma_points))
    if half < 1:
        return values
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_points) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def normalise(freq_thz: np.ndarray, dos: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return dos
    if mode == "max":
        peak = float(np.max(dos))
        return dos / peak if peak > 0.0 else dos
    if mode == "area":
        area = float(np.trapz(dos, x=freq_thz))
        return dos / area if area > 0.0 else dos
    raise ValueError(f"Unsupported normalisation mode: {mode}")


def finalise_curve(freq_thz, dos_raw, *, freq_min_thz, freq_max_thz,
                   smooth_sigma_thz, normalize):
    """Trim to the requested range, smooth, normalise. Returns (freq, raw, final)."""
    freq, raw = select_frequency_range(freq_thz, dos_raw, freq_min_thz, freq_max_thz)
    if freq.size >= 2 and smooth_sigma_thz > 0.0:
        df = float(np.median(np.diff(freq)))
        sigma_points = smooth_sigma_thz / max(df, 1.0e-15)
    else:
        sigma_points = 0.0
    return freq, raw, normalise(freq, gaussian_smooth(raw, sigma_points), normalize)


__all__ = [
    "channel_directions",
    "onsite_autocorrelation",
    "channel_correlation",
    "spectrum_from_correlation",
    "select_frequency_range",
    "gaussian_smooth",
    "normalise",
    "finalise_curve",
]
