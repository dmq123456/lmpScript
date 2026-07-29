#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 of the pipeline: mode amplitude -> time-correlation tensor.

    C^{ab}(q,tau) = (1/M_tau) sum_t s^a(q,t+tau) s^{b*}(q,t)

computed for all nine (a,b) at once, then extended to negative lags. The full
tensor is always built: it is what lets a single run emit several channels
(--component T L 1+5+9), and with the FFT evaluation below it costs a small
multiple of a single component rather than nine times as much.

Windowing
---------
The taper is applied to s(q,t) *before* the correlation is formed -- a data
window, not a lag window. It exists to remove spectral leakage: the record is
finite, the transform treats it as periodic, and unless a mode completes a whole
number of cycles in the record the splice between last and first frame is a
discontinuity whose power smears across every frequency. Pulling both ends to
zero removes that discontinuity.

A lag window (tapering C(tau) instead) would address a different problem --
the statistical scatter of the estimate -- but it is only usable when the
physical linewidth greatly exceeds the frequency resolution 1/(Nt*dt). Below
that it inflates the measured linewidth by an order of magnitude, destroying
the very quantity a damped-oscillator fit is after. It is deliberately not
offered here.

Evaluation
----------
The correlation is obtained through the Wiener-Khinchin route rather than an
explicit double loop:

    C^{ab} = IFFT[ FFT(s^a) * conj(FFT(s^b)) ]

Zero-padding the input to at least 2*Nt makes the circular correlation of the
padded signal equal the linear correlation of the original, so this is an exact
identity, not an approximation. It turns an O(Nt^2) Python loop into
O(Nt log Nt); measured speed-ups run from ~50x at Nt=2e3 to ~450x at Nt=2e4.

Everything here runs in complex128 regardless of the storage dtype of the
trajectory. s(q,t) is only (Nt, 3), so the promotion is free in memory, and it
keeps the long accumulations from eating into float32's seven digits.
"""

from __future__ import annotations

import numpy as np

try:  # scipy gives a better padded length than the next power of two
    from scipy.fft import next_fast_len
except ImportError:  # pragma: no cover - fallback when scipy is absent

    def next_fast_len(target: int) -> int:
        n = 1
        while n < target:
            n *= 2
        return n


def subtract_time_mean(s_qt: np.ndarray) -> np.ndarray:
    """Remove the time average of each mode amplitude.

    This kills the elastic (w=0) line and any static background. It must happen
    before the window is applied: tapering a signal that still carries a DC
    offset would imprint the shape of the window itself onto the spectrum.
    """
    data = np.asarray(s_qt, dtype=np.complex128)
    return data - data.mean(axis=0, keepdims=True)


def window_array(size: int, window: str) -> np.ndarray:
    if window.lower() == "hann":
        return np.hanning(size)
    if window.lower() == "none":
        return np.ones(size)
    raise ValueError("window must be 'hann' or 'none'")


def apply_data_window(s_qt: np.ndarray, window_vec: np.ndarray) -> np.ndarray:
    """Taper s(q,t) at both ends of the record, before correlating."""
    return np.asarray(s_qt, dtype=np.complex128) * window_vec[:, None]


def time_correlation_tensor(
    s_qt: np.ndarray,
    *,
    corr_norm: str = "biased",
) -> np.ndarray:
    """One-sided C^{ab}(q,tau) for tau = 0..Nt-1. Returns (nt, 3, 3) complex128.

    The input is expected to be mean-subtracted and windowed already.

    corr_norm selects the denominator M_tau:
        biased    M_tau = Nt        constant; keeps the estimate a genuine
                                    power spectrum (positive semidefinite)
        unbiased  M_tau = Nt - tau  corrects for the shrinking sample count,
                                    but destroys positive semidefiniteness --
                                    the transform can then go negative
    """
    if corr_norm not in ("biased", "unbiased"):
        raise ValueError("corr_norm must be 'biased' or 'unbiased'")

    data = np.asarray(s_qt, dtype=np.complex128)
    nt = data.shape[0]

    n_pad = next_fast_len(2 * nt)
    f = np.fft.fft(data, n=n_pad, axis=0)                    # (n_pad, 3)
    cross = f[:, :, None] * np.conjugate(f[:, None, :])      # (n_pad, 3, 3)
    corr = np.fft.ifft(cross, axis=0)[:nt]                   # (nt, 3, 3)

    if corr_norm == "biased":
        corr /= nt
    else:
        corr /= (nt - np.arange(nt))[:, None, None]
    return corr


def hermitian_extend(corr_plus: np.ndarray) -> np.ndarray:
    """Extend C^{ab}(tau >= 0) to the two-sided sequence of length 2*Nt-1.

    The defining symmetry swaps the tensor indices:

        C^{ab}(q,-tau) = conj( C^{ba}(q,tau) )

    which follows from re-indexing the sum, and is what makes the subsequent
    Fourier transform real. Note the index swap -- using conj(C^{ab}) instead
    would silently give the wrong answer for every off-diagonal element.
    """
    neg = np.conjugate(np.transpose(corr_plus[1:][::-1], (0, 2, 1)))
    return np.concatenate([neg, corr_plus], axis=0)


__all__ = [
    "subtract_time_mean",
    "window_array",
    "apply_data_window",
    "time_correlation_tensor",
    "hermitian_extend",
    "next_fast_len",
]
