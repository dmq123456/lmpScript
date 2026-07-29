#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3 of the pipeline: correlation tensor -> S^{ab}(q,w).

    S^{ab}(q,w_k) = sum_{tau=-(Nt-1)}^{Nt-1} exp(-i w_k tau dt) Cwin^{ab}(q,tau)

evaluated by FFT after shifting the two-sided sequence into FFT order. There is
no dt prefactor, so the result is a relative intensity: absolute values are not
comparable between runs with different dt.

Only non-negative frequencies are kept. That is exact when the dynamics are
classical and the system is centrosymmetric, which together give
S(q,-w) = S(q,w). It is *not* a consequence of the Hermitian extension alone:
that makes S real but not even in w. A single propagating mode
s(q,t) = A exp(-i w0 t) yields a peak at w = -w0 only, so for chiral systems
(Dzyaloshinskii-Moriya, handed magnons) weight genuinely sits at w < 0 and is
discarded here.
"""

from __future__ import annotations

import numpy as np


def frequency_grid(dt_fs: float, n_corr: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative FFT frequencies of a length-n_corr transform.

    Returns (freq_thz, pos_mask). With n_corr = 2*Nt-1 odd there are exactly Nt
    non-negative bins, and the largest is (Nt-1)/((2Nt-1) dt) -- approaching but
    never reaching the Nyquist frequency 1/(2 dt).
    """
    dt_s = dt_fs * 1.0e-15
    freq_hz_all = np.fft.fftfreq(n_corr, d=dt_s)
    pos_mask = freq_hz_all >= 0.0
    return freq_hz_all[pos_mask] / 1.0e12, pos_mask


def tensor_spectrum(corr_windowed: np.ndarray, pos_mask: np.ndarray) -> np.ndarray:
    """FFT the windowed two-sided correlation -> S^{ab}(q,w).

    Returns (nfreq, 3, 3) complex, Hermitian in (a,b) at each frequency and
    positive-semidefinite as a spectral-density matrix -- the property that
    channels.py relies on to decide when clipping is legitimate.
    """
    for_fft = np.fft.ifftshift(corr_windowed, axes=0)
    return np.fft.fft(for_fft, axis=0)[pos_mask, :, :]


__all__ = ["frequency_grid", "tensor_spectrum"]
