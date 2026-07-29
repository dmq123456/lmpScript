#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1 of the pipeline: real-space field -> q-resolved mode amplitude.

    s^a(q,t) = (1/sqrt(N)) * Lambda(q) * sum_j w_j S_j^a(t) exp(+i q.r_j)

The exp(+i q.r) convention is chosen so a plane wave exp(i(q.r - wt)) shows up
at the same signed q in S(q,w). The 1/sqrt(N) prefactor makes |s|^2 intensive.
No projection happens here: the full Cartesian triple is always produced, and
the choice of output channel is applied later as a weight on the correlation
tensor (see channels.py).
"""

from __future__ import annotations

import math

import numpy as np

from trajectory import FieldTrajectory


def validate_translation_repeats(repeats) -> np.ndarray:
    out = np.asarray(repeats, dtype=np.int64)
    if out.shape != (3,) or np.any(out < 1):
        raise ValueError("translation_repeats must be three positive integers")
    return out


def lattice_sum_factor(
    lattice: np.ndarray,
    q: np.ndarray,
    repeats: np.ndarray,
) -> complex:
    """Lambda(q) for explicitly repeating the loaded cell N1 x N2 x N3 times.

        Lambda(q) = (1/sqrt(N1 N2 N3)) * prod_d sum_{n=0}^{N_d-1} exp(i n q.a_d)

    The 1/sqrt(N1 N2 N3) keeps the 1/sqrt(N) normalization of the spatial sum
    intact after the repetition. Lambda == 1 when all repeats are 1.
    """
    repeats = validate_translation_repeats(repeats)
    if np.all(repeats == 1):
        return 1.0 + 0.0j
    total = np.complex128(1.0 + 0.0j)
    for axis, nrep in enumerate(repeats):
        theta = float(np.dot(lattice[axis], q))
        total *= np.exp(1j * theta * np.arange(int(nrep), dtype=np.float64)).sum(
            dtype=np.complex128
        )
    return complex(total / math.sqrt(float(np.prod(repeats))))


def spatial_fourier_transform(
    traj: FieldTrajectory,
    q: np.ndarray,
    *,
    use_instantaneous_pos: bool = False,
    translation_repeats=(1, 1, 1),
) -> np.ndarray:
    """s^a(q,t) for one q-point. Returns (nt, 3) complex128.

    Positions are frozen at the first frame unless use_instantaneous_pos is
    set, in which case the full position history is used and the phase factor
    is rebuilt every frame.
    """
    nt = traj.n_frames
    natoms = traj.n_atoms
    norm = 1.0 / math.sqrt(natoms)
    scale = lattice_sum_factor(traj.lattice, q, translation_repeats)

    out = np.empty((nt, 3), dtype=np.complex128)

    if use_instantaneous_pos:
        if traj.positions.shape[0] != nt:
            raise ValueError(
                "Instantaneous positions requested but only the first frame was kept; "
                "reload the trajectory with use_instantaneous_pos=True."
            )
        for it in range(nt):
            phase = np.exp(1j * (traj.positions[it] @ q))
            if traj.weights is not None:
                phase = phase * traj.weights
            out[it] = norm * (traj.spins[it].T.astype(np.complex128) @ phase)
    else:
        phase = np.exp(1j * (traj.positions[0] @ q))
        if traj.weights is not None:
            phase = phase * traj.weights
        # (nt, natoms, 3) contracted over atoms -> (nt, 3)
        out[:] = norm * np.tensordot(traj.spins, phase, axes=([1], [0]))

    out *= scale
    return out


__all__ = [
    "spatial_fourier_transform",
    "lattice_sum_factor",
    "validate_translation_repeats",
]
