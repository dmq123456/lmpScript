#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The one place where parallelism lives.

map_over_q distributes q-points across MPI ranks, runs a per-q kernel, reports
progress and gathers the result on rank 0. Keeping it here means the physics
routines stay serial, single-q and testable, and an MPI bug has exactly one
place to hide.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from mpi import (
    _mpi_rank_size,
    _progress_report_interval,
    _q_indices_for_rank,
    _report_q_progress,
)


def map_over_q(
    q_vectors: np.ndarray,
    kernel: Callable[[int, np.ndarray], np.ndarray],
    out_width: int,
    *,
    n_channels: int = 1,
    aux_shape: tuple | None = None,
    aux_dtype=np.complex128,
    mpi_comm: object | None = None,
    progress: bool = True,
    progress_reports: int = 20,
    label: str = "S(q,w)",
):
    """Run kernel(iq, q_cart) for every q-point and gather the results.

    The kernel returns an array of shape (n_channels, out_width). If aux_shape
    is given it must instead return a pair (main, aux) where aux has that shape;
    the auxiliary array is gathered alongside the main one and returned as
    (nq,) + aux_shape. This is how the raw correlation C^{ab}(q,tau) is carried
    out of the loop for --save-corr-plus without a second pass over the data.

    Returns the (n_channels, nq, out_width) array on rank 0 -- plus the
    auxiliary array when requested. Non-root ranks get empty arrays with
    matching trailing shape.
    """
    nq = int(q_vectors.shape[0])
    rank, size = _mpi_rank_size(mpi_comm)
    is_root = size == 1 or rank == 0

    out = np.zeros((n_channels, nq if is_root else 0, out_width), dtype=np.float64)
    aux_out = None
    if aux_shape is not None:
        aux_out = np.zeros(((nq if is_root else 0),) + tuple(aux_shape), dtype=aux_dtype)

    local_indices = _q_indices_for_rank(nq, rank, size)
    local_out = np.zeros((n_channels, local_indices.size, out_width), dtype=np.float64)
    local_aux = None
    if aux_shape is not None:
        local_aux = np.zeros((local_indices.size,) + tuple(aux_shape), dtype=aux_dtype)

    max_local_steps = (nq + size - 1) // size if nq > 0 else 0
    report_interval = _progress_report_interval(max_local_steps, progress_reports)
    started = time.perf_counter()

    for iloc in range(max_local_steps):
        if iloc < local_indices.size:
            iq = int(local_indices[iloc])
            result = kernel(iq, q_vectors[iq])
            if aux_shape is None:
                local_out[:, iloc, :] = result
            else:
                main, aux = result
                local_out[:, iloc, :] = main
                local_aux[iloc] = aux

        step = iloc + 1
        if progress and (step == max_local_steps or step % report_interval == 0):
            _report_q_progress(
                label=label,
                step=step,
                local_total=int(local_indices.size),
                global_total=nq,
                start_time=started,
                mpi_comm=mpi_comm,
                mpi_rank=rank,
            )

    if size == 1:
        out[:, local_indices, :] = local_out
        if aux_shape is not None:
            aux_out[local_indices] = local_aux
    else:
        payload = (local_indices, local_out) if aux_shape is None else (
            local_indices, local_out, local_aux
        )
        gathered = mpi_comm.gather(payload, root=0)
        if rank == 0:
            for chunk in gathered:
                if aux_shape is None:
                    idx, main = chunk
                else:
                    idx, main, aux = chunk
                    aux_out[idx] = aux
                out[:, idx, :] = main

    return out if aux_shape is None else (out, aux_out)


__all__ = ["map_over_q"]
