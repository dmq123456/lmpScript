#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import time

import numpy as np


def _mpi_size_hint_from_env() -> int:
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            size = int(value)
        except ValueError:
            continue
        if size > 0:
            return size
    return 1


def _mpi_rank_hint_from_env() -> int:
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def resolve_mpi_comm() -> tuple[object | None, int, int]:
    if _mpi_size_hint_from_env() <= 1:
        return None, 0, 1

    try:
        from mpi4py import MPI
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Detected mpirun/mpiexec multi-rank launch but mpi4py is unavailable. "
            "Install mpi4py or run without mpirun."
        ) from exc

    comm = MPI.COMM_WORLD
    return comm, int(comm.Get_rank()), int(comm.Get_size())


def _mpi_rank_size(mpi_comm: object | None) -> tuple[int, int]:
    if mpi_comm is None:
        return 0, 1
    if not hasattr(mpi_comm, "Get_rank") or not hasattr(mpi_comm, "Get_size"):
        raise ValueError("mpi_comm must provide Get_rank() and Get_size() methods")
    return int(mpi_comm.Get_rank()), int(mpi_comm.Get_size())


def _q_indices_for_rank(nq: int, rank: int, size: int) -> np.ndarray:
    return np.arange(rank, nq, size, dtype=np.int64)


def _progress_report_interval(total_steps: int, target_reports: int) -> int:
    if total_steps <= 0:
        return 1
    return max(1, math.ceil(total_steps / max(int(target_reports), 1)))


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"

    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _report_q_progress(
    label: str,
    step: int,
    local_total: int,
    global_total: int,
    start_time: float,
    mpi_comm: object | None,
    mpi_rank: int,
) -> None:
    local_done = min(step, local_total)
    if mpi_comm is None:
        global_done = local_done
    else:
        global_done = int(mpi_comm.allreduce(int(local_done)))

    if mpi_rank != 0:
        return

    elapsed = time.perf_counter() - start_time
    frac = float(global_done) / float(max(global_total, 1))
    rate = float(global_done) / elapsed if elapsed > 0.0 else 0.0
    eta = (global_total - global_done) / rate if rate > 0.0 else float("inf")
    print(
        f"[INFO] {label} progress: {global_done}/{global_total} q-points "
        f"({100.0 * frac:.1f}%), elapsed {_format_duration(elapsed)}, "
        f"ETA {_format_duration(eta)}",
        flush=True,
    )


__all__ = [
    "_format_duration",
    "_mpi_rank_hint_from_env",
    "_mpi_rank_size",
    "_mpi_size_hint_from_env",
    "_progress_report_interval",
    "_q_indices_for_rank",
    "_report_q_progress",
    "resolve_mpi_comm",
]
