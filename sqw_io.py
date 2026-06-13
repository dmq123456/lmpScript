#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import List

import numpy as np

from sqw_geometry import lammps_box_to_lattice
from sqw_mpi import _format_duration, _mpi_rank_hint_from_env


DEFAULT_FIELD_COLUMNS = ("vx", "vy", "vz")


def normalize_frame_selection(
    frame_start: int | None = None,
    frame_stop: int | None = None,
    frame_step: int | None = None,
) -> tuple[int, int | None, int]:
    start = 0 if frame_start is None else int(frame_start)
    stop = None if frame_stop is None else int(frame_stop)
    step = 1 if frame_step is None else int(frame_step)

    if start < 0:
        raise ValueError("frame_start must be >= 0")
    if stop is not None and stop < 0:
        raise ValueError("frame_stop must be >= 0")
    if stop is not None and stop <= start:
        raise ValueError("frame_stop must be greater than frame_start")
    if step < 1:
        raise ValueError("frame_step must be >= 1")

    return start, stop, step


def normalize_field_columns(field_columns: tuple[str, str, str] | list[str] | None) -> tuple[str, str, str]:
    if field_columns is None:
        return DEFAULT_FIELD_COLUMNS

    if len(field_columns) != 3:
        raise ValueError("field_columns must contain exactly three column names")

    normalized = tuple(str(name).strip() for name in field_columns)
    if any(not name for name in normalized):
        raise ValueError("field_columns entries must be non-empty")
    return normalized


def binary_cache_path(source_path: Path) -> Path:
    return Path(f"{source_path}.sqwcache.npz")


def _report_dump_read_progress(
    *,
    scanned_frames: int,
    kept_frames: int,
    current_bytes: int | None,
    total_bytes: int | None,
    total_frames: int | None,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time

    if total_frames is not None:
        frac = float(scanned_frames) / float(max(total_frames, 1))
        rate = float(scanned_frames) / elapsed if elapsed > 0.0 else 0.0
        eta = (total_frames - scanned_frames) / rate if rate > 0.0 else float("inf")
        message = (
            f"[INFO] Dump read progress: {scanned_frames}/{total_frames} frames "
            f"({100.0 * frac:.1f}%), kept {kept_frames}"
        )
        message += (
            f", elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)}"
        )
        print(message, flush=True)
        return

    if total_bytes is not None and total_bytes > 0 and current_bytes is not None:
        frac = min(float(current_bytes) / float(total_bytes), 1.0)
        rate = float(current_bytes) / elapsed if elapsed > 0.0 else 0.0
        eta = (total_bytes - current_bytes) / rate if rate > 0.0 else float("inf")
        print(
            f"[INFO] Dump read progress: file {100.0 * frac:.1f}%, "
            f"scanned {scanned_frames} frames, kept {kept_frames}, "
            f"elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)}",
            flush=True,
        )
        return

    print(
        f"[INFO] Dump read progress: scanned {scanned_frames} frames, kept {kept_frames}, "
        f"elapsed {_format_duration(elapsed)}",
        flush=True,
    )


def load_binary_arrays(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            required = ("timesteps", "positions", "spins", "lattice")
            if any(key not in data for key in required):
                return None
            timesteps_arr = np.asarray(data["timesteps"]).copy()
            positions_arr = np.asarray(data["positions"]).copy()
            spins_arr = np.asarray(data["spins"]).copy()
            lattice_arr = np.asarray(data["lattice"]).copy()
            atom_types_arr = np.asarray(data["atom_types"]).copy() if "atom_types" in data else None
            return timesteps_arr, positions_arr, spins_arr, lattice_arr, atom_types_arr
    except Exception:
        return None


def load_binary_cache_if_valid(
    cache_path: Path,
    source_path: Path,
    keep_all_positions: bool,
    dtype: np.dtype,
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
    spin_threshold: float,
    field_columns: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    if not cache_path.exists():
        return None

    try:
        source_stat = source_path.stat()
        source_resolved = str(source_path.resolve())
    except OSError:
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            required = ("timesteps", "positions", "spins", "lattice")
            meta_required = (
                "meta_source_path",
                "meta_source_size",
                "meta_source_mtime_ns",
                "meta_dtype",
                "meta_frame_start",
                "meta_frame_stop",
                "meta_frame_step",
                "meta_keep_all_positions",
                "meta_spin_threshold",
                "meta_field_columns",
            )
            if any(key not in data for key in required + meta_required):
                return None

            meta_source_path = str(np.asarray(data["meta_source_path"]).item())
            meta_source_size = int(np.asarray(data["meta_source_size"]).item())
            meta_source_mtime_ns = int(np.asarray(data["meta_source_mtime_ns"]).item())
            meta_dtype = str(np.asarray(data["meta_dtype"]).item())
            meta_frame_start = int(np.asarray(data["meta_frame_start"]).item())
            meta_frame_stop = int(np.asarray(data["meta_frame_stop"]).item())
            meta_frame_step = int(np.asarray(data["meta_frame_step"]).item())
            meta_keep_all_positions = int(np.asarray(data["meta_keep_all_positions"]).item())
            meta_spin_threshold = float(np.asarray(data["meta_spin_threshold"]).item())
            meta_field_columns = str(np.asarray(data["meta_field_columns"]).item())

            requested_frame_stop = -1 if frame_stop is None else int(frame_stop)
            if meta_source_path != source_resolved:
                return None
            if meta_source_size != int(source_stat.st_size):
                return None
            if meta_source_mtime_ns != int(source_stat.st_mtime_ns):
                return None
            if meta_dtype != np.dtype(dtype).name:
                return None
            if meta_frame_start != int(frame_start):
                return None
            if meta_frame_stop != requested_frame_stop:
                return None
            if meta_frame_step != int(frame_step):
                return None
            if meta_keep_all_positions != int(bool(keep_all_positions)):
                return None
            if abs(meta_spin_threshold - float(spin_threshold)) > 1.0e-15:
                return None
            if meta_field_columns != "\t".join(field_columns):
                return None

            timesteps_arr = np.asarray(data["timesteps"]).copy()
            positions_arr = np.asarray(data["positions"]).copy()
            spins_arr = np.asarray(data["spins"]).copy()
            lattice_arr = np.asarray(data["lattice"]).copy()
            atom_types_arr = np.asarray(data["atom_types"]).copy() if "atom_types" in data else None
            return timesteps_arr, positions_arr, spins_arr, lattice_arr, atom_types_arr
    except Exception:
        return None


def write_binary_cache(
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
    field_columns: tuple[str, str, str],
    atom_types: np.ndarray | None = None,
) -> None:
    source_stat = source_path.stat()
    source_resolved = str(source_path.resolve())
    frame_stop_value = -1 if frame_stop is None else int(frame_stop)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f".{cache_path.name}.{os.getpid()}.tmp.npz"
    save_kwargs = dict(
        timesteps=np.asarray(timesteps),
        positions=np.asarray(positions),
        spins=np.asarray(spins),
        lattice=np.asarray(lattice),
        meta_source_path=np.asarray(source_resolved),
        meta_source_size=np.asarray(int(source_stat.st_size), dtype=np.int64),
        meta_source_mtime_ns=np.asarray(int(source_stat.st_mtime_ns), dtype=np.int64),
        meta_dtype=np.asarray(np.dtype(dtype).name),
        meta_frame_start=np.asarray(int(frame_start), dtype=np.int64),
        meta_frame_stop=np.asarray(frame_stop_value, dtype=np.int64),
        meta_frame_step=np.asarray(int(frame_step), dtype=np.int64),
        meta_keep_all_positions=np.asarray(int(bool(keep_all_positions)), dtype=np.int64),
        meta_spin_threshold=np.asarray(float(spin_threshold), dtype=np.float64),
        meta_field_columns=np.asarray("\t".join(field_columns)),
    )
    if atom_types is not None:
        save_kwargs["atom_types"] = np.asarray(atom_types, dtype=np.int64)
    try:
        np.savez_compressed(tmp_path, **save_kwargs)
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def apply_frame_selection_to_loaded_arrays(
    timesteps_arr: np.ndarray,
    positions_arr: np.ndarray,
    spins_arr: np.ndarray,
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame_start == 0 and frame_stop is None and frame_step == 1:
        return timesteps_arr, positions_arr, spins_arr

    nt_total = int(timesteps_arr.shape[0])
    stop = nt_total if frame_stop is None else min(int(frame_stop), nt_total)
    indices = np.arange(frame_start, stop, frame_step, dtype=np.int64)

    timesteps_sel = timesteps_arr[indices]
    spins_sel = spins_arr[indices]
    if positions_arr.shape[0] == nt_total:
        positions_sel = positions_arr[indices]
    elif positions_arr.shape[0] == 1:
        positions_sel = positions_arr
    else:
        raise ValueError("Invalid positions array shape in binary input.")

    return timesteps_sel, positions_sel, spins_sel


def finalize_loaded_arrays(
    timesteps_arr: np.ndarray,
    positions_arr: np.ndarray,
    spins_arr: np.ndarray,
    lattice_arr: np.ndarray,
    keep_all_positions: bool,
    dtype: np.dtype,
    spin_threshold: float,
    atom_types: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    dtype = np.dtype(dtype)
    timesteps_arr = np.asarray(timesteps_arr, dtype=int)
    positions_arr = np.asarray(positions_arr, dtype=dtype)
    spins_arr = np.asarray(spins_arr, dtype=dtype)
    lattice_arr = np.asarray(lattice_arr, dtype=float)
    if atom_types is not None:
        atom_types = np.asarray(atom_types, dtype=np.int64)
        if atom_types.ndim != 1 or atom_types.shape[0] != spins_arr.shape[1]:
            raise ValueError(
                "atom_types length does not match the number of atoms in the field array."
            )

    if timesteps_arr.size < 2:
        raise ValueError("Need at least 2 frames to compute a frequency spectrum.")
    if spins_arr.ndim != 3 or spins_arr.shape[0] != timesteps_arr.size:
        raise ValueError("Invalid spins array in input.")
    if positions_arr.ndim != 3 or positions_arr.shape[2] != 3:
        raise ValueError("Invalid positions array in input.")
    if keep_all_positions and positions_arr.shape[0] != timesteps_arr.size:
        raise ValueError(
            "Instantaneous positions requested but full position history is not available. "
            "Re-read dump with keep_all_positions=True."
        )
    if not keep_all_positions and positions_arr.shape[0] > 1:
        positions_arr = positions_arr[:1]
    if positions_arr.shape[0] == 0:
        raise ValueError(
            "No position frames kept; adjust frame_start/frame_stop/frame_step settings."
        )

    if spin_threshold > 0.0:
        spin_norm = np.linalg.norm(spins_arr, axis=2)
        keep_atoms = np.max(spin_norm, axis=0) > float(spin_threshold)
        if not np.any(keep_atoms):
            raise ValueError(
                "No atoms remain after applying spin threshold; "
                "decrease --spin-threshold."
            )
        spins_arr = spins_arr[:, keep_atoms, :]
        positions_arr = positions_arr[:, keep_atoms, :]
        if atom_types is not None:
            atom_types = atom_types[keep_atoms]

    return timesteps_arr, positions_arr, spins_arr, lattice_arr, atom_types


def parse_text_dump(
    filename: str | Path,
    keep_all_positions: bool,
    dtype: np.dtype,
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
    field_columns: tuple[str, str, str],
    progress: bool = False,
    progress_reports: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timesteps: List[int] = []
    positions: List[np.ndarray] = []
    spins: List[np.ndarray] = []
    lattice: np.ndarray | None = None
    atom_types: np.ndarray | None = None

    line_num = 0
    frame_count_total = 0
    natoms_ref: int | None = None
    source_path = Path(filename)
    is_root_rank = _mpi_rank_hint_from_env() == 0
    show_progress = bool(progress) and is_root_rank
    progress_start = time.perf_counter()
    total_frames_target = int(frame_stop) if frame_stop is not None else None
    total_bytes: int | None = None
    try:
        total_bytes = int(source_path.stat().st_size)
    except OSError:
        total_bytes = None

    frame_report_interval = None
    byte_report_interval = None
    next_frame_report = None
    next_byte_report = None
    last_report_frames = 0
    last_report_bytes = 0
    if show_progress:
        if progress_reports < 1:
            raise ValueError("progress_reports must be >= 1")
        if total_frames_target is not None:
            frame_report_interval = max(1, math.ceil(total_frames_target / progress_reports))
            next_frame_report = frame_report_interval
        elif total_bytes is not None and total_bytes > 0:
            byte_report_interval = max(1, math.ceil(total_bytes / progress_reports))
            next_byte_report = byte_report_interval

    field_columns = normalize_field_columns(field_columns)

    with open(filename, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_num += 1

            stripped = line.strip()
            if not stripped:
                continue

            if stripped != "ITEM: TIMESTEP":
                raise ValueError(
                    f"Expected 'ITEM: TIMESTEP' at line {line_num}, got: {stripped!r}"
                )

            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file after ITEM: TIMESTEP")
            line_num += 1
            timestep = int(line.strip())

            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file after timestep value")
            line_num += 1
            if line.strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Expected 'ITEM: NUMBER OF ATOMS' at line {line_num}")

            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file after ITEM: NUMBER OF ATOMS")
            line_num += 1
            natoms = int(line.strip())
            if natoms_ref is None:
                natoms_ref = natoms
            elif natoms != natoms_ref:
                raise ValueError("NUMBER OF ATOMS changes across frames; unsupported")

            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file before ITEM: BOX BOUNDS")
            line_num += 1
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Expected 'ITEM: BOX BOUNDS' at line {line_num}")
            box_header = line

            box_lines: List[str] = []
            for _ in range(3):
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file while reading BOX BOUNDS")
                line_num += 1
                box_lines.append(line)
            if lattice is None:
                lattice = lammps_box_to_lattice(box_header, box_lines)

            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file before ITEM: ATOMS")
            line_num += 1
            atoms_header = line.strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Expected 'ITEM: ATOMS' at line {line_num}")
            header_cols = atoms_header.split()[2:]
            col_map = {name: idx for idx, name in enumerate(header_cols)}
            required = ["x", "y", "z", *field_columns]
            missing = [c for c in required if c not in col_map]
            if missing:
                raise ValueError(f"Missing required columns in ATOMS header: {missing}")
            has_type = "type" in col_map

            keep_frame = frame_count_total >= frame_start
            if frame_stop is not None and frame_count_total >= frame_stop:
                keep_frame = False
            elif keep_frame:
                keep_frame = ((frame_count_total - frame_start) % frame_step) == 0

            capture_types = keep_frame and has_type and atom_types is None
            if keep_frame:
                pos_frame = np.empty((natoms, 3), dtype=dtype)
                field_frame = np.empty((natoms, 3), dtype=dtype)
            if capture_types:
                type_frame = np.empty(natoms, dtype=np.int32)

            for ia in range(natoms):
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file while reading atom lines")
                line_num += 1
                if not keep_frame:
                    continue
                vals = line.split()
                pos_frame[ia, 0] = float(vals[col_map["x"]])
                pos_frame[ia, 1] = float(vals[col_map["y"]])
                pos_frame[ia, 2] = float(vals[col_map["z"]])
                field_frame[ia, 0] = float(vals[col_map[field_columns[0]]])
                field_frame[ia, 1] = float(vals[col_map[field_columns[1]]])
                field_frame[ia, 2] = float(vals[col_map[field_columns[2]]])
                if capture_types:
                    type_frame[ia] = int(vals[col_map["type"]])

            if keep_frame:
                timesteps.append(timestep)
                spins.append(field_frame)
                if keep_all_positions:
                    positions.append(pos_frame)
                elif not positions:
                    positions.append(pos_frame)
            if capture_types:
                atom_types = type_frame

            frame_count_total += 1
            if show_progress:
                current_bytes = f.tell() if total_bytes is not None else None
                should_report = False
                if total_frames_target is not None and next_frame_report is not None:
                    if frame_count_total >= next_frame_report or frame_count_total == total_frames_target:
                        should_report = True
                        next_frame_report += frame_report_interval
                elif (
                    total_bytes is not None
                    and byte_report_interval is not None
                    and next_byte_report is not None
                    and current_bytes is not None
                    and current_bytes >= next_byte_report
                ):
                    should_report = True
                    while next_byte_report <= current_bytes:
                        next_byte_report += byte_report_interval

                if should_report:
                    _report_dump_read_progress(
                        scanned_frames=frame_count_total,
                        kept_frames=len(timesteps),
                        current_bytes=current_bytes,
                        total_bytes=total_bytes,
                        total_frames=total_frames_target,
                        start_time=progress_start,
                    )
                    last_report_frames = frame_count_total
                    if current_bytes is not None:
                        last_report_bytes = current_bytes

            if frame_stop is not None and frame_count_total >= frame_stop:
                break

        if show_progress:
            current_bytes = f.tell() if total_bytes is not None else None
            final_needs_report = frame_count_total > 0 and (
                frame_count_total != last_report_frames
                or (
                    total_bytes is not None
                    and current_bytes is not None
                    and current_bytes != last_report_bytes
                )
            )
            if final_needs_report:
                _report_dump_read_progress(
                    scanned_frames=frame_count_total,
                    kept_frames=len(timesteps),
                    current_bytes=current_bytes,
                    total_bytes=total_bytes,
                    total_frames=total_frames_target,
                    start_time=progress_start,
                )

    timesteps_arr = np.asarray(timesteps, dtype=int)
    if timesteps_arr.size < 2:
        raise ValueError("Need at least 2 frames to compute a frequency spectrum.")
    if lattice is None:
        raise ValueError("No BOX BOUNDS block found in dump file.")
    if not positions:
        raise ValueError("No position frames kept; adjust frame_start/frame_stop/frame_step settings.")

    positions_arr = np.asarray(positions, dtype=dtype)
    spins_arr = np.asarray(spins, dtype=dtype)
    lattice_arr = np.asarray(lattice, dtype=float)
    return timesteps_arr, positions_arr, spins_arr, lattice_arr, atom_types


def load_spin_dump(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = np.dtype(dtype)
    frame_start, frame_stop, frame_step = normalize_frame_selection(
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_step=frame_step,
    )
    spin_threshold = float(spin_threshold)
    if spin_threshold < 0.0:
        raise ValueError("spin_threshold must be >= 0")
    field_columns = normalize_field_columns(field_columns)
    source_path = Path(filename)

    if source_path.suffix.lower() == ".npz":
        binary_loaded = load_binary_arrays(source_path)
        if binary_loaded is None:
            raise ValueError(
                "Binary input does not contain required arrays: "
                "'timesteps', 'positions', 'spins', 'lattice'."
            )
        timesteps_arr, positions_arr, spins_arr, lattice_arr, atom_types_arr = binary_loaded
        timesteps_arr, positions_arr, spins_arr = apply_frame_selection_to_loaded_arrays(
            timesteps_arr=timesteps_arr,
            positions_arr=positions_arr,
            spins_arr=spins_arr,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
        )
        return finalize_loaded_arrays(
            timesteps_arr=timesteps_arr,
            positions_arr=positions_arr,
            spins_arr=spins_arr,
            lattice_arr=lattice_arr,
            keep_all_positions=keep_all_positions,
            dtype=dtype,
            spin_threshold=spin_threshold,
            atom_types=atom_types_arr,
        )

    cache_path = Path(cache_file) if cache_file is not None else binary_cache_path(source_path)
    if cache_binary:
        cached_loaded = load_binary_cache_if_valid(
            cache_path=cache_path,
            source_path=source_path,
            keep_all_positions=keep_all_positions,
            dtype=dtype,
            frame_start=frame_start,
            frame_stop=frame_stop,
            frame_step=frame_step,
            spin_threshold=spin_threshold,
            field_columns=field_columns,
        )
        if cached_loaded is not None:
            return finalize_loaded_arrays(
                timesteps_arr=cached_loaded[0],
                positions_arr=cached_loaded[1],
                spins_arr=cached_loaded[2],
                lattice_arr=cached_loaded[3],
                keep_all_positions=keep_all_positions,
                dtype=dtype,
                spin_threshold=spin_threshold,
                atom_types=cached_loaded[4],
            )

    timesteps_arr, positions_arr, spins_arr, lattice_arr, atom_types_arr = parse_text_dump(
        filename=source_path,
        keep_all_positions=keep_all_positions,
        dtype=dtype,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_step=frame_step,
        field_columns=field_columns,
        progress=progress,
        progress_reports=progress_reports,
    )

    if cache_binary and _mpi_rank_hint_from_env() == 0:
        try:
            write_binary_cache(
                cache_path=cache_path,
                source_path=source_path,
                timesteps=timesteps_arr,
                positions=positions_arr,
                spins=spins_arr,
                lattice=lattice_arr,
                keep_all_positions=keep_all_positions,
                dtype=dtype,
                frame_start=frame_start,
                frame_stop=frame_stop,
                frame_step=frame_step,
                spin_threshold=spin_threshold,
                field_columns=field_columns,
                atom_types=atom_types_arr,
            )
        except Exception:
            pass

    return finalize_loaded_arrays(
        timesteps_arr=timesteps_arr,
        positions_arr=positions_arr,
        spins_arr=spins_arr,
        lattice_arr=lattice_arr,
        keep_all_positions=keep_all_positions,
        dtype=dtype,
        spin_threshold=spin_threshold,
        atom_types=atom_types_arr,
    )


__all__ = [
    "apply_frame_selection_to_loaded_arrays",
    "binary_cache_path",
    "finalize_loaded_arrays",
    "load_binary_arrays",
    "load_binary_cache_if_valid",
    "load_spin_dump",
    "normalize_field_columns",
    "normalize_frame_selection",
    "parse_text_dump",
    "write_binary_cache",
]
