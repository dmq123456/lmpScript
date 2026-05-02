#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a selected frame of a LAMMPS dump, write magnetic atoms to XSF, "
            "and optionally plot spin texture."
        )
    )
    parser.add_argument("input_dump", type=Path)
    parser.add_argument("output_xsf", type=Path, nargs='?', default="test.xsf")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help=(
            "Frame index to extract (0-based). Negative indices are from the end "
            "(-1 means last frame, default: -1)."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spin texture from the generated XSF.",
    )
    parser.add_argument(
        "--plot-prefix",
        type=Path,
        default=None,
        help="Output prefix for plot files. Defaults to the XSF filename without suffix.",
    )
    parser.add_argument(
        "--color-component",
        choices=["x", "y", "z"],
        default="x",
        help="Spin component mapped to the color field in the plot (default: x).",
    )
    return parser.parse_args()


def parse_box_bounds(bound_lines: list[str]) -> tuple[list[float], list[float], list[float]]:
    xlo_bound, xhi_bound, xy = (float(x) for x in bound_lines[0].split())
    ylo_bound, yhi_bound, xz = (float(x) for x in bound_lines[1].split())
    zlo_bound, zhi_bound, yz = (float(x) for x in bound_lines[2].split())

    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    zlo = zlo_bound
    zhi = zhi_bound

    avec = [xhi - xlo, 0.0, 0.0]
    bvec = [xy, yhi - ylo, 0.0]
    cvec = [xz, yz, zhi - zlo]
    return avec, bvec, cvec


def _read_required_line(fh, err_msg: str) -> str:
    line = fh.readline()
    if not line:
        raise ValueError(err_msg)
    return line


def _read_one_frame(fh) -> tuple[int, list[str], list[str], list[str]] | None:
    first = fh.readline()
    if not first:
        return None
    if first.strip() != "ITEM: TIMESTEP":
        raise ValueError(f"Unexpected dump format: expected 'ITEM: TIMESTEP', got {first.strip()!r}")

    timestep = int(_read_required_line(fh, "Unexpected EOF after ITEM: TIMESTEP").strip())

    if _read_required_line(fh, "Unexpected EOF before NUMBER OF ATOMS").strip() != "ITEM: NUMBER OF ATOMS":
        raise ValueError("Unexpected dump format: missing NUMBER OF ATOMS block")
    natoms = int(_read_required_line(fh, "Unexpected EOF after NUMBER OF ATOMS").strip())

    box_header = _read_required_line(fh, "Unexpected EOF before BOX BOUNDS").strip()
    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError("Unexpected dump format: missing BOX BOUNDS block")
    box_lines = [
        _read_required_line(fh, "Unexpected EOF while reading BOX BOUNDS").strip()
        for _ in range(3)
    ]

    atoms_header = _read_required_line(fh, "Unexpected EOF before ATOMS").strip()
    if not atoms_header.startswith("ITEM: ATOMS "):
        raise ValueError("Unexpected dump format: missing ATOMS block")
    atom_lines = [
        _read_required_line(fh, "Unexpected EOF while reading ATOMS lines").strip()
        for _ in range(natoms)
    ]
    return timestep, box_lines, atoms_header.split()[2:], atom_lines


def _count_frames(dump_path: Path) -> int:
    count = 0
    with dump_path.open() as fh:
        while True:
            frame = _read_one_frame(fh)
            if frame is None:
                break
            count += 1
    return count


def read_frame_by_index(dump_path: Path, frame_index: int) -> tuple[int, list[str], list[str], list[str], int]:
    if frame_index < 0:
        nframes = _count_frames(dump_path)
        if nframes == 0:
            raise ValueError(f"No frame found in dump file: {dump_path}")
        target = nframes + frame_index
        if target < 0:
            raise ValueError(
                f"frame_index {frame_index} out of range for {nframes} frames"
            )
    else:
        nframes = -1
        target = frame_index

    with dump_path.open() as fh:
        current = 0
        while True:
            frame = _read_one_frame(fh)
            if frame is None:
                break
            if current == target:
                timestep, box_lines, columns, atom_lines = frame
                if nframes < 0:
                    nframes = current + 1
                return current, box_lines, columns, atom_lines, timestep
            current += 1

    if nframes < 0:
        nframes = current
    raise ValueError(f"frame_index {frame_index} out of range for {nframes} frames")


def write_xsf(
    output_path: Path,
    avec: list[float],
    bvec: list[float],
    cvec: list[float],
    atoms: list[tuple[int, float, float, float, float, float, float]],
) -> None:
    with output_path.open("w") as fh:
        fh.write("XSF file generated from LAMMPS dump\n")
        fh.write("CRYSTAL\n")
        fh.write("PRIMVEC\n")
        fh.write(f"{avec[0]:22.16f} {avec[1]:22.16f} {avec[2]:22.16f}\n")
        fh.write(f"{bvec[0]:22.16f} {bvec[1]:22.16f} {bvec[2]:22.16f}\n")
        fh.write(f"{cvec[0]:22.16f} {cvec[1]:22.16f} {cvec[2]:22.16f}\n")
        fh.write("PRIMCOORD\n")
        fh.write(f"{len(atoms)}\n")
        for atomic_number, x, y, z, sx, sy, sz in atoms:
            fh.write(
                f"{atomic_number:d} "
                f"{x:.10f} {y:.10f} {z:.10f} "
                f"{sx:.10f} {sy:.10f} {sz:.10f}\n"
            )


def read_xsf_spin(fname: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with fname.open() as fh:
        for line in fh:
            fields = line.split()
            if len(fields) != 7:
                continue
            try:
                rows.append([float(x) for x in fields])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No spin rows found in XSF file: {fname}")

    data = np.array(rows)
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    sx, sy, sz = data[:, 4], data[:, 5], data[:, 6]
    return x, y, z, sx, sy, sz


def split_layers(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z0 = 0.5 * (z.min() + z.max())
    # z0 = 0.2
    up = z > z0
    dn = z <= z0
    return up, dn


def select_color_component(
    component: str,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
) -> tuple[np.ndarray, str]:
    mapping = {
        "x": (sx, r"$S_x$"),
        "y": (sy, r"$S_y$"),
        "z": (sz, r"$S_z$"),
    }
    return mapping[component]


def plot_layer(ax, x, y, sx, sy, color_values, title):
    tri = mtri.Triangulation(x, y)

    tcf = ax.tripcolor(tri, color_values, shading="gouraud", cmap="cividis", vmin=-1, vmax=1)
    ax.quiver(
        x,
        y,
        sx,
        sy,
        color="k",
        angles="xy",
        scale_units="xy",
        scale=0.35,
        width=0.0025,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
    )

    ax.set_title(title, fontsize=18, weight="bold")
    ax.set_xlabel(r"L$_x$ ($\mathrm{\AA}$)", fontsize=16)
    ax.set_ylabel(r"L$_y$ ($\mathrm{\AA}$)", fontsize=16)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    return tcf


def plot_spin_texture(
    xsf_path: Path,
    output_prefix: Path | None = None,
    color_component: str = "x",
) -> None:
    x, y, z, sx, sy, sz = read_xsf_spin(xsf_path)
    up, dn = split_layers(z)
    color_values, color_label = select_color_component(color_component, sx, sy, sz)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    mappable = plot_layer(axs[0], x[up], y[up], sx[up], sy[up], color_values[up], "Top Layer")
    mappable = plot_layer(axs[1], x[dn], y[dn], sx[dn], sy[dn], color_values[dn], "Bottom Layer")

    cbar = fig.colorbar(mappable, ax=axs, shrink=0.85, pad=0.03)
    cbar.set_label(color_label, fontsize=18)
    cbar.ax.tick_params(labelsize=13)

    prefix = output_prefix if output_prefix is not None else xsf_path.with_suffix("")
    plt.savefig(f"{prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{prefix}.eps", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    selected_idx, box_lines, columns, atom_lines, timestep = read_frame_by_index(
        args.input_dump,
        args.frame_index,
    )
    avec, bvec, cvec = parse_box_bounds(box_lines)

    col_index = {name: idx for idx, name in enumerate(columns)}
    required = ["type", "x", "y", "z", "c_outsp[1]", "c_outsp[2]", "c_outsp[3]"]
    missing = [name for name in required if name not in col_index]
    if missing:
        raise ValueError(f"Missing required dump columns: {', '.join(missing)}")

    magnetic_raw: list[tuple[int, float, float, float, float, float, float]] = []
    for line in atom_lines:
        parts = line.split()
        if abs(float(parts[-1])) == 0.0:
            continue
        magnetic_raw.append(
            (
                int(parts[col_index["type"]]),
                float(parts[col_index["x"]]),
                float(parts[col_index["y"]]),
                float(parts[col_index["z"]]),
                float(parts[col_index["c_outsp[1]"]]),
                float(parts[col_index["c_outsp[2]"]]),
                float(parts[col_index["c_outsp[3]"]]),
            )
        )

    magnetic_types = sorted({atom_type for atom_type, *_ in magnetic_raw})
    type_to_atomic_number = {
        atom_type: idx + 25 for idx, atom_type in enumerate(magnetic_types)
    }

    atoms = [
        (type_to_atomic_number[atom_type], x, y, z, sx, sy, sz)
        for atom_type, x, y, z, sx, sy, sz in magnetic_raw
    ]

    write_xsf(args.output_xsf, avec, bvec, cvec, atoms)
    print(f"[INFO] Extracted frame index: {selected_idx}")
    print(f"[INFO] Extracted timestep   : {timestep}")
    print(f"[INFO] Output XSF          : {args.output_xsf}")
    if args.plot:
        plot_spin_texture(
            args.output_xsf,
            args.plot_prefix,
            color_component=args.color_component,
        )

if __name__ == "__main__":
    main()
