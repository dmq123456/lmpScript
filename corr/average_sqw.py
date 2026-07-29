#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Average S(q,w) over several runs and plot the result.

    average_sqw.py run1.npz run2.npz run3.npz --plot --plot-file mean.png

Averaging is the correct cure for the statistical scatter of a spectrum
estimated from molecular dynamics. A single trajectory is one realisation of a
thermal random process, and the estimate at each frequency bin carries a
scatter of order the value itself -- roughly 100% -- no matter how long the run.
Lengthening a trajectory buys a finer frequency grid, not a better value per
bin. Averaging M independent estimates is what reduces the scatter, as
1/sqrt(M).

The script refuses to combine incompatible files: the q-path, the frequency
grid and the channel must all match, since averaging across different channels
or grids is meaningless rather than merely inaccurate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot import plot_sqw


# Keys that must agree across inputs before averaging is meaningful.
_MUST_MATCH = ("freq_thz", "q_vectors", "components")

# Keys carried through from the first input unchanged.
_METADATA = (
    "timesteps", "q_frac", "q_vectors", "freq_thz", "energy_meV",
    "components", "field_columns", "lattice", "reciprocal_lattice",
    "translation_repeats", "dt_fs", "corr_norm",
    "q_node_indices", "q_path_distance",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="average_sqw.py",
        description="Average S(q,w) across npz files written by sqw_spin_corr.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("npz", nargs="+", type=Path, help="Input .npz files (shell globs work)")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the averaged result to this .npz")
    p.add_argument("--average-corr-plus", action="store_true",
                   help="Also average corr_plus, when every input carries it. This is "
                        "what you want before fitting a decay envelope, but the array "
                        "is large -- check the reported size first")
    p.add_argument("--plot", action="store_true", help="Render the averaged intensity map")
    p.add_argument("--plot-file", type=Path, default=Path("sqw_mean.png"),
                   help="Figure output path")
    p.add_argument("--mev", action="store_true", help="Label the energy axis in meV")
    p.add_argument("--max-freq-thz", type=float, default=None, help="Frequency cutoff for the plot")
    p.add_argument("--cbar-min", type=float, default=None, help="Colorbar minimum")
    p.add_argument("--cbar-max", type=float, default=None, help="Colorbar maximum")
    p.add_argument("--title", default=None, help="Plot title (default: derived from the channel)")
    return p


def _describe_mismatch(key: str, a, b, first: Path, other: Path) -> str:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return (f"{other.name}: '{key}' has shape {b.shape}, but {first.name} has {a.shape}. "
                f"Different q-paths or frame counts cannot be averaged.")
    return (f"{other.name}: '{key}' differs in value from {first.name}. "
            f"The runs were not produced with the same settings.")


def load_and_check(paths: list[Path]) -> tuple[list[np.ndarray], dict, list[np.ndarray] | None]:
    """Load every input, verify compatibility, return the sqw arrays."""
    first_data = None
    first_path = paths[0]
    sqw_list: list[np.ndarray] = []
    corr_list: list[np.ndarray] = []
    have_corr = True

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        data = np.load(path, allow_pickle=True)
        if "sqw" not in data.files:
            raise ValueError(f"{path.name}: no 'sqw' array -- is this an sqw_spin_corr output?")

        if first_data is None:
            first_data = data
        else:
            for key in _MUST_MATCH:
                if key not in data.files or key not in first_data.files:
                    continue
                a, b = np.asarray(first_data[key]), np.asarray(data[key])
                same = (a.shape == b.shape) and (
                    np.array_equal(a, b) if a.dtype.kind in "US O" else np.allclose(a, b)
                )
                if not same:
                    raise ValueError(_describe_mismatch(key, a, b, first_path, path))

        sqw_list.append(np.asarray(data["sqw"], dtype=np.float64))
        if "corr_plus" in data.files:
            corr_list.append(np.asarray(data["corr_plus"]))
        else:
            have_corr = False

    meta = {k: first_data[k] for k in _METADATA if k in first_data.files}
    return sqw_list, meta, (corr_list if have_corr else None)


def report_scatter(sqw_list: list[np.ndarray], mean: np.ndarray) -> None:
    """Show how much the averaging actually bought."""
    m = len(sqw_list)
    if m < 2:
        return
    stack = np.stack(sqw_list)
    # Restrict to bins with real signal; the noise floor would dominate otherwise.
    strong = mean > 0.1 * mean.max()
    if not strong.any():
        return
    per_run = np.median(stack.std(axis=0)[strong] / np.maximum(mean[strong], 1e-300))
    print(f"[INFO] Scatter between runs at strong bins: {per_run:.3f} (relative)")
    print(f"[INFO] Expected reduction from averaging {m} independent runs: "
          f"1/sqrt({m}) = {1/np.sqrt(m):.3f}")
    print(f"[INFO] Residual scatter of the mean:        {per_run/np.sqrt(m):.3f}")
    if per_run < 0.3:
        print("[WARN] The runs scatter far less than the ~1.0 expected of independent")
        print("[WARN] estimates. If these are consecutive windows of one trajectory, they")
        print("[WARN] overlap in time and the averaging gains less than 1/sqrt(M).")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    paths = list(args.npz)
    print(f"[INFO] Averaging {len(paths)} file(s)")
    sqw_list, meta, corr_list = load_and_check(paths)
    for p, s in zip(paths, sqw_list):
        print(f"[INFO]   {p.name}  sqw{s.shape}")

    mean = np.mean(np.stack(sqw_list), axis=0)
    channel = str(meta.get("components", "?"))
    print(f"[INFO] Channel: {channel}   result shape {mean.shape}")
    report_scatter(sqw_list, mean)

    out_arrays = dict(meta)
    out_arrays["sqw"] = mean
    out_arrays["n_averaged"] = np.asarray(len(paths), dtype=np.int64)
    out_arrays["source_files"] = np.asarray([str(p) for p in paths])

    if args.average_corr_plus:
        if corr_list is None:
            raise ValueError(
                "--average-corr-plus given but not every input carries 'corr_plus'. "
                "Re-run those with --save-corr-plus, or drop this flag."
            )
        shapes = {c.shape for c in corr_list}
        if len(shapes) != 1:
            raise ValueError(f"corr_plus shapes differ across inputs: {sorted(shapes)}")
        corr_mean = np.mean(np.stack(corr_list), axis=0)
        out_arrays["corr_plus"] = corr_mean
        print(f"[INFO] Averaged corr_plus{corr_mean.shape}, "
              f"{corr_mean.nbytes/2**20:.0f} MiB")

    if args.output is not None:
        np.savez(args.output, **out_arrays)
        print(f"[INFO] Written: {args.output}")

    if args.plot:
        title = args.title or f"S(q,w)  component={channel}  (mean of {len(paths)})"
        plot_sqw(
            q_vectors=np.asarray(meta["q_vectors"]),
            freq_thz=np.asarray(meta["freq_thz"]),
            sqw=mean,
            outfile=str(args.plot_file),
            max_freq_thz=args.max_freq_thz,
            cbar_min=args.cbar_min,
            cbar_max=args.cbar_max,
            use_meV=args.mev,
            title=title,
            q_node_indices=meta.get("q_node_indices"),
            q_node_labels=None,
        )


if __name__ == "__main__":
    main()
