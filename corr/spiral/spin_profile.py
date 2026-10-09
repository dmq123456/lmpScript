#!/usr/bin/env python3
"""Average a spin texture along its own modulation direction.

    spin_profile.py neg-hold-test.lammpstrj --frame 10 --supercell 30 30 1

In a triclinic cell the stripes of a spiral do not run along a Cartesian axis,
so averaging over y smears them out. The direction to average along is set by
the modulation wavevector, which this script measures rather than assumes: the
magnetic atoms sit on an integer lattice, a 2D FFT over that lattice gives the
dominant q, and the profile is then taken along the matching k.

Two conventions make separate frames comparable:

* The sign of q is fixed by requiring its first non-zero index to be positive.
  A real field always has equal power at +q and -q, so which one `argmax`
  returns is a coin toss; left unfixed, a left-handed spiral read along -k is
  indistinguishable from a right-handed one read along +k.

* Atoms are grouped by exact phase class rather than binned by distance. The
  phase of site (i1,i2) is 2*pi*(m1*i1/N1 + m2*i2/N2), which takes only
  N/gcd distinct values, so every atom lands in a class with no rounding and
  every class holds the same number of atoms.

The profile is periodic in the wavelength by construction; it is drawn across
the full box so the spatial scale is visible, but the repeats carry no new
information.
"""

from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def read_frame(path: Path, wanted: int):
    """Return (timestep, cell, columns, rows) for one frame."""
    index = 0
    with path.open() as fh:
        while True:
            line = fh.readline()
            if not line:
                raise SystemExit(
                    f"error: frame {wanted} not found; the dump holds {index} frame(s)"
                )
            if line.strip() != "ITEM: TIMESTEP":
                raise SystemExit(f"error: expected 'ITEM: TIMESTEP', got {line.strip()!r}")
            timestep = int(fh.readline().strip())
            fh.readline()
            natoms = int(fh.readline().strip())
            box_header = fh.readline().strip()
            box_lines = [fh.readline().strip() for _ in range(3)]
            columns = fh.readline().split()[2:]
            rows = [fh.readline().split() for _ in range(natoms)]
            if index == wanted:
                return timestep, parse_box(box_lines, box_header), columns, rows
            index += 1


def parse_box(box_lines: list[str], header: str) -> np.ndarray:
    """Undo the bounding box a triclinic dump reports, back to cell vectors."""
    values = [[float(v) for v in line.split()] for line in box_lines]
    if "xy" in header:
        (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo, zhi, yz) = values
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
    else:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = values
        xy = xz = yz = 0.0
    return np.array([[xhi - xlo, 0.0, 0.0],
                     [xy, yhi - ylo, 0.0],
                     [xz, yz, zhi - zlo]])


# ----------------------------------------------------------------------
# The modulation
# ----------------------------------------------------------------------
def canonical(m1: int, m2: int, n1: int, n2: int) -> tuple[int, int]:
    """Fold to the first Brillouin zone and fix the sign.

    A real field gives |F(q)| = |F(-q)| exactly, so the two peaks are
    degenerate and `argmax` picks between them arbitrarily. Requiring the first
    non-zero index to be positive makes the choice depend only on the index, so
    the same physical modulation always yields the same k direction -- which is
    what lets chirality be compared between frames at all.
    """
    a = m1 - n1 if m1 > n1 // 2 else m1
    b = m2 - n2 if m2 > n2 // 2 else m2
    if a < 0 or (a == 0 and b < 0):
        a, b = -a, -b
    return a, b


def find_modulation(grid: np.ndarray) -> tuple[tuple[int, int], float, np.ndarray]:
    """Dominant wavevector of a spin field on an (N1,N2,3) lattice.

    Returns the canonical index pair, the fraction of the total modulated power
    it carries, and the power spectrum.
    """
    n1, n2 = grid.shape[:2]
    spectrum = np.fft.fft2(grid, axes=(0, 1))
    power = (np.abs(spectrum) ** 2).sum(axis=-1)
    power[0, 0] = 0.0                      # the uniform component is not a modulation
    flat = int(power.argmax())
    peak = np.unravel_index(flat, power.shape)
    # +q and -q are the same modulation, so the pair carries twice the entry.
    fraction = 2.0 * power[peak] / power.sum()
    return canonical(peak[0], peak[1], n1, n2), fraction, power


def peak_table(power: np.ndarray, count: int = 4) -> list[tuple[tuple[int, int], float]]:
    """The strongest few modulations, with Hermitian duplicates removed."""
    n1, n2 = power.shape
    total = power.sum()
    seen: list[tuple[int, int]] = []
    out = []
    for flat in np.argsort(power.ravel())[::-1]:
        a, b = np.unravel_index(int(flat), power.shape)
        key = canonical(a, b, n1, n2)
        if key in seen:
            continue
        seen.append(key)
        out.append((key, 2.0 * power[a, b] / total))
        if len(out) == count:
            break
    return out


# ----------------------------------------------------------------------
# The profile
# ----------------------------------------------------------------------
def wavefront_index(i1, i2, m1, m2, n1, n2):
    """Which wavefront each site sits on, and how many fit in one wavelength.

    The phase is 2*pi*(m1*i1/N1 + m2*i2/N2). Over a common denominator and with
    the common factor divided out it becomes 2*pi*n/M for an integer n, so every
    site falls on the wavefront numbered n, at a distance n*lambda/M along k.

    Taking n modulo M folds every period onto one; leaving it alone keeps the
    wavefronts where they are, which is the difference between averaging the
    whole box into a single period and reading the texture off position by
    position.
    """
    lcm = n1 * n2 // gcd(n1, n2)
    numerator = m1 * i1 * (lcm // n1) + m2 * i2 * (lcm // n2)
    divisor = gcd(gcd(abs(m1) * (lcm // n1), abs(m2) * (lcm // n2)), lcm)
    per_period = lcm // divisor
    # Translating by a lattice vector shifts n by m1*lcm/divisor or m2*lcm/divisor,
    # so wavefronts repeat -- with the very same atoms on them -- every
    # M*gcd(|m1|,|m2|) steps. That is as far as the texture can be unfolded
    # before it starts repeating itself.
    translation_period = per_period * gcd(abs(m1), abs(m2))
    return numerator // divisor, per_period, translation_period


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spin_profile.py",
        description="Average sx, sy, sz along the modulation direction of a spin texture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dump", type=Path, help="Input dump")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to analyse")
    parser.add_argument("--supercell", type=int, nargs=3, metavar=("N1", "N2", "N3"),
                        default=(30, 30, 1),
                        help="Cell repeats; the primitive cell is the box divided by these")
    parser.add_argument("--spin-columns", nargs=3, default=["c_outsp[1]", "c_outsp[2]", "c_outsp[3]"],
                        metavar=("SX", "SY", "SZ"), help="Dump columns holding the spin")
    parser.add_argument("--type", type=int, default=None,
                        help="Atom type to use; default is every atom with a non-zero spin")
    parser.add_argument("--q", type=int, nargs=2, default=None, metavar=("M1", "M2"),
                        help="Force the modulation indices instead of measuring them")
    parser.add_argument("--fold", action=argparse.BooleanOptionalAction, default=True,
                        help="Fold every period onto one before averaging, which assumes the "
                             "modulation is the same everywhere. --no-fold unfolds as far as "
                             "the lattice allows, so wavefronts one wavelength apart are "
                             "measured on different atoms and a texture that varies across "
                             "the box shows up; each point then carries proportionally "
                             "fewer atoms")
    parser.add_argument("--basis", choices=["xyz", "k"], default="xyz",
                        help="Components to plot. 'xyz' is the Cartesian frame; 'k' "
                             "resolves the spin along the modulation direction instead, "
                             "into s_par (along k), s_perp (in-plane, across k) and s_z. "
                             "That basis separates a cycloid, which turns in the plane "
                             "containing k and z, from a proper screw, which turns in the "
                             "plane across k")
    parser.add_argument("--xmax", type=float, default=None,
                        help="Extent of the horizontal axis in Angstrom; default is the box lx")
    parser.add_argument("--out", type=Path, default=Path("spin_profile.png"), help="Figure")
    parser.add_argument("--data", type=Path, default=None, help="Also write the profile as .csv")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not args.dump.exists():
        sys.exit(f"error: no such file: {args.dump}")

    timestep, cell, columns, rows = read_frame(args.dump, args.frame)
    index = {name: i for i, name in enumerate(columns)}
    for name in ("x", "y", "z", *args.spin_columns):
        if name not in index:
            sys.exit(f"error: dump has no '{name}' column; present: {' '.join(columns)}")

    positions = np.array([[float(r[index[c]]) for c in ("x", "y", "z")] for r in rows])
    spins = np.array([[float(r[index[c]]) for c in args.spin_columns] for r in rows])

    if args.type is not None:
        if "type" not in index:
            sys.exit("error: --type given but the dump has no 'type' column")
        keep = np.array([int(r[index["type"]]) == args.type for r in rows])
    else:
        keep = np.linalg.norm(spins, axis=1) > 1e-8
    if not keep.any():
        sys.exit("error: no magnetic atoms selected")

    supercell = np.asarray(args.supercell, dtype=int)
    primitive = cell / supercell[:, None]
    fractional = positions[keep] @ np.linalg.inv(primitive)
    i1 = np.rint(fractional[:, 0]).astype(int) % supercell[0]
    i2 = np.rint(fractional[:, 1]).astype(int) % supercell[1]

    offset = np.abs(fractional[:, :2] - np.rint(fractional[:, :2])).max()
    sites = len(set(zip(i1.tolist(), i2.tolist())))
    if sites != keep.sum():
        sys.exit(
            f"error: {keep.sum()} magnetic atoms map onto only {sites} lattice sites.\n"
            f"       Check --supercell; the largest offset from a site is {offset:.3f} cells."
        )

    grid = np.zeros((supercell[0], supercell[1], 3))
    grid[i1, i2] = spins[keep]
    measured, fraction, power = find_modulation(grid)
    m1, m2 = args.q if args.q else measured
    if args.q:
        m1, m2 = canonical(m1 % supercell[0], m2 % supercell[1], supercell[0], supercell[1])

    reciprocal = 2.0 * np.pi * np.linalg.inv(primitive).T
    q = np.array([m1 / supercell[0], m2 / supercell[1], 0.0])
    k = q @ reciprocal
    wavelength = 2.0 * np.pi / np.linalg.norm(k)

    vectors = spins[keep]
    if args.basis == "k":
        k_hat = k / np.linalg.norm(k)
        if abs(k_hat[2]) > 1e-6:
            print(f"[WARN] k has an out-of-plane component ({k_hat[2]:+.4f}); the k-basis "
                  f"assumes k lies in the xy plane, so s_par and s_z are not orthogonal here.")
        perp = np.cross([0.0, 0.0, 1.0], k_hat)
        perp /= np.linalg.norm(perp)
        basis = np.array([k_hat, perp, [0.0, 0.0, 1.0]])
        vectors = vectors @ basis.T
        print(f"[INFO] Basis          : k-frame   "
              f"par={np.round(k_hat, 4)}  perp={np.round(perp, 4)}  z=[0 0 1]")

    fronts, per_period, translation_period = wavefront_index(
        i1, i2, m1, m2, supercell[0], supercell[1]
    )
    # Reducing modulo the translation period costs nothing: wavefronts that
    # differ by it hold the same atoms, reached through a lattice vector. It is
    # also what removes the sparse ends, since those were only sparse because
    # the box cut their lines short.
    period = per_period if args.fold else translation_period
    fronts = fronts % period
    levels = np.arange(period)
    counts = np.array([int((fronts == n).sum()) for n in levels])
    profile = np.array([vectors[fronts == n].mean(axis=0) for n in levels])
    coherence = np.linalg.norm(profile, axis=1)

    print(f"[INFO] Frame          : {args.frame}  (timestep {timestep})")
    print(f"[INFO] Magnetic atoms : {int(keep.sum())} on a "
          f"{supercell[0]}x{supercell[1]} lattice, max offset {offset:.4f} cells")
    print(f"[INFO] q (fractional) : ({m1:+d},{m2:+d})/({supercell[0]},{supercell[1]})"
          f" = ({q[0]:+.4f}, {q[1]:+.4f}, 0)" + ("   [forced]" if args.q else ""))
    print(f"[INFO] k (Cartesian)  : ({k[0]:+.5f}, {k[1]:+.5f}, {k[2]:+.5f}) 1/A")
    print(f"[INFO] Wavelength     : {wavelength:.4f} A")
    print(f"[INFO] Main peak      : {100 * fraction:.1f}% of the modulated power")
    for (a, b), frac in peak_table(power):
        print(f"[INFO]   ({a:+d},{b:+d}) : {100 * frac:5.1f}%")
    print(f"[INFO] Averaging      : {'folded' if args.fold else 'local'}, {period} points "
          f"of {counts.min()}-{counts.max()} atoms, spacing "
          f"{wavelength / per_period:.3f} A, repeats every "
          f"{period * wavelength / per_period:.3f} A "
          f"({period / per_period:g} wavelength(s))")
    print(f"[INFO] Coherence      : {coherence.mean():.4f}  "
          f"(1 means the spins within a class all point the same way)")

    step = wavelength / per_period
    xmax = args.xmax if args.xmax is not None else cell[0, 0]
    n_points = int(np.ceil(xmax / step)) + 1
    distance = np.arange(n_points) * step
    tiled = profile[np.arange(n_points) % period]
    xlim = (0.0, xmax)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    labels = ([(r"$\langle s_x\rangle$", "#999999"),
               (r"$\langle s_y\rangle$", "#1f77b4"),
               (r"$\langle s_z\rangle$", "#d62728")]
              if args.basis == "xyz" else
              [(r"$\langle s_\parallel\rangle$", "#999999"),
               (r"$\langle s_\perp\rangle$", "#1f77b4"),
               (r"$\langle s_z\rangle$", "#d62728")])
    for column, (label, colour) in enumerate(labels):
        ax.plot(distance, tiled[:, column], "-", color=colour, lw=2.0, label=label)
    for edge in np.arange(np.floor(xlim[0] / wavelength) * wavelength,
                          xlim[1] + wavelength, wavelength):
        ax.axvline(edge, color="k", ls=":", lw=0.7, alpha=0.5)

    ax.set_xlim(*xlim)
    ax.set_xlabel(r"distance along $+\mathbf{k}$   ($\mathrm{\AA}$)", fontsize=13)
    ax.set_ylabel(r"$\langle s_\alpha\rangle$", fontsize=14)
    ax.set_title(
        f"frame {args.frame} (timestep {timestep})    "
        f"q = ({m1:+d},{m2:+d})/({supercell[0]},{supercell[1]})    "
        rf"$\lambda$ = {wavelength:.2f} $\mathrm{{\AA}}$    "
        f"main peak {100 * fraction:.1f}%    "
        f"{'folded' if args.fold else 'local'}",
        fontsize=12, weight="bold",
    )
    ax.grid(alpha=0.3, ls="--")
    ax.legend(ncol=3, frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"[INFO] Figure         : {args.out}")

    if args.data is not None:
        names = ("sx,sy,sz" if args.basis == "xyz" else "s_par,s_perp,s_z")
        header = f"distance_A,{names},coherence,count\n"
        with args.data.open("w") as fh:
            fh.write(header)
            for j, n in enumerate(levels):
                fh.write(f"{n * step:.6f},{profile[j, 0]:.6f},{profile[j, 1]:.6f},"
                         f"{profile[j, 2]:.6f},{coherence[j]:.6f},{counts[j]}\n")
        print(f"[INFO] Data           : {args.data}  ({len(levels)} independent points)")


if __name__ == "__main__":
    main()
