#!/usr/bin/env python3
"""Profile a spin spiral along its own modulation direction, and draw its plane.

    spin_spiral.py pos-test.lammpstrj --supercell 30 30 1 --basis k --plane

In a triclinic cell the stripes of a spiral do not run along a Cartesian axis,
so averaging over y smears them out. The direction to average along is set by
the modulation wavevector, which this script measures rather than assumes: the
magnetic atoms sit on an integer lattice, a 2D FFT over that lattice gives the
dominant q, and the profile is then taken along the matching k.

The sign of q is fixed by requiring its first non-zero index to be positive. A
real field always has equal power at +q and -q, so which one `argmax` returns is
a coin toss; left unfixed, a left-handed spiral read along -k is
indistinguishable from a right-handed one read along +k.

<s_alpha> is averaged in one of two ways (--mode):

  phase  Atoms are grouped by exact phase class. The phase of site (i1,i2) is
         2*pi*(m1*i1/N1 + m2*i2/N2), which takes only N/gcd distinct values, so
         every atom lands in a class with no rounding and every class holds the
         same number of atoms. Classes are unfolded as far as the lattice
         allows -- wavefronts one wavelength apart are then measured on
         different atoms, so a texture that varies across the box shows up.

  bin    Atoms are binned by their projected distance along +k with a fixed bin
         width. Since q.r = |q| * (r.k_hat) the phase depends only on that
         distance, so a bin is a set of near-constant phase; the bins just hold
         uneven numbers of atoms.

The same Fourier component that locates q also gives the spiral plane: writing
S(r) = u cos(q.r) + v sin(q.r), the plane normal is n = u x v, oriented so that
the spins turn counter-clockwise (right-handed) about n on advancing along +k.
That orientation is what distinguishes +n from -n: with --sense cw the sign is
flipped and the spins turn clockwise about the reported n instead.
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
# The spiral plane
# ----------------------------------------------------------------------
def spiral_plane(grid: np.ndarray, m1: int, m2: int, sense: str):
    """u, v and the oriented normal of S(r) = u cos(q.r) + v sin(q.r).

    numpy's transform carries the opposite phase convention to the one used for
    the wavefronts, F[m] = sum_j s_j exp(-i phi_j), so the component at m is
    C = <s exp(-i phi)> and s_j = 2 Re[C exp(i phi_j)] = u cos(phi_j) + v sin(phi_j).

    Advancing along +k turns S from u towards v, i.e. counter-clockwise about
    u x v. Reporting that vector rather than its negative is what fixes +n
    against -n; --sense cw reverses the convention.
    """
    n1, n2 = grid.shape[:2]
    component = np.fft.fft2(grid, axes=(0, 1))[m1 % n1, m2 % n2] / (n1 * n2)
    u, v = 2.0 * component.real, -2.0 * component.imag
    normal = np.cross(u, v)
    length = np.linalg.norm(normal)
    if length < 1e-12:
        return u, v, None, 0.0
    normal = normal / length
    if sense == "cw":
        normal = -normal
    # ellipticity: the ratio of the two axes of the traced ellipse
    gram = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    eigenvalues = np.sort(np.linalg.eigvalsh(gram))[::-1]
    ellipticity = float(np.sqrt(max(eigenvalues[1], 0.0) / eigenvalues[0]))
    return u, v, normal, ellipticity


# ----------------------------------------------------------------------
# The profile
# ----------------------------------------------------------------------
def wavefront_index(i1, i2, m1, m2, n1, n2):
    """Which wavefront each site sits on, and how many fit in one wavelength.

    The phase is 2*pi*(m1*i1/N1 + m2*i2/N2). Over a common denominator and with
    the common factor divided out it becomes 2*pi*n/M for an integer n, so every
    site falls on the wavefront numbered n, at a distance n*lambda/M along k.

    Under periodic boundaries that index is only defined modulo M*gcd(|m1|,|m2|):
    a site and its periodic image differ by a lattice translation, which shifts
    the raw index by m1*lcm/divisor or m2*lcm/divisor. Reducing by that period
    is therefore free, and it is also what removes the sparse ends, since those
    were only sparse because the box cut their lines short.
    """
    lcm = n1 * n2 // gcd(n1, n2)
    numerator = m1 * i1 * (lcm // n1) + m2 * i2 * (lcm // n2)
    divisor = gcd(gcd(abs(m1) * (lcm // n1), abs(m2) * (lcm // n2)), lcm)
    per_period = lcm // divisor
    translation_period = per_period * gcd(abs(m1), abs(m2))
    return numerator // divisor, per_period, translation_period


def profile_by_phase(vectors, i1, i2, m1, m2, n1, n2, wavelength):
    """Average within exact phase classes, unfolded as far as the lattice allows."""
    fronts, per_period, period = wavefront_index(i1, i2, m1, m2, n1, n2)
    fronts = fronts % period
    levels = np.arange(period)
    counts = np.array([int((fronts == n).sum()) for n in levels])
    if (counts == 0).any():
        raise SystemExit("error: empty phase class; check --supercell and --q")
    values = np.array([vectors[fronts == n].mean(axis=0) for n in levels])
    step = wavelength / per_period
    note = (f"{period} classes of {counts.min()}-{counts.max()} atoms, spacing "
            f"{step:.3f} A, repeats every {period * step:.3f} A "
            f"({period / per_period:g} wavelength(s))")
    return levels * step, values, counts, step, note


def profile_by_bin(vectors, distance, width):
    """Average within fixed-width bins of the projected distance along +k."""
    shifted = distance - distance.min()
    index = np.floor(shifted / width).astype(int)
    filled = np.flatnonzero(np.bincount(index) > 0)
    counts = np.bincount(index)[filled]
    values = np.array([vectors[index == b].mean(axis=0) for b in filled])
    note = (f"{len(filled)} bins of {counts.min()}-{counts.max()} atoms, width "
            f"{width:.3f} A, span {shifted.max():.3f} A")
    return (filled + 0.5) * width, values, counts, width, note


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
LIM, ZLIM, CLIP = 1.30, 1.25, 0.98      # the 3-D box, in units of the spin length
ELEV, AZIM = 14, -148


def _arrow_class():
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    class Arrow3D(FancyArrowPatch):
        """A straight arrow living in data (3-D) coordinates."""

        def __init__(self, start, end, **kw):
            super().__init__((0, 0), (0, 0), **kw)
            self._ends = (np.asarray(start, float), np.asarray(end, float))

        def do_3d_projection(self, renderer=None):
            (x0, y0, z0), (x1, y1, z1) = self._ends
            xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return min(zs)

    return Arrow3D


def plane_surface(ax, normal, plt, cmap="viridis", ngrid=200):
    """The plane through the origin with the given normal, shaded by height."""
    seed = np.array([0.0, 0.0, 1.0])
    if abs(normal @ seed) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    s, t = np.meshgrid(*(np.linspace(-3.2, 3.2, ngrid),) * 2)
    points = s[..., None] * e1 + t[..., None] * e2
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    outside = (np.abs(x) > CLIP) | (np.abs(y) > CLIP) | (np.abs(z) > ZLIM * 0.99)
    x, y, z = [np.where(outside, np.nan, c) for c in (x, y, z)]
    colours = plt.get_cmap(cmap)(plt.Normalize(-ZLIM, ZLIM)(np.nan_to_num(z)))
    colours[..., 3] = 0.75
    ax.plot_surface(x, y, z, facecolors=colours, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=1)


def great_arc(start, end, radius=0.55, points=80):
    """The shorter great-circle arc between two directions."""
    a = np.asarray(start, float) / np.linalg.norm(start)
    b = np.asarray(end, float) / np.linalg.norm(end)
    w = b - (a @ b) * a
    w /= np.linalg.norm(w)
    t = np.linspace(0.0, np.arccos(np.clip(a @ b, -1.0, 1.0)), points)
    return radius * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * w)


def draw_plane(ax, u, v, normal, labels, theta, phi, sense, plt):
    """The spiral plane, the locus of the spin tips and the angles of n."""
    Arrow3D = _arrow_class()

    def arrow(start, end, colour, lw=1.8, ms=12, zorder=5):
        ax.add_artist(Arrow3D(start, end, mutation_scale=ms, lw=lw,
                              arrowstyle="-|>", color=colour, zorder=zorder))

    ax.set_box_aspect((2 * LIM, 2 * LIM, 2 * ZLIM))
    ax.view_init(elev=ELEV, azim=AZIM)

    grid = np.linspace(-1.0, 1.0, 2)                     # the horizontal plane
    xh, yh = np.meshgrid(grid, grid)
    ax.plot_surface(xh, yh, np.zeros_like(xh), color="khaki", alpha=0.5,
                    linewidth=0, shade=False, zorder=0)

    plane_surface(ax, normal, plt)

    angle = np.linspace(0.0, 2 * np.pi, 73)[:-1]         # locus of the spin tips
    locus = np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v
    ax.plot(locus[:, 0], locus[:, 1], locus[:, 2], ls="none", marker="o", ms=4.0,
            color="navy", zorder=6)

    fine_angle = np.linspace(0.0, 2 * np.pi, 400)        # the sense of rotation
    fine = np.cos(fine_angle)[:, None] * u + np.sin(fine_angle)[:, None] * v
    start = int(np.argmax(fine[:, 2]))
    arrow(fine[start], fine[(start + 26) % len(fine)], "0.25", lw=2.4, ms=17, zorder=7)

    for axis, label, offset in (
            (np.array([1.0, 0, 0]) * 1.18, labels[0], (0.30, 0, -0.05)),
            (np.array([0, 1.0, 0]) * 1.18, labels[1], (0, 0.26, -0.05)),
            (np.array([0, 0, 1.0]) * ZLIM * 0.9, labels[2], (0, 0, 0.20))):
        arrow([0, 0, 0], axis, "k", lw=1.7, ms=14, zorder=8)
        ax.text(*(axis + np.array(offset)), label, fontsize=15, fontweight="bold",
                ha="center", va="center", zorder=10)

    arrow([0, 0, 0], 0.92 * normal, "green", lw=2.3, ms=16, zorder=9)
    ax.text(*(1.14 * normal), r"$\mathbf{n}$", fontsize=17, fontweight="bold",
            color="green", ha="center", va="center", zorder=11)
    flat = np.array([normal[0], normal[1], 0.0])
    if np.linalg.norm(flat) > 1e-6:
        ax.plot(*zip(0.92 * normal, 0.92 * flat), ls=":", color="green", lw=1.6, zorder=8)
        ax.plot(*zip([0, 0, 0], 0.92 * flat), ls=":", color="green", lw=1.6, zorder=8)

        azimuth = great_arc([1, 0, 0], flat, radius=0.45)
        ax.plot(*azimuth.T, color="gold", lw=2.4, zorder=9)
        ax.text(*(azimuth[len(azimuth) // 2] * 1.30 + np.array([0, 0, -0.14])),
                r"$\varphi$", fontsize=17, color="goldenrod", fontweight="bold",
                zorder=11)

    polar = great_arc([0, 0, 1], normal, radius=0.60)
    ax.plot(*polar.T, color="0.15", lw=1.7, zorder=9)
    ax.text(*(polar[len(polar) // 2] * 1.28), r"$\theta$", fontsize=17,
            color="0.1", zorder=11)

    ax.text2D(0.60, 0.92, f"$\\theta$ = {theta:.0f}$^\\circ$\n"
                          f"$\\varphi$ = {phi:.0f}$^\\circ$",
              transform=ax.transAxes, fontsize=18, va="top")
    ax.text2D(0.02, 0.14, f"spins turn {sense.upper()} about $\\mathbf{{n}}$\n"
                          r"on advancing along $+\mathbf{k}$",
              transform=ax.transAxes, fontsize=11, color="0.3")

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(-ZLIM, ZLIM)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        limit = ZLIM if axis is ax.zaxis else LIM
        axis.set_ticks(np.linspace(-limit, limit, 5))
        axis.set_ticklabels([])
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.25")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color="0.85", linewidth=0.8)
        axis.line.set_color("0.25")
    ax.tick_params(length=0)


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spin_spiral.py",
        description="Profile a spin spiral along its modulation direction and draw its plane.",
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
    parser.add_argument("--mode", choices=["phase", "bin"], default="phase",
                        help="How <s> is averaged. 'phase' groups atoms into exact phase "
                             "classes, unfolded as far as the lattice allows, so each point "
                             "holds the same number of atoms and no rounding enters. 'bin' "
                             "averages within fixed-width bins of the distance along +k")
    parser.add_argument("--bin", type=float, default=0.5, metavar="WIDTH",
                        help="Bin width in Angstrom, for --mode bin")
    parser.add_argument("--basis", choices=["xyz", "k"], default="k",
                        help="Components to plot. 'xyz' is the Cartesian frame; 'k' "
                             "resolves the spin along the modulation direction instead, "
                             "into s_par (along k), s_perp (in-plane, across k) and s_z. "
                             "That basis separates a cycloid, which turns in the plane "
                             "containing k and z, from a proper screw, which turns in the "
                             "plane across k")
    parser.add_argument("--sense", choices=["ccw", "cw"], default="ccw",
                        help="Orientation of n: the sense in which the spins turn about it "
                             "on advancing along +k. 'ccw' reports n = u x v, the "
                             "right-handed normal; 'cw' reports its negative")
    parser.add_argument("--plane", action="store_true",
                        help="Add the 3-D schematic of the spiral plane below the profile")
    parser.add_argument("--xmax", type=float, default=None,
                        help="Extent of the horizontal axis in Angstrom; default is the "
                             "box lx for --mode phase and the sampled span for --mode bin")
    parser.add_argument("--out", type=Path, default=Path("spin_spiral.png"), help="Figure")
    parser.add_argument("--data", type=Path, default=None,
                        help="Profile as text; default is --out with a .dat suffix, "
                             "'none' to skip")
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

    # ---- the magnetic atoms on their integer lattice -----------------------
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
    q_fractional = np.array([m1 / supercell[0], m2 / supercell[1], 0.0])
    k = q_fractional @ reciprocal
    wavelength = 2.0 * np.pi / np.linalg.norm(k)

    # ---- the plotting basis -------------------------------------------------
    u, v, normal, ellipticity = spiral_plane(grid, m1, m2, args.sense)
    vectors = spins[keep]
    spin_length = float(np.linalg.norm(vectors, axis=1).mean())
    k_hat = k / np.linalg.norm(k)
    if args.basis == "k":
        if abs(k_hat[2]) > 1e-6:
            print(f"[WARN] k has an out-of-plane component ({k_hat[2]:+.4f}); the k-basis "
                  f"assumes k lies in the xy plane, so s_par and s_z are not orthogonal here.")
        perp = np.cross([0.0, 0.0, 1.0], k_hat)
        perp /= np.linalg.norm(perp)
        basis = np.array([k_hat, perp, [0.0, 0.0, 1.0]])
        labels = [r"$\mathbf{S}_{/\!/Q}$", r"$\mathbf{S}_{\perp Q}$", r"$\mathbf{S}_{z}$"]
        names = ("s_par", "s_perp", "s_z")
        print(f"[INFO] Basis          : k-frame   "
              f"par={np.round(k_hat, 4)}  perp={np.round(perp, 4)}  z=[0 0 1]")
    else:
        basis = np.eye(3)
        labels = [r"$\mathbf{S}_{x}$", r"$\mathbf{S}_{y}$", r"$\mathbf{S}_{z}$"]
        names = ("sx", "sy", "sz")
        print("[INFO] Basis          : Cartesian")
    vectors = vectors @ basis.T

    # ---- the profile ---------------------------------------------------------
    if args.mode == "phase":
        distance, values, counts, step, note = profile_by_phase(
            vectors, i1, i2, m1, m2, supercell[0], supercell[1], wavelength)
    else:
        distance, values, counts, step, note = profile_by_bin(
            vectors, positions[keep] @ k_hat, args.bin)
    coherence = np.linalg.norm(values, axis=1) / spin_length

    # ---- report ---------------------------------------------------------------
    print(f"[INFO] Frame          : {args.frame}  (timestep {timestep})")
    print(f"[INFO] Magnetic atoms : {int(keep.sum())} on a "
          f"{supercell[0]}x{supercell[1]} lattice, max offset {offset:.4f} cells")
    s0 = grid[0, 0]
    print(f"[INFO] s0 at site(0,0): ({s0[0]:+.5f}, {s0[1]:+.5f}, {s0[2]:+.5f})"
          f"   [one spin, carries its own thermal noise -- see s0 (fitted) below]")
    print(f"[INFO] q (fractional) : ({m1:+d},{m2:+d})/({supercell[0]},{supercell[1]})"
          f" = ({q_fractional[0]:+.4f}, {q_fractional[1]:+.4f}, 0)"
          + ("   [forced]" if args.q else ""))
    print(f"[INFO] k (Cartesian)  : ({k[0]:+.5f}, {k[1]:+.5f}, {k[2]:+.5f}) 1/A"
          f"   |k| = {np.linalg.norm(k):.5f} 1/A")
    print(f"[INFO] Wavelength     : {wavelength:.4f} A")
    print(f"[INFO] Main peak      : {100 * fraction:.1f}% of the modulated power")
    for (a, b), share in peak_table(power):
        print(f"[INFO]   ({a:+d},{b:+d}) : {100 * share:5.1f}%")
    print(f"[INFO] Averaging      : {args.mode}, {note}")
    print(f"[INFO] Coherence      : {coherence.mean():.4f}  "
          f"(1 means the spins within a class all point the same way)")

    theta = phi = float("nan")
    print(f"[INFO] u (Cartesian)  : ({u[0]:+.5f}, {u[1]:+.5f}, {u[2]:+.5f})")
    print(f"[INFO] v (Cartesian)  : ({v[0]:+.5f}, {v[1]:+.5f}, {v[2]:+.5f})")
    if normal is None:
        print("[WARN] u x v vanishes: the texture is a collinear wave, not a spiral; "
              "n is undefined.")
    else:
        normal_basis = basis @ normal
        theta = float(np.degrees(np.arccos(np.clip(normal_basis[2], -1.0, 1.0))))
        phi = float(np.degrees(np.arctan2(normal_basis[1], normal_basis[0])))
        print(f"[INFO] n (Cartesian)  : ({normal[0]:+.5f}, {normal[1]:+.5f}, {normal[2]:+.5f})"
              f"   [spins turn {args.sense} about n along +k]")

        # The spiral's own phase reference, which is what reconstructing the
        # texture needs. The transform puts phase zero at site (0,0), so the
        # modulated part contributes u there; the cone contributes the uniform
        # magnetisation along n, which is a separate, q = 0 quantity. Both are
        # averages over every site, so their noise falls as 1/sqrt(N) instead of
        # being one spin's.
        #
        # This is why s0 must not be read off site (0,0). A single spin's tilt
        # out of the plane is thermal -- 7 to 9 degrees rms here -- and
        # make_spiral.py faithfully turns that tilt into a cone. The cone is a
        # real degree of freedom and is fitted rather than assumed away, but it
        # has to be measured from the mean of N spins, where the same thermal
        # spread is suppressed by sqrt(N).
        flat = grid.reshape(-1, 3)
        cone = float(np.dot(flat.mean(axis=0), normal))
        cone_error = float(np.std(flat @ normal, ddof=1)) / np.sqrt(flat.shape[0])
        s0_fitted = u + normal * cone
        s0_fitted /= np.linalg.norm(s0_fitted)
        print(f"[INFO] s0 (fitted)    : ({s0_fitted[0]:+.5f}, {s0_fitted[1]:+.5f}, "
              f"{s0_fitted[2]:+.5f})   [use this for make_spiral.py --s0]")
        print(f"[INFO] Cone           : <S.n> = {cone:+.5f} +- {cone_error:.5f}"
              f"   ({np.degrees(np.arcsin(min(abs(cone), 1.0))):.2f} deg from the plane, "
              f"{abs(cone) / cone_error:.1f} sigma)")
        if abs(cone) <= 3.0 * cone_error:
            print("[INFO]   consistent with zero, so the spiral is flat, not conical.")
        tilt = float(np.degrees(np.arcsin(
            abs(float(np.dot(normal, s0 / np.linalg.norm(s0)))))))
        if tilt > 3.0 * max(np.degrees(np.arcsin(min(abs(cone), 1.0))), 1.0):
            print(f"[WARN] s0 at site(0,0) is {tilt:.1f} deg out of the plane, far more than the "
                  f"fitted cone.")
            print(f"[WARN]   That is one spin's thermal noise, not structure -- use s0 (fitted).")
        if args.basis == "k":
            print(f"[INFO] n ({names[0]},{names[1]},{names[2]}) : "
                  f"({normal_basis[0]:+.5f}, {normal_basis[1]:+.5f}, {normal_basis[2]:+.5f})")
        print(f"[INFO] n angles       : theta = {theta:.1f} deg (from {names[2]}), "
              f"phi = {phi:.1f} deg (from {names[0]} towards {names[1]})")
        print(f"[INFO] Ellipticity    : {ellipticity:.4f}  (1 is a circular spiral, "
              f"0 a collinear wave)")
        if ellipticity < 0.1:
            print("[WARN] the spiral is nearly collinear; the plane is poorly defined.")

    # ---- the figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xmax = args.xmax
    if xmax is None:
        xmax = cell[0, 0] if args.mode == "phase" else distance.max()
    if args.mode == "phase":                 # the profile is periodic by construction
        points = int(np.ceil(xmax / step)) + 1
        x_plot = np.arange(points) * step
        y_plot = values[np.arange(points) % len(values)]
    else:
        x_plot, y_plot = distance, values

    if args.plane:
        figure = plt.figure(figsize=(9.0, 10.2), dpi=160)
        ax = figure.add_axes([0.10, 0.790, 0.86, 0.165])
        ax3d = figure.add_axes([0.02, -0.02, 0.96, 0.79], projection="3d")
        ax3d.patch.set_alpha(0.0)
    else:
        figure, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
        ax3d = None

    colours = ("#999999", "#1f77b4", "#d62728")
    tex = ((r"$\langle s_x\rangle$", r"$\langle s_y\rangle$", r"$\langle s_z\rangle$")
           if args.basis == "xyz" else
           (r"$\langle s_\parallel\rangle$", r"$\langle s_\perp\rangle$", r"$\langle s_z\rangle$"))
    for column in range(3):
        ax.plot(x_plot, y_plot[:, column], "-", color=colours[column], lw=2.0,
                label=tex[column])
    for edge in np.arange(0.0, xmax + wavelength, wavelength):
        ax.axvline(edge, color="k", ls=":", lw=0.7, alpha=0.5)

    ax.set_xlim(0.0, xmax)
    ax.set_xlabel(r"distance along $+\mathbf{k}$   ($\mathrm{\AA}$)", fontsize=13)
    ax.set_ylabel(r"$\langle s_\alpha\rangle$", fontsize=14)
    ax.set_title(
        f"frame {args.frame} (timestep {timestep})    "
        f"q = ({m1:+d},{m2:+d})/({supercell[0]},{supercell[1]})    "
        rf"$\lambda$ = {wavelength:.2f} $\mathrm{{\AA}}$    "
        f"main peak {100 * fraction:.1f}%    {args.mode}",
        fontsize=11, weight="bold",
    )
    ax.grid(alpha=0.3, ls="--")
    ax.legend(ncol=3, frameon=False, fontsize=12, loc="lower right")

    if ax3d is not None:
        if normal is None:
            ax3d.set_axis_off()
            ax3d.text2D(0.5, 0.5, "no spiral plane:\nthe texture is collinear",
                        transform=ax3d.transAxes, ha="center", fontsize=14, color="0.3")
        else:
            draw_plane(ax3d, (basis @ u) / spin_length, (basis @ v) / spin_length,
                       basis @ normal, labels, theta, phi, args.sense, plt)
    else:
        figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    print(f"[INFO] Figure         : {args.out}")

    # ---- the data --------------------------------------------------------------
    if args.data is None or str(args.data).lower() != "none":
        path = args.out.with_suffix(".dat") if args.data is None else args.data
        with Path(path).open("w") as fh:
            fh.write(f"# frame {args.frame}  timestep {timestep}\n")
            fh.write(f"# q_fractional ({m1:+d},{m2:+d})/({supercell[0]},{supercell[1]})\n")
            fh.write(f"# k_cartesian {k[0]:+.8f} {k[1]:+.8f} {k[2]:+.8f}  1/A\n")
            fh.write(f"# wavelength {wavelength:.6f}  A\n")
            fh.write(f"# main_peak {fraction:.6f}\n")
            fh.write(f"# basis {args.basis}   mode {args.mode}   sense {args.sense}\n")
            if normal is not None:
                fh.write(f"# n_cartesian {normal[0]:+.8f} {normal[1]:+.8f} {normal[2]:+.8f}"
                         f"   (spins turn {args.sense} about n along +k)\n")
                fh.write(f"# n_basis {normal_basis[0]:+.8f} {normal_basis[1]:+.8f} "
                         f"{normal_basis[2]:+.8f}\n")
                fh.write(f"# theta_deg {theta:.4f}   phi_deg {phi:.4f}\n")
                fh.write(f"# ellipticity {ellipticity:.6f}\n")
            fh.write(f"# distance_A {names[0]} {names[1]} {names[2]} coherence count\n")
            for j in range(len(distance)):
                fh.write(f"{distance[j]:12.6f} {values[j, 0]:+12.6f} {values[j, 1]:+12.6f} "
                         f"{values[j, 2]:+12.6f} {coherence[j]:10.6f} {counts[j]:6d}\n")
        print(f"[INFO] Data           : {path}  ({len(distance)} independent points)")


if __name__ == "__main__":
    main()
