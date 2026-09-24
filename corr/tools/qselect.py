#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How completely one spiral domain has been selected, per frame.

The ground state of a triangular-lattice helimagnet carries three symmetry-
equivalent propagation vectors at 120 degrees. A field can leave only one of
them standing, and what this module measures is how far along that is: one
number per frame, zero when all three are equally present and one when a single
domain holds the whole ring.

    A_m = sum_{q in ring} S(q) exp(-i m theta_q),    eta = |A_2| / A_0

with the sum over the allowed q whose length is near |Q|. Two facts make m = 2
the right harmonic and make eta a genuine order parameter rather than a fitted
score.

Because a spiral puts weight at +Q and -Q alike, S(theta) has period 180
degrees and only even m survive. The three domain axes are then 60 degrees
apart, so for equal weights the m = 2 terms cancel exactly,

    1 + exp(-i 2 pi/3) + exp(-i 4 pi/3) = 0,

while a single domain leaves |A_2| = A_0 because its two peaks sit 180 degrees
apart and exp(-i 2 pi) = 1. The cancellation is not approximate: on a grid
closed under the three-fold rotation the q in a radial shell fall into complete
C3 orbits, and the identity above kills each orbit term by term. A perfectly
C3-symmetric intensity gives |A_2|/A_0 = 4e-17 here.

The phase comes free and is worth as much as the modulus:

    theta_0 = -arg(A_2) / 2   (mod 180 degrees)

is the axis of the surviving domain as a continuous angle, not a choice among
three labels, so a partially selected state or a drifting domain is tracked
rather than rounded.

Why a ring and not three windows
--------------------------------
Integrating a disc around each expected Q and comparing the three totals needs
a radius to tune, needs the peaks located first, and breaks down when Q is
incommensurate and its weight splits across neighbouring allowed q. The ring
needs none of that: the background is divided out by A_0, and the result is
insensitive to the ring width because the q between the peaks carry almost
nothing. Widening the ring from 5% to 40% of |Q| moves eta by less than 1e-4.

What else comes out
-------------------
`selectivity` also returns the ring radius |Q| as an intensity-weighted first
moment (the pitch, which a field may shift), its rms width (a domain size
2 pi / dq, saturating at the box length), and the longitudinal fraction
S_par/S_tot at the peak, which is sin^2(theta)/2 for the angle theta between
the spiral normal and Q -- zero for a proper screw, one half for a cycloid.
That last one is what distinguishes a chiral selection from the achiral,
n-parallel-to-Q state an alternating field leaves behind.

Run this file directly to execute the self-test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from structurefactor import (  # noqa: E402
    DEFAULT_CHANNEL,
    amplitude,
    average_intensity,
    frame_intensity,
    layer_geometry,
    project,
)

#: Half-width of the ring, as a fraction of |Q|. Not a tuning knob -- see above.
DEFAULT_RING_WIDTH = 0.15

#: Harmonic that distinguishes one domain from three. See the module docstring.
SELECTIVITY_HARMONIC = 2


# ----------------------------------------------------------------------
# The ring
# ----------------------------------------------------------------------
def ring_radius(intensity: np.ndarray, q_grid: np.ndarray) -> float:
    """|Q| of the strongest wavevector other than Gamma.

    Only a starting point for the ring; `selectivity` refines it into a
    weighted first moment once the ring is drawn. Gamma is excluded because a
    ferromagnetic component would otherwise capture the search, and it carries
    no angle anyway.
    """
    length = np.linalg.norm(q_grid, axis=-1)
    off_gamma = length > 1e-9
    if not np.any(off_gamma):
        raise ValueError("Every allowed q is Gamma; there is no ring to measure.")
    masked = np.where(off_gamma, intensity, -np.inf)
    return float(length.ravel()[int(np.argmax(masked))])


def ring_mask(q_grid: np.ndarray, radius: float,
              width: float = DEFAULT_RING_WIDTH) -> np.ndarray:
    """Allowed q within `width`*radius of the ring, Gamma excluded."""
    length = np.linalg.norm(q_grid, axis=-1)
    return (np.abs(length - radius) <= width * radius) & (length > 1e-9)


def harmonic(intensity: np.ndarray, q_grid: np.ndarray, mask: np.ndarray,
             m: int = SELECTIVITY_HARMONIC) -> tuple[float, complex]:
    """(A_0, A_m) over the masked q."""
    angle = np.arctan2(q_grid[..., 1], q_grid[..., 0])
    weights = intensity[mask]
    return float(weights.sum()), complex((weights * np.exp(-1j * m * angle[mask])).sum())


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------
def selectivity(intensity: np.ndarray, q_grid: np.ndarray, *,
                width: float = DEFAULT_RING_WIDTH,
                radius: float | None = None,
                s_par: np.ndarray | None = None) -> dict:
    """Everything one frame has to say about which domain is winning.

    `s_par` is the longitudinal channel on the same grid; when given, the
    longitudinal fraction at the peak is reported too.
    """
    if radius is None:
        radius = ring_radius(intensity, q_grid)
    mask = ring_mask(q_grid, radius, width)
    if not np.any(mask):
        raise ValueError(
            f"No allowed q lies within {width:.0%} of |Q| = {radius:.4f} 1/Angstrom. "
            "Either the supercell is too small to resolve the spiral, or --ring-width "
            "is too narrow."
        )

    a0, a2 = harmonic(intensity, q_grid, mask)
    if a0 <= 0.0:
        raise ValueError("The ring carries no intensity, so selectivity is undefined.")

    # Diffuse scattering is spread over the ring, so it lands in A_0 while
    # averaging to nothing in A_2, and eta reads low for a state that is simply
    # disordered rather than unselected. Subtracting a flat background
    # separates the two: how ordered the texture is, and which domain won.
    # The level is the ring's median, which the handful of Bragg peaks cannot
    # move, so no peak has to be identified first.
    background = float(np.median(intensity[mask]))
    cleaned = np.maximum(intensity - background, 0.0)
    a0_sub, a2_sub = harmonic(cleaned, q_grid, mask)

    length = np.linalg.norm(q_grid, axis=-1)
    weights = intensity[mask]
    q_mean = float((weights * length[mask]).sum() / a0)
    q_rms = float(np.sqrt(max((weights * (length[mask] - q_mean) ** 2).sum() / a0, 0.0)))

    peak = int(np.argmax(np.where(mask, intensity, -np.inf)))
    out = {
        "eta": abs(a2) / a0,
        "eta_sub": abs(a2_sub) / a0_sub if a0_sub > 0.0 else float("nan"),
        "bg_fraction": background * float(mask.sum()) / a0,
        "theta_deg": float(np.degrees(-np.angle(a2) / 2.0) % 180.0),
        "q_peak": float(length.ravel()[peak]),
        "q_mean": q_mean,
        "q_rms": q_rms,
        # 2 pi / dq is a correlation length; with dq at the grid spacing it is
        # reporting the box, not the texture, which is why the spacing goes out
        # alongside it.
        "xi": float(2.0 * np.pi / q_rms) if q_rms > 0.0 else float("inf"),
        "ring_weight": a0,
        "ring_count": int(mask.sum()),
        "radius": float(radius),
        "peak_intensity": float(intensity.ravel()[peak]),
        "total_weight": float(intensity.sum()),
    }
    if s_par is not None:
        out["longitudinal_fraction"] = float(
            s_par.ravel()[peak] / intensity.ravel()[peak]
        ) if intensity.ravel()[peak] > 0.0 else float("nan")

    # The Z3 reading of the same ring, computed here so that every caller gets
    # both numbers and a disagreement between them cannot go unnoticed.
    out["weights"] = domain_weights(intensity, q_grid, out["theta_deg"],
                                    radius=radius, width=width)
    out["psi"] = abs(z3_order_parameter(out["weights"]))
    return out


def domain_weights(intensity: np.ndarray, q_grid: np.ndarray, theta_deg: float,
                   *, radius: float | None = None,
                   width: float = DEFAULT_RING_WIDTH) -> np.ndarray:
    """Share of the ring nearest each of the three domain axes.

    An independent reading of the same thing: the weights feed the Z3 order
    parameter psi = sum_a w_a exp(2 pi i a / 3), whose modulus should track eta.
    Two definitions that disagree mean the ring is picking up something other
    than three domains, which is worth knowing.
    """
    if radius is None:
        radius = ring_radius(intensity, q_grid)
    mask = ring_mask(q_grid, radius, width)
    angle = np.degrees(np.arctan2(q_grid[..., 1], q_grid[..., 0])) % 180.0
    axes = (theta_deg + np.array([0.0, 60.0, 120.0])) % 180.0
    separation = np.abs(angle[mask][:, None] - axes[None, :])
    separation = np.minimum(separation, 180.0 - separation)
    nearest = np.argmin(separation, axis=1)
    weights = np.array([intensity[mask][nearest == a].sum() for a in range(3)])
    return weights / weights.sum()


def z3_order_parameter(weights: np.ndarray) -> complex:
    """psi = sum_a w_a exp(2 pi i a / 3); |psi| is 0 for three equal domains."""
    return complex((np.asarray(weights, float)
                    * np.exp(2j * np.pi * np.arange(3) / 3.0)).sum())


# ----------------------------------------------------------------------
# Frame interface
# ----------------------------------------------------------------------
def frame_selectivity(frame, cfg: dict | None = None) -> list[dict]:
    """One dict per layer. `frame` may be a list, which is averaged first."""
    cfg = cfg or {}
    frames = frame if isinstance(frame, (list, tuple)) else [frame]
    maps = average_intensity(frames, cfg) if len(frames) > 1 else frame_intensity(frames[0], cfg)

    want_longitudinal = bool(cfg.get("longitudinal", True))
    parallel_maps = None
    if want_longitudinal and str(cfg.get("channel", DEFAULT_CHANNEL)) not in ("L", "T"):
        spins = np.stack([frames[0]["u"], frames[0]["v"], frames[0]["w"]], axis=1).astype(float)
        parallel_maps = []
        for where, site_of, q_grid, _ in layer_geometry(frames[0], cfg):
            parallel_maps.append(project(amplitude(spins[where], site_of), q_grid, "L"))

    out = []
    for index, (q_grid, intensity, _) in enumerate(maps):
        record = selectivity(
            intensity, q_grid,
            width=float(cfg.get("ring_width", DEFAULT_RING_WIDTH)),
            radius=cfg.get("ring_radius"),
            s_par=None if parallel_maps is None else parallel_maps[index],
        )
        record["layer"] = index + 1
        record["frame_index"] = int(frames[0]["frame_index"])
        record["timestep"] = int(frames[0]["timestep"])
        record["n_frames"] = len(frames)
        out.append(record)
    return out


__all__ = [
    "DEFAULT_RING_WIDTH",
    "SELECTIVITY_HARMONIC",
    "domain_weights",
    "frame_selectivity",
    "harmonic",
    "ring_mask",
    "ring_radius",
    "selectivity",
    "z3_order_parameter",
]


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
_COLUMNS = (("frame", "frame_index", "d"), ("timestep", "timestep", "d"),
            ("layer", "layer", "d"), ("eta", "eta", ".6f"),
            ("eta_sub", "eta_sub", ".6f"), ("bg_frac", "bg_fraction", ".6f"),
            ("psi", "psi", ".6f"),
            ("theta_deg", "theta_deg", ".3f"), ("q_mean", "q_mean", ".6f"),
            ("q_rms", "q_rms", ".6f"), ("xi_A", "xi", ".3f"),
            ("w1", None, ".6f"), ("w2", None, ".6f"), ("w3", None, ".6f"),
            ("peak", "peak_intensity", ".5g"), ("ring_weight", "ring_weight", ".5g"),
            ("L_frac", "longitudinal_fraction", ".5f"))


def _row(record: dict) -> list[str]:
    out = []
    for name, key, fmt in _COLUMNS:
        if key is None:                                  # w1, w2, w3
            value = record["weights"][int(name[1]) - 1]
        else:
            value = record.get(key, float("nan"))
        out.append(f"{value:>14{fmt}}" if fmt != "d" else f"{int(value):>10d}")
    return out


def _header() -> str:
    parts = []
    for name, _, fmt in _COLUMNS:
        parts.append(f"{name:>10s}" if fmt == "d" else f"{name:>14s}")
    return "# " + "  ".join(parts)[2:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qselect.py",
        description="Domain selectivity of a spiral texture, per frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dump", type=Path, help="Input LAMMPS dump file")
    parser.add_argument("--vector", nargs=3, required=True, metavar=("SX", "SY", "SZ"),
                        help="The three spin columns, e.g. "
                             "--vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]'")
    parser.add_argument("--element", type=str, default="all",
                        help="Element symbol to include, or 'all'")
    parser.add_argument("--drop-zero-vector", action="store_true",
                        help="Skip atoms whose spin is exactly zero")
    parser.add_argument("--single-layer", action="store_true",
                        help="Treat the frame as one layer instead of splitting at z")
    parser.add_argument("--lattice-grid", "--topo-grid", type=int, nargs=2,
                        dest="lattice_grid", metavar=("N1", "N2"), default=None,
                        help="Cell repeats, read off the positions when omitted")
    parser.add_argument("--channel", type=str, default=DEFAULT_CHANNEL, metavar="C",
                        help="Projection of s^a(q) s^b*(q) behind S(q): 'total', 'L', 'T', "
                             "or a constant token such as xx")
    parser.add_argument("--ring-width", type=float, default=DEFAULT_RING_WIDTH, metavar="W",
                        help="Ring half-width as a fraction of |Q|. The answer barely "
                             "depends on it -- 0.05 and 0.40 agree to 1e-4 -- because the "
                             "q between the peaks carry almost nothing")
    parser.add_argument("--ring-radius", type=float, default=None, metavar="Q",
                        help="Fix |Q| in 1/Angstrom instead of taking it from the strongest "
                             "peak. Use it when the peak wanders between frames and the "
                             "ring should not wander with it")
    parser.add_argument("--frame", type=int, default=None, metavar="N",
                        help="Report this frame alone instead of the whole trajectory")
    parser.add_argument("--frame-average", type=int, default=1, metavar="W",
                        help="Average S(q) over W consecutive frames before measuring. A "
                             "single snapshot is one thermal realisation; averaging over a "
                             "window shorter than the selection but longer than a magnon "
                             "period cuts the noise as 1/sqrt(W)")
    parser.add_argument("--frame-start", type=int, default=0, help="First frame")
    parser.add_argument("--frame-stop", type=int, default=None, help="Stop before this frame")
    parser.add_argument("--frame-step", type=int, default=1, help="Keep one frame every N")
    parser.add_argument("--out", type=Path, default=Path("qselect.dat"),
                        help="Output table; '-' writes to the terminal only")
    parser.add_argument("--plot", type=Path, default=None, metavar="PNG",
                        help="Also draw eta(t) and theta_0(t)")
    return parser


def _report(record: dict) -> None:
    print(f"[INFO] layer {record['layer']}: "
          f"{record['ring_count']} allowed q in the ring at |Q| = {record['radius']:.5f} 1/A")
    print(f"[INFO]   eta          = {record['eta']:.6f}   "
          f"(0 = three domains equally, 1 = one domain)")
    print(f"[INFO]   eta_sub      = {record['eta_sub']:.6f}   "
          f"(the same with the diffuse background removed)")
    print(f"[INFO]   background   = {record['bg_fraction']:.4f} of the ring   "
          f"(how much of it is diffuse rather than Bragg)")
    print(f"[INFO]   psi          = {record['psi']:.6f}   "
          f"(the Z3 order parameter, an independent reading of the same thing)")
    print(f"[INFO]   theta_0      = {record['theta_deg']:.3f} deg   (surviving domain axis)")
    print(f"[INFO]   weights      = {np.array2string(record['weights'], precision=4)}")
    print(f"[INFO]   |Q|          = {record['q_mean']:.6f} +- {record['q_rms']:.6f} 1/A"
          f"   -> xi = {record['xi']:.1f} A")
    if "longitudinal_fraction" in record:
        fraction = record["longitudinal_fraction"]
        angle = np.degrees(np.arcsin(np.sqrt(np.clip(2.0 * fraction, 0.0, 1.0))))
        print(f"[INFO]   S_par/S_tot  = {fraction:.5f} at the peak   -> the spiral normal is "
              f"{angle:.1f} deg from Q")
        print(f"[INFO]                  (0 is a proper screw, n parallel to Q; "
              f"0.5 a cycloid, n perpendicular)")


def main(argv=None) -> None:
    from dumpframe import stream_frames

    args = build_parser().parse_args(argv)
    cfg = {
        "single_layer": args.single_layer,
        "lattice_grid": tuple(args.lattice_grid) if args.lattice_grid else None,
        "channel": args.channel,
        "ring_width": args.ring_width,
        "ring_radius": args.ring_radius,
    }
    window = max(1, int(args.frame_average))
    start = args.frame if args.frame is not None else args.frame_start
    stop = (args.frame + window) if args.frame is not None else args.frame_stop

    # The transform costs well under a millisecond; reading the dump is the
    # whole cost, so the frames are streamed and nothing is parallelised.
    pending: list[dict] = []
    rows: list[dict] = []
    for frame in stream_frames(
        args.input_dump, vector=tuple(args.vector), element=args.element,
        drop_zero_vector=args.drop_zero_vector, frame_start=start,
        frame_stop=stop, frame_step=args.frame_step,
    ):
        pending.append(frame)
        if len(pending) < window:
            continue
        rows.extend(frame_selectivity(pending if window > 1 else pending[0], cfg))
        pending = []

    if not rows:
        raise SystemExit(
            "No frames were measured. With --frame-average "
            f"{window} the dump must hold at least that many frames in the selected range."
        )

    if args.frame is not None or len(rows) <= len(set(r["layer"] for r in rows)):
        print(f"[INFO] Frame {rows[0]['frame_index']}, timestep {rows[0]['timestep']}"
              + (f", averaged over {window} frames" if window > 1 else "")
              + f", channel {args.channel}")
        for record in rows:
            _report(record)
        if str(args.out) == "-":
            return

    if str(args.out) != "-":
        with args.out.open("w") as fh:
            fh.write(f"# {' '.join(sys.argv)}\n")
            fh.write("# eta = |A_2|/A_0 on the ring at |Q|: 0 for three equal domains, "
                     "1 for one\n")
            fh.write("# eta_sub is eta after removing a flat background at the ring's "
                     "median; bg_frac is that\n")
            fh.write("#   background's share of the ring, so it says how disordered the "
                     "frame is\n")
            fh.write("# theta_deg is the surviving domain axis, mod 180; w1..w3 are the "
                     "shares nearest each axis\n")
            fh.write("# L_frac at the peak is sin^2(theta)/2 for the angle between the "
                     "spiral normal and Q\n")
            fh.write(_header() + "\n")
            for record in rows:
                fh.write("  ".join(_row(record)) + "\n")
        print(f"[INFO] {len(rows)} row(s) -> {args.out}")

    if args.plot is not None:
        _plot(rows, args.plot)


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({r["layer"] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True,
                             constrained_layout=True)
    for layer in layers:
        picked = [r for r in rows if r["layer"] == layer]
        steps = [r["timestep"] for r in picked]
        label = f"layer {layer}" if len(layers) > 1 else None
        axes[0].plot(steps, [r["eta"] for r in picked], "o-", ms=3, lw=1.2, label=label)
        # theta_0 lives on a circle of period 180 degrees, so joining the dots
        # across the wrap would draw a line through angles the domain never had.
        angles = np.array([r["theta_deg"] for r in picked])
        breaks = np.nonzero(np.abs(np.diff(angles)) > 90.0)[0]
        for chunk_steps, chunk_angles in zip(np.split(np.array(steps), breaks + 1),
                                             np.split(angles, breaks + 1)):
            axes[1].plot(chunk_steps, chunk_angles, "o-", ms=3, lw=1.2,
                         color=axes[0].lines[-1].get_color())
    axes[0].set_ylabel(r"$\eta = |A_2| / A_0$", fontsize=14)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].axhline(0.0, color="k", lw=0.6, alpha=0.4)
    axes[0].axhline(1.0, color="k", lw=0.6, alpha=0.4)
    axes[0].grid(alpha=0.25)
    if len(layers) > 1:
        axes[0].legend(fontsize=11)
    axes[1].set_ylabel(r"$\theta_0$ (deg)", fontsize=14)
    axes[1].set_ylim(0.0, 180.0)
    axes[1].set_yticks([0, 60, 120, 180])
    axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("timestep", fontsize=14)
    fig.suptitle("Spiral domain selection", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[INFO] plot -> {path}")


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _three_arms(reciprocal, n):
    """The (h, k) of one wavevector and its two C3 partners, in cell units."""
    rotation = np.array([[-0.5, -np.sqrt(3) / 2, 0.0],
                         [np.sqrt(3) / 2, -0.5, 0.0], [0.0, 0.0, 1.0]])
    arms, hk = [], np.array([4.0, 0.0])
    for _ in range(3):
        arms.append(np.rint(hk).astype(int))
        cartesian = rotation @ ((hk[0] / n) * reciprocal[0] + (hk[1] / n) * reciprocal[1])
        hk = np.linalg.solve(reciprocal[:2, :2].T, cartesian[:2]) * n
    return arms


def _texture(arms, n, weights, normal=(0.0, 0.0, 1.0), noise=0.0, rng=None,
             phases=(0.0, 1.1, 2.3)):
    """Superposed spirals with the given weights, optionally with thermal noise."""
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    seed = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    spins = np.zeros((n * n, 3))
    for (h, k), weight, phase0 in zip(arms, weights, phases):
        phase = 2.0 * np.pi * (h * i.ravel() / n + k * j.ravel() / n) + phase0
        spins += weight * (np.cos(phase)[:, None] * e1 + np.sin(phase)[:, None] * e2)
    if noise:
        spins += noise * rng.normal(size=spins.shape)
    return spins / np.linalg.norm(spins, axis=1, keepdims=True)


def _selftest() -> int:
    from geometry import reciprocal_lattice_from_real
    from lattice_grid import site_of_grid
    from polarization import _triangular_lattice
    from structurefactor import allowed_q, primitive_lattice

    rng = np.random.default_rng(0)
    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failures += 0 if ok else 1

    n = 36
    positions, lattice = _triangular_lattice(n, n)
    site_of, n1, n2 = site_of_grid(positions, lattice, None)
    q_grid = allowed_q(lattice, n1, n2)
    reciprocal = reciprocal_lattice_from_real(primitive_lattice(lattice, n1, n2))
    arms = _three_arms(reciprocal, n)
    check("the three arms are C3 related", len(arms) == 3 and len({tuple(a) for a in arms}) == 3,
          f"(h,k) = {[tuple(int(v) for v in a) for a in arms]}")

    def measure(weights, **kwargs):
        spins = _texture(arms, n, weights, rng=rng, **kwargs)
        intensity = project(amplitude(spins, site_of), q_grid, "total")
        return selectivity(intensity, q_grid), intensity

    # -- the identity the whole method rests on -------------------------------
    # A C3-symmetric intensity has to give A_2 = 0 exactly, not approximately:
    # the allowed q fall into complete C3 orbits and each orbit cancels on its
    # own. Symmetrising an arbitrary map is the sharpest way to test that.
    fractional = np.stack(np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n,
                                      indexing="ij"), axis=-1)
    rotation = np.array([[-0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, -0.5]])
    mapping = np.rint(fractional @ np.linalg.solve(reciprocal[:2, :2].T,
                                                   rotation @ reciprocal[:2, :2].T).T
                      ).astype(int) % n
    noise_map = rng.random((n, n))
    once = noise_map[mapping[..., 0], mapping[..., 1]]
    symmetric = (noise_map + once + once[mapping[..., 0], mapping[..., 1]]) / 3.0
    a0, a2 = harmonic(symmetric, q_grid, ring_mask(q_grid, 0.2026))
    check("a C3-symmetric map has no m=2 weight", abs(a2) / a0 < 1e-12,
          f"|A_2|/A_0 = {abs(a2) / a0:.2e}, exact cancellation on the discrete grid")

    # -- the two limits -------------------------------------------------------
    three, _ = measure((1.0, 1.0, 1.0))
    single, _ = measure((1.0, 0.0, 0.0))
    check("three equal domains give eta = 0", three["eta"] < 0.05,
          f"eta = {three['eta']:.4f}")
    check("a single domain gives eta = 1", abs(single["eta"] - 1.0) < 1e-9,
          f"eta = {single['eta']:.9f}")
    check("psi agrees with eta at both limits",
          three["psi"] < 0.05 and abs(single["psi"] - 1.0) < 1e-9,
          f"psi = {three['psi']:.4f} and {single['psi']:.9f}")

    # -- the interpolation between them ---------------------------------------
    etas = [measure((w, 1.0, 1.0))[0]["eta"] for w in (1.0, 2.0, 4.0, 8.0, 32.0)]
    check("eta rises monotonically with the dominant weight",
          all(b > a for a, b in zip(etas, etas[1:])) and etas[-1] > 0.99,
          "  ".join(f"{e:.4f}" for e in etas))

    # -- background subtraction separates order from selection ----------------
    # A diffuse background dilutes eta, so a merely disordered frame reads as
    # an unselected one. Removing it has to raise the single-domain value
    # without raising the three-domain value, or it is fitting noise.
    noisy_three, _ = measure((1.0, 1.0, 1.0), noise=1.2)
    noisy_one, _ = measure((1.0, 0.0, 0.0), noise=1.2)
    check("removing the background lifts the ceiling, not the floor",
          noisy_three["eta_sub"] < 0.06 and noisy_one["eta_sub"] > noisy_one["eta"] + 0.05,
          f"three domains {noisy_three['eta']:.4f} -> {noisy_three['eta_sub']:.4f}, "
          f"one domain {noisy_one['eta']:.4f} -> {noisy_one['eta_sub']:.4f}")
    check("the background fraction tracks the disorder",
          noisy_one["bg_fraction"] > measure((1.0, 0.0, 0.0))[0]["bg_fraction"],
          f"{measure((1.0, 0.0, 0.0))[0]['bg_fraction']:.4f} clean against "
          f"{noisy_one['bg_fraction']:.4f} noisy")

    # -- the phase identifies which domain ------------------------------------
    angles = [measure(w)[0]["theta_deg"] for w in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    separations = sorted(min(abs(a - b), 180.0 - abs(a - b))
                         for a, b in ((angles[0], angles[1]), (angles[1], angles[2]),
                                      (angles[2], angles[0])))
    check("theta_0 separates the three domains by 60 deg",
          all(abs(s - 60.0) < 1e-6 for s in separations),
          f"axes at {', '.join(f'{a:.2f}' for a in angles)} deg")

    # -- the ring width is not a knob -----------------------------------------
    spins = _texture(arms, n, (1.0, 0.0, 0.0), rng=rng)
    intensity = project(amplitude(spins, site_of), q_grid, "total")
    widths = [selectivity(intensity, q_grid, width=w)["eta"]
              for w in (0.05, 0.10, 0.15, 0.25, 0.40)]
    check("eta does not depend on the ring width",
          max(widths) - min(widths) < 1e-4,
          f"spread {max(widths) - min(widths):.2e} over half-widths 0.05 to 0.40")

    # -- noise ----------------------------------------------------------------
    rows = []
    for level in (0.0, 0.3, 0.6, 1.0):
        one, _ = measure((1.0, 0.0, 0.0), noise=level)
        rows.append((level, one["eta"], one["theta_deg"]))
    check("noise degrades eta gracefully",
          all(e > 0.85 for _, e, _ in rows) and all(abs(t - rows[0][2]) < 1.0 for _, _, t in rows),
          "  ".join(f"{lv:g}->{e:.3f}" for lv, e, _ in rows) + " with theta_0 within 1 deg")

    # -- rotating the map rotates the answer, nothing else --------------------
    # Turning the intensity by 120 degrees on the grid is the honest test of
    # invariance: it exercises this module alone, whereas permuting the weights
    # of a synthetic texture also permutes phases the transform can see.
    reference_map = measure((4.0, 1.0, 1.0))[1]
    turned = reference_map[mapping[..., 0], mapping[..., 1]]
    before = selectivity(reference_map, q_grid)
    after = selectivity(turned, q_grid)
    # Indexing through `mapping` pulls the map back through the rotation, so the
    # picture turns by -120 degrees and the axis, read modulo 180, moves by +60.
    shift = (after["theta_deg"] - before["theta_deg"]) % 180.0
    check("eta is invariant when the map is turned by 120 deg",
          abs(before["eta"] - after["eta"]) < 1e-12 and abs(shift - 60.0) < 1e-6,
          f"eta {before['eta']:.12f} -> {after['eta']:.12f}, "
          f"theta_0 {before['theta_deg']:.2f} -> {after['theta_deg']:.2f} deg")

    # -- the chirality diagnostic ---------------------------------------------
    peak_q = q_grid[tuple(arms[0] % n)]
    axis = np.cross(peak_q / np.linalg.norm(peak_q), [0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    unit = peak_q / np.linalg.norm(peak_q)
    worst, rows = 0.0, []
    for degrees in (0.0, 30.0, 45.0, 60.0, 90.0):
        angle = np.radians(degrees)
        normal = np.cos(angle) * unit + np.sin(angle) * np.cross(axis, unit)
        spins = _texture(arms, n, (1.0, 0.0, 0.0), normal=normal, rng=rng)
        s_q = amplitude(spins, site_of)
        record = selectivity(project(s_q, q_grid, "total"), q_grid,
                             s_par=project(s_q, q_grid, "L"))
        worst = max(worst, abs(record["longitudinal_fraction"]
                               - 0.5 * np.sin(angle) ** 2))
        rows.append(f"{degrees:g}deg->{record['longitudinal_fraction']:.4f}")
    check("S_par/S_tot reads the spiral normal angle", worst < 1e-9,
          "  ".join(rows) + "; sin^2(theta)/2, so 0 is a proper screw")

    # -- pitch and width ------------------------------------------------------
    one, intensity = measure((1.0, 0.0, 0.0))
    expected = float(np.linalg.norm(q_grid[tuple(arms[0] % n)]))
    check("|Q| is the peak's own length", abs(one["q_mean"] - expected) < 1e-9,
          f"{one['q_mean']:.6f} against {expected:.6f} 1/Angstrom")
    check("a sharp peak has no radial width", one["q_rms"] < 1e-9,
          f"dq = {one['q_rms']:.2e}, so xi saturates at the box")

    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    return failures


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        raise SystemExit(_selftest())
