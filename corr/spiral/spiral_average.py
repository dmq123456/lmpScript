#!/usr/bin/env python3
"""Average the spiral parameters over the frames of a dump.

    spiral_average.py field-4.19-off-last.lammpstrj --supercell 50 50 1 --type 1

Every frame is fitted the way spin_spiral.py fits one: the dominant q from a 2D
FFT over the magnetic lattice, and S(r) = u cos(q.r) + v sin(q.r) + cone * n
from that Fourier component. The parameters are then combined frame by frame,
each in the way that suits it:

  q      an integer index on the supercell -- never averaged, only checked to
         be the same in every frame (a non-integer q would not fit the box).
  n      the oriented plane normal, averaged as a vector and renormalised;
         a frame whose n points against the mean has flipped chirality.
  cone   <S.n>, a scalar, averaged as a scalar.
  phase  where in the plane the spin at site (0,0) points. A spiral that slides
         changes it from frame to frame, so it is NOT folded into an averaged
         s0 (that would shrink the in-plane part and fake a larger cone); it is
         reported, and its circular mean is used only if the frames agree.

The averaged s0 is then rebuilt as cos(c) u_hat + sin(c) n, and a command line
for make_spiral.py is printed. As a check, the ideal spiral from the averaged
parameters is compared with every frame, site by site.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from spin_spiral import canonical, find_modulation, parse_box, peak_table, spiral_plane


def iter_frames(path: Path):
    """Yield (timestep, cell, columns, rows) for every frame, in one pass."""
    with path.open() as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                sys.exit(f"error: expected 'ITEM: TIMESTEP', got {line.strip()!r}")
            timestep = int(fh.readline())
            fh.readline()
            natoms = int(fh.readline())
            box_header = fh.readline().strip()
            box_lines = [fh.readline().strip() for _ in range(3)]
            columns = fh.readline().split()[2:]
            rows = [fh.readline().split() for _ in range(natoms)]
            yield timestep, parse_box(box_lines, box_header), columns, rows


def lattice_grid(cell, columns, rows, supercell, spin_columns, atom_type):
    """Spins of the magnetic atoms on their (N1, N2) lattice."""
    index = {name: i for i, name in enumerate(columns)}
    for name in ("x", "y", "z", *spin_columns):
        if name not in index:
            sys.exit(f"error: dump has no '{name}' column; present: {' '.join(columns)}")
    data = np.array(rows)
    positions = data[:, [index[c] for c in ("x", "y", "z")]].astype(float)
    spins = data[:, [index[c] for c in spin_columns]].astype(float)
    if atom_type is not None:
        keep = data[:, index["type"]].astype(int) == atom_type
    else:
        keep = np.linalg.norm(spins, axis=1) > 1e-8

    primitive = cell / supercell[:, None]
    fractional = positions[keep] @ np.linalg.inv(primitive)
    i1 = np.rint(fractional[:, 0]).astype(int) % supercell[0]
    i2 = np.rint(fractional[:, 1]).astype(int) % supercell[1]
    if len(set(zip(i1.tolist(), i2.tolist()))) != keep.sum():
        sys.exit("error: magnetic atoms do not map one-to-one onto lattice sites; check --supercell")
    grid = np.zeros((supercell[0], supercell[1], 3))
    grid[i1, i2] = spins[keep]
    return grid


def site_phases(m1, m2, supercell):
    """2*pi*(m1*i1/N1 + m2*i2/N2) on the whole lattice: the phase q.r of every site."""
    i1, i2 = np.meshgrid(np.arange(supercell[0]), np.arange(supercell[1]), indexing="ij")
    return 2.0 * np.pi * (m1 * i1 / supercell[0] + m2 * i2 / supercell[1])


def ideal_spiral(normal, s0, phases):
    """Rodrigues rotation of s0 about n by each phase, as make_spiral.py builds it."""
    cross = np.cross(normal, s0)
    along = normal * float(normal @ s0)
    c, s = np.cos(phases)[..., None], np.sin(phases)[..., None]
    return s0 * c + cross * s + along * (1.0 - c)


def angle_stats(a, b):
    """Mean and max angle (deg) between unit-vector fields a and b."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))
    return float(ang.mean()), float(np.sqrt((ang ** 2).mean())), float(ang.max())


def build_parser():
    p = argparse.ArgumentParser(
        prog="spiral_average.py",
        description="Average spiral parameters (q, n, cone, phase) over the frames of a dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("dump", type=Path, help="Input dump with several frames")
    p.add_argument("--supercell", type=int, nargs=3, default=(50, 50, 1), metavar=("N1", "N2", "N3"))
    p.add_argument("--spin-columns", nargs=3, default=["c_outsp[1]", "c_outsp[2]", "c_outsp[3]"],
                   metavar=("SX", "SY", "SZ"))
    p.add_argument("--type", type=int, default=1, help="Magnetic atom type")
    p.add_argument("--spin-length", type=float, default=2.0,
                   help="Passed on to make_spiral.py (the dump stores unit directions)")
    p.add_argument("--out", type=Path, default=None,
                   help="Per-frame table; default: <dump stem>.spiral.dat")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.dump.exists():
        sys.exit(f"error: no such file: {args.dump}")
    supercell = np.asarray(args.supercell, dtype=int)

    records, grids = [], []
    for timestep, cell, columns, rows in iter_frames(args.dump):
        grid = lattice_grid(cell, columns, rows, supercell, args.spin_columns, args.type)
        (m1, m2), fraction, power = find_modulation(grid)
        second = peak_table(power, 2)[1][1] if len(peak_table(power, 2)) > 1 else 0.0
        u, v, normal, ellipticity = spiral_plane(grid, m1, m2, "ccw")
        if normal is None:
            sys.exit(f"error: timestep {timestep}: u x v vanishes, not a spiral")
        unit = grid / np.linalg.norm(grid, axis=-1, keepdims=True)
        cone = float(unit.reshape(-1, 3).mean(axis=0) @ normal)
        records.append(dict(timestep=timestep, q=(m1, m2), fraction=fraction, second=second,
                            u=u, v=v, n=normal, ellipticity=ellipticity, cone=cone))
        grids.append(unit)
    if not records:
        sys.exit("error: no frames read")
    nf = len(records)
    print(f"[INFO] Frames read     : {nf}  (timesteps {records[0]['timestep']} .. {records[-1]['timestep']})")

    # ---- q: must be the same everywhere --------------------------------------
    qcount = Counter(r["q"] for r in records)
    (m1, m2), nq = qcount.most_common(1)[0]
    print(f"[INFO] q index         : " + ", ".join(f"({a:+d},{b:+d}) x{c}" for (a, b), c in qcount.items()))
    if nq != nf:
        print(f"[WARN] q is not the same in every frame; using the most common ({m1:+d},{m2:+d}).")
    print(f"[INFO] Main peak       : min {100*min(r['fraction'] for r in records):.1f}%  "
          f"mean {100*np.mean([r['fraction'] for r in records]):.1f}% of the modulated power; "
          f"largest other peak {100*max(r['second'] for r in records):.1f}%")
    print(f"[INFO] Ellipticity     : mean {np.mean([r['ellipticity'] for r in records]):.4f}  "
          f"min {min(r['ellipticity'] for r in records):.4f}   (1 = circular)")

    # ---- n: vector mean -------------------------------------------------------
    normals = np.array([r["n"] for r in records])
    n_mean = normals.mean(axis=0)
    n_mean /= np.linalg.norm(n_mean)
    dots = normals @ n_mean
    flipped = int((dots < 0).sum())
    spread = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    print(f"[INFO] n (mean)        : ({n_mean[0]:+.5f}, {n_mean[1]:+.5f}, {n_mean[2]:+.5f})")
    print(f"[INFO] n spread        : mean {spread.mean():.3f} deg, max {spread.max():.3f} deg"
          + (f"   [WARN] {flipped} frame(s) with reversed chirality" if flipped else ""))

    # ---- cone: scalar mean ----------------------------------------------------
    cones = np.array([r["cone"] for r in records])
    cone_deg = np.degrees(np.arcsin(np.clip(cones, -1, 1)))
    c = float(np.mean(cone_deg))
    print(f"[INFO] Cone            : <S.n> = {cones.mean():+.5f} +- {cones.std(ddof=1) if nf > 1 else 0:.5f}"
          f"  ->  {c:+.3f} +- {cone_deg.std(ddof=1) if nf > 1 else 0:.3f} deg out of the plane")

    # ---- phase: in a fixed in-plane basis -------------------------------------
    e1 = records[0]["u"] - n_mean * (records[0]["u"] @ n_mean)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_mean, e1)
    phases = np.array([np.arctan2(r["u"] @ e2, r["u"] @ e1) for r in records])
    unwrapped = np.degrees(np.unwrap(phases))
    resultant = np.abs(np.exp(1j * phases).mean())
    print(f"[INFO] Phase of u      : first {unwrapped[0]:+.2f}, last {unwrapped[-1]:+.2f} deg; "
          f"std {unwrapped.std():.2f} deg; circular coherence R = {resultant:.4f}")
    if resultant > 0.9:
        phi = float(np.angle(np.exp(1j * phases).mean()))
        note = "circular mean over the frames"
    else:
        phi = float(phases[-1])
        note = "frames disagree (spiral slides) -> last frame used"
        print("[WARN] The phase is not stable; s0 takes the last frame's phase.")
    u_hat = np.cos(phi) * e1 + np.sin(phi) * e2
    c_rad = np.radians(c)
    s0 = np.cos(c_rad) * u_hat + np.sin(c_rad) * n_mean
    print(f"[INFO] s0 (averaged)   : ({s0[0]:+.5f}, {s0[1]:+.5f}, {s0[2]:+.5f})   phase: {note}")

    # ---- check: ideal spiral vs every frame -----------------------------------
    phases_site = site_phases(m1, m2, supercell)
    ideal = ideal_spiral(n_mean, s0, phases_site)
    print("[CHECK] angle between each frame and the ideal spiral (deg):")
    print("        averaged parameters            |  frame's own phase (q, n, cone averaged)")
    rows_out = []
    for k, (r, g) in enumerate(zip(records, grids)):
        mean_a, rms_a, max_a = angle_stats(g, ideal)
        # same averaged n and cone, but this frame's phase: separates phase drift from shape
        uk = np.cos(phases[k]) * e1 + np.sin(phases[k]) * e2
        own = ideal_spiral(n_mean, np.cos(c_rad) * uk + np.sin(c_rad) * n_mean, phases_site)
        mean_b, rms_b, max_b = angle_stats(g, own)
        rows_out.append((r["timestep"], r["q"][0], r["q"][1], r["fraction"], r["ellipticity"],
                         *r["n"], r["cone"], np.degrees(phases[k]), mean_a, rms_a, mean_b, rms_b))
        if k < 3 or k >= nf - 3 or k == nf // 2:
            print(f"        t={r['timestep']:7d}  mean {mean_a:6.2f}  rms {rms_a:6.2f}  max {max_a:6.1f}"
                  f"  |  mean {mean_b:6.2f}  rms {rms_b:6.2f}  max {max_b:6.1f}")
        elif k == 3:
            print("        ...")
    # thermal noise or real distortion? average each site over time first
    stack = np.array(grids)
    site_mean = stack.mean(axis=0)
    order = np.linalg.norm(site_mean, axis=-1)
    mean_t, rms_t, max_t = angle_stats(site_mean, ideal)
    print(f"[CHECK] per-site order |<S_i>_t| : mean {order.mean():.4f}, min {order.min():.4f}"
          f"   (1 = the site does not wander)")
    print(f"[CHECK] time-averaged texture vs ideal spiral: mean {mean_t:.2f}  rms {rms_t:.2f}  max {max_t:.1f} deg"
          f"   (what remains is not thermal: ellipticity, harmonics, defects)")
    table = np.array(rows_out, dtype=float)
    print(f"[CHECK] over all frames: averaged parameters rms {table[:, 11].mean():.2f} deg, "
          f"frame's own phase rms {table[:, 13].mean():.2f} deg")

    out = args.out or args.dump.with_suffix(".spiral.dat")
    np.savetxt(out, table, fmt="%d %d %d %.5f %.5f %+.6f %+.6f %+.6f %+.6f %+.3f %.3f %.3f %.3f %.3f",
               header="timestep m1 m2 main_peak ellipticity nx ny nz cone phase_deg "
                      "angle_mean angle_rms angle_mean_ownphase angle_rms_ownphase")
    print(f"[INFO] Per-frame table : {out}")

    print("\n[RESULT] make_spiral.py command for the averaged spiral:")
    make_spiral = Path(__file__).resolve().parent / "make_spiral.py"
    print(f"python {make_spiral} --supercell {supercell[0]} {supercell[1]} {supercell[2]} "
          f"--q-index {m1} {m2} 0 --e {n_mean[0]:.5f} {n_mean[1]:.5f} {n_mean[2]:.5f} "
          f"--s0 {s0[0]:.5f} {s0[1]:.5f} {s0[2]:.5f} --spin-length {args.spin_length} --out spiral_avg.lammpstrj")


if __name__ == "__main__":
    main()
