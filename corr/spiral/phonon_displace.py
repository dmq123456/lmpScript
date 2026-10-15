#!/usr/bin/env python3
"""Freeze one magnetic configuration and move the atoms along a phonon mode.

    phonon_displace.py spiral.lammpstrj --supercell 50 50 1 \\
        --band-yaml band.yaml --q 0 0 0 --bands 7 9 --direction Ni 1 0 0 \\
        --amax 0.05 --nframes 11 --out phonon.lammpstrj

The atoms are NOT taken from the input dump: the structure is always the ideal
one built into make_spiral.py, replicated by --supercell. The dump only supplies
the spins, copied site by site onto that lattice, so every output frame carries
exactly the same magnetic configuration and differs only in the displacement

    u_j(R) = A * Re[ e_j exp(2 pi i q.R) ] / sqrt(m_j)

of atom j in the cell at lattice vector R, with e the phonopy eigenvector (or a
combination of degenerate ones, see below) and m the masses from band.yaml.

Choosing the mode
-----------------
--q picks the q-point by its coordinates (fractional, primitive reciprocal
basis); a point that appears twice on the path -- Gamma at the joint of two
segments -- is taken at its first appearance unless --occurrence says
otherwise. Band numbers are 1-based and refer to the order of that q-point in
band.yaml, which follows band connectivity, not frequency: check the printed
frequencies.

Several --bands must be degenerate. Any real combination of them is then an
eigenvector too, and one is fixed by
  --direction EL X Y Z   the combination in which the first atom of element EL
                         moves along (X, Y, Z) (Cartesian, dump frame); or
  --mix-angle DEG        e = cos(DEG) e_1 + sin(DEG) e_2 (two bands only).
A positive amplitude moves that atom along +direction.

Amplitude
---------
--amp-unit disp (default): A is the largest displacement of any atom, in Angstrom.
--amp-unit Q: A is the normal coordinate in Angstrom*sqrt(amu), u = Q e / sqrt(m).
Give the values with --amplitudes, or --amax and --nframes for an even grid
from -amax to +amax.

Next steps: dump_to_extxyz.py, then predict_forces.py --energy-file. The
amplitude of every frame is written to <out>.amp.dat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dumpio import read_frames
from make_spiral import BASIS, ELEMENTS, MAGNETIC_TYPE, PRIM_CELL, write_dump


# ----------------------------------------------------------------------
# band.yaml
# ----------------------------------------------------------------------
def load_band_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("error: PyYAML is needed to read band.yaml (pip install pyyaml)")
    try:
        loader = yaml.CSafeLoader
    except AttributeError:
        loader = yaml.SafeLoader
    with path.open() as fh:
        return yaml.load(fh, Loader=loader)


def segment_label(data: dict, index: int) -> str:
    """'K-G' style label of the path segment that holds q-point `index`."""
    counts = data.get("segment_nqpoint")
    labels = data.get("labels")
    if not counts or not labels:
        return ""
    start = 0
    for count, pair in zip(counts, labels):
        if index < start + count:
            where = "start" if index == start else "end" if index == start + count - 1 else ""
            return f"segment {pair[0]}-{pair[1]}" + (f", {where}" if where else "")
        start += count
    return ""


def select_mode(data: dict, q, occurrence: int, bands: list[int], tol: float):
    """The chosen q-point and the eigenvectors (natom, 3, nband) of the chosen bands."""
    phonon = data["phonon"]
    q = np.asarray(q, dtype=float)
    hits = [i for i, p in enumerate(phonon)
            if np.allclose(np.asarray(p["q-position"], dtype=float), q, atol=1e-6)]
    if not hits:
        sys.exit(f"error: q = {q.tolist()} is not on the path in band.yaml")
    if not 1 <= occurrence <= len(hits):
        sys.exit(f"error: q = {q.tolist()} appears {len(hits)} time(s); --occurrence {occurrence} is out of range")
    index = hits[occurrence - 1]
    point = phonon[index]
    nband = len(point["band"])

    qpos = np.asarray(point["q-position"], dtype=float)
    label = segment_label(data, index)
    print(f"[MODE] q-point        : #{index + 1} of {len(phonon)}   "
          f"q = ({qpos[0]:+.7f}, {qpos[1]:+.7f}, {qpos[2]:+.7f})"
          + (f"   [{label}]" if label else ""))
    if len(hits) > 1:
        others = ", ".join(f"#{i + 1}" for i in hits if i != index)
        print(f"[MODE]   the same q also appears as {others}; band order may differ there "
              f"(use --occurrence)")

    vectors, freqs = [], []
    for b in bands:
        if not 1 <= b <= nband:
            sys.exit(f"error: band {b} out of range 1..{nband}")
        entry = point["band"][b - 1]
        freqs.append(float(entry["frequency"]))
        ev = np.asarray(entry["eigenvector"], dtype=float)        # (natom, 3, 2)
        vectors.append(ev[..., 0] + 1j * ev[..., 1])
        print(f"[MODE] band {b:3d}       : {freqs[-1]:10.6f} THz")
    if len(bands) > 1 and max(freqs) - min(freqs) > tol:
        sys.exit(f"error: the selected bands are not degenerate "
                 f"(spread {max(freqs) - min(freqs):.6f} THz > --degeneracy-tol {tol} THz)")
    return np.stack(vectors, axis=-1), freqs


# ----------------------------------------------------------------------
# Combining degenerate eigenvectors
# ----------------------------------------------------------------------
def combine(vectors: np.ndarray, masses: np.ndarray, symbols: list[str], args) -> np.ndarray:
    """One normalised eigenvector (natom, 3) from the selected band(s)."""
    nb = vectors.shape[-1]
    if nb == 1:
        if args.mix_angle is not None:
            sys.exit("error: --mix-angle needs two degenerate bands")
        e = vectors[..., 0]
    elif args.mix_angle is not None:
        if nb != 2:
            sys.exit("error: --mix-angle works with exactly two bands")
        t = np.radians(args.mix_angle)
        e = np.cos(t) * vectors[..., 0] + np.sin(t) * vectors[..., 1]
    elif args.direction is not None:
        if np.abs(vectors.imag).max() > 1e-8:
            sys.exit("error: --direction needs real eigenvectors (e.g. at Gamma); use --mix-angle")
        atom, d = direction_target(args.direction, symbols)
        # the atom's displacement for coefficients c is M c; ask for M c parallel to d
        M = vectors.real[atom] / np.sqrt(masses[atom])             # (3, nb)
        P = np.eye(3) - np.outer(d, d)
        w, V = np.linalg.eigh(M.T @ P @ M)
        c = V[:, 0]
        e = vectors.real @ c
        moved = M @ c
        if np.linalg.norm(moved) < 1e-10:
            sys.exit(f"error: {symbols[atom]} does not move in this subspace along any direction")
    else:
        sys.exit("error: several bands given; choose the combination with --direction or --mix-angle")
    return e / np.sqrt((np.abs(e) ** 2).sum())


def direction_target(spec: list[str], symbols: list[str]):
    element, *xyz = spec
    if len(xyz) != 3:
        sys.exit("error: --direction takes an element and three numbers, e.g. --direction Ni 1 0 0")
    if element not in symbols:
        sys.exit(f"error: no {element} in band.yaml (atoms: {symbols})")
    d = np.asarray([float(v) for v in xyz])
    if np.linalg.norm(d) < 1e-12:
        sys.exit("error: --direction must be a non-zero vector")
    return symbols.index(element), d / np.linalg.norm(d)


# ----------------------------------------------------------------------
# Matching band.yaml to the make_spiral.py structure
# ----------------------------------------------------------------------
def rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Proper rotation taking the a, b directions of `src` onto those of `dst`.

    Only directions are matched -- the c axes differ in length (different
    vacuum), so a fractional-coordinate round trip would rescale z components.
    """
    def frame(cell):
        a = cell[0] / np.linalg.norm(cell[0])
        n = np.cross(cell[0], cell[1]); n /= np.linalg.norm(n)
        return np.array([a, np.cross(n, a), n])
    return frame(dst).T @ frame(src)


def match_atoms(data: dict, rotation: np.ndarray):
    """For every make_spiral BASIS atom, the band.yaml atom it is."""
    lattice = np.asarray(data["lattice"], dtype=float)
    points = data["points"]
    ph_frac = np.array([p["coordinates"] for p in points], dtype=float)
    ph_sym = [p["symbol"] for p in points]
    ph_cart = (ph_frac @ lattice) @ rotation.T

    b_cart = np.array([[x, y, z] for _, x, y, z in BASIS])
    b_sym = [ELEMENTS[t] for t, *_ in BASIS]
    b_frac = b_cart @ np.linalg.inv(PRIM_CELL)

    # heights relative to the magnetic atom, which both cells have once
    ref_b = next(i for i, (t, *_) in enumerate(BASIS) if t == MAGNETIC_TYPE)
    ref_p = ph_sym.index(b_sym[ref_b])
    dz_b = b_cart[:, 2] - b_cart[ref_b, 2]
    dz_p = ph_cart[:, 2] - ph_cart[ref_p, 2]

    mapping, report = [], []
    for j in range(len(BASIS)):
        best, score = None, np.inf
        for p in range(len(points)):
            if ph_sym[p] != b_sym[j]:
                continue
            df = b_frac[j, :2] - ph_frac[p, :2]
            df -= np.rint(df)
            s = np.linalg.norm(df) + abs(dz_b[j] - dz_p[p]) / 10.0
            if s < score:
                best, score = p, s
        if best is None:
            sys.exit(f"error: no {b_sym[j]} in band.yaml to match basis atom {j + 1}")
        df = b_frac[j, :2] - ph_frac[best, :2]
        df -= np.rint(df)
        mapping.append(best)
        report.append((j, best, b_sym[j], np.abs(df).max(), dz_b[j] - dz_p[best]))
    if len(set(mapping)) != len(mapping):
        sys.exit(f"error: atom matching is not one-to-one: {mapping}")
    return mapping, report


# ----------------------------------------------------------------------
# Spins from the dump, onto the ideal lattice
# ----------------------------------------------------------------------
def spins_from_dump(path: Path, frame: int, supercell, spin_columns, atom_type,
                    length_column: str, spin_length: float):
    for k, (timestep, cell, columns, rows) in enumerate(read_frames(path)):
        if k == frame:
            break
    else:
        sys.exit(f"error: frame {frame} not found in {path}")
    index = {name: i for i, name in enumerate(columns)}
    for name in ("x", "y", "z", "type", *spin_columns):
        if name not in index:
            sys.exit(f"error: dump has no '{name}' column; present: {' '.join(columns)}")
    data = np.array(rows)
    keep = data[:, index["type"]].astype(int) == atom_type
    positions = data[keep][:, [index[c] for c in ("x", "y", "z")]].astype(float)
    spins = data[keep][:, [index[c] for c in spin_columns]].astype(float)
    if length_column in index:
        lengths = data[keep][:, index[length_column]].astype(float)
        source = f"column '{length_column}'"
    else:
        lengths = np.full(len(spins), spin_length)
        source = f"--spin-length {spin_length}"
    norms = np.linalg.norm(spins, axis=1)
    if np.any(norms < 1e-8):
        sys.exit("error: a magnetic atom in the dump has a zero spin")
    spins /= norms[:, None]

    primitive = cell / supercell[:, None]
    frac = positions @ np.linalg.inv(primitive)
    sites = np.rint(frac).astype(int) % supercell
    nsites = int(np.prod(supercell))
    flat = np.ravel_multi_index(sites.T, supercell)
    if len(np.unique(flat)) != len(flat) or len(flat) != nsites:
        sys.exit(f"error: {len(flat)} magnetic atoms in the dump do not map one-to-one onto the "
                 f"{nsites} sites of a {tuple(supercell)} supercell; check --supercell")
    grid = np.zeros((nsites, 3))
    lgrid = np.zeros(nsites)
    grid[flat] = spins
    lgrid[flat] = lengths
    offset = np.abs(frac - np.rint(frac))[:, :2].max()
    a_dump = np.linalg.norm(primitive[0])
    return timestep, grid, lgrid, source, offset, a_dump


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="phonon_displace.py",
        description="Same spins in every frame, atoms displaced along a phonon mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("dump", type=Path, help="Dump that supplies the spins")
    p.add_argument("--frame", type=int, default=0, help="Frame of the dump to take the spins from")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("N1", "N2", "N3"),
                   help="Supercell of the dump; the ideal structure is built with the same repeats")
    p.add_argument("--spin-columns", nargs=3, default=["c_outsp[1]", "c_outsp[2]", "c_outsp[3]"],
                   metavar=("SX", "SY", "SZ"))
    p.add_argument("--length-column", default="sp_len",
                   help="Spin-length column, used when the dump has it")
    p.add_argument("--spin-length", type=float, default=2.0,
                   help="Spin length when the dump has no length column")
    p.add_argument("--band-yaml", type=Path, required=True, help="phonopy band.yaml")
    p.add_argument("--q", type=float, nargs=3, required=True, metavar=("H", "K", "L"),
                   help="q-point, fractional coordinates of the primitive reciprocal lattice")
    p.add_argument("--occurrence", type=int, default=1,
                   help="Which appearance of --q on the path, when it appears more than once")
    p.add_argument("--bands", type=int, nargs="+", required=True,
                   help="Band number(s) at that q-point, 1-based; several must be degenerate")
    p.add_argument("--degeneracy-tol", type=float, default=1e-3, metavar="THZ",
                   help="Largest frequency spread allowed among --bands")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--direction", nargs=4, metavar=("EL", "X", "Y", "Z"),
                   help="Combination of degenerate bands in which the first atom EL moves along "
                        "(X, Y, Z), Cartesian; this also sets the sign of a positive amplitude")
    g.add_argument("--mix-angle", type=float, metavar="DEG",
                   help="e = cos(DEG) e1 + sin(DEG) e2, for two degenerate bands")
    a = p.add_mutually_exclusive_group(required=True)
    a.add_argument("--amplitudes", type=float, nargs="+", help="Amplitude of every frame")
    a.add_argument("--amax", type=float, help="Largest amplitude; frames run from -amax to +amax")
    p.add_argument("--nframes", type=int, default=11, help="Frames for --amax (odd includes 0)")
    p.add_argument("--amp-unit", choices=["disp", "Q"], default="disp",
                   help="'disp': largest atomic displacement in Angstrom; "
                        "'Q': normal coordinate in Angstrom*sqrt(amu)")
    p.add_argument("--out", type=Path, default=Path("phonon.lammpstrj"), help="Output dump")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for f in (args.dump, args.band_yaml):
        if not f.exists():
            sys.exit(f"error: no such file: {f}")
    supercell = np.asarray(args.supercell, dtype=int)

    # ---- the mode ----------------------------------------------------------
    data = load_band_yaml(args.band_yaml)
    symbols = [pt["symbol"] for pt in data["points"]]
    masses = np.array([float(pt["mass"]) for pt in data["points"]])
    vectors, freqs = select_mode(data, args.q, args.occurrence, args.bands, args.degeneracy_tol)
    if args.direction is not None and len(args.bands) == 1:
        sys.exit("error: --direction chooses among degenerate bands; with one band the mode is "
                 "fixed (flip the sign of the amplitude instead)")

    # ---- align band.yaml with the make_spiral.py cell ------------------------
    # done before combining, so that --direction is read in the make_spiral (dump) frame
    lattice = np.asarray(data["lattice"], dtype=float)
    rotation = rotation_between(lattice, PRIM_CELL)
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    a_ph, a_ms = np.linalg.norm(lattice[0]), np.linalg.norm(PRIM_CELL[0])
    print(f"[CELL] a (band.yaml)  : {a_ph:.5f} A   a (make_spiral) : {a_ms:.5f} A   "
          f"({100 * (a_ms / a_ph - 1):+.2f}%)   rotation {angle:.3f} deg")
    if abs(a_ms / a_ph - 1) > 0.01:
        print("[WARN] lattice constants differ by more than 1%")
    vectors = np.einsum("ij,ajb->aib", rotation, vectors)    # Cartesian, make_spiral frame
    e = combine(vectors, masses, symbols, args)
    mapping, report = match_atoms(data, rotation)
    for j, pidx, sym, df, ddz in report:
        print(f"[CELL] basis atom {j + 1} ({sym}) <- band.yaml atom {pidx + 1}   "
              f"in-plane mismatch {df:.2e}, height mismatch {ddz:+.4f} A")

    # direction / sign: a positive amplitude moves the reference atom along +direction
    if args.direction is not None:
        ref_atom, d = direction_target(args.direction, symbols)
    else:
        ref_atom = int(np.argmax(np.linalg.norm(e.real, axis=1) / np.sqrt(masses)))
        d = e.real[ref_atom] / max(np.linalg.norm(e.real[ref_atom]), 1e-30)
    if (e.real[ref_atom] @ d) < 0:
        e = -e

    # ---- spins and the ideal lattice -----------------------------------------
    timestep, spin_grid, length_grid, source, offset, a_dump = spins_from_dump(
        args.dump, args.frame, supercell, args.spin_columns, MAGNETIC_TYPE,
        args.length_column, args.spin_length)
    print(f"[SPIN] dump           : {args.dump}  frame {args.frame} (timestep {timestep}), "
          f"{len(spin_grid)} spins, max site offset {offset:.3f} cells, lengths from {source}")
    if abs(a_dump / a_ms - 1) > 0.01:
        print(f"[WARN] the dump's lattice constant ({a_dump:.4f} A) differs from make_spiral's; "
              f"only the spins are used, the positions come from make_spiral")

    n1, n2, n3 = supercell
    nb = len(BASIS)
    cell = PRIM_CELL * supercell[:, None]
    cells = np.array([(i, j, m) for i in range(n1) for j in range(n2) for m in range(n3)])
    types = np.tile([t for t, *_ in BASIS], len(cells))
    basis_xyz = np.array([[x, y, z] for _, x, y, z in BASIS])
    positions0 = (cells @ PRIM_CELL)[:, None, :] + basis_xyz[None, :, :]
    positions0 = positions0.reshape(-1, 3)
    basis_of = np.tile(np.arange(nb), len(cells))
    cell_of = np.repeat(np.arange(len(cells)), nb)

    directions = np.tile([0.0, 0.0, 1.0], (len(types), 1))
    lengths = np.zeros(len(types))
    magnetic = types == MAGNETIC_TYPE
    site = np.ravel_multi_index(cells[cell_of[magnetic]].T, supercell)
    directions[magnetic] = spin_grid[site]
    lengths[magnetic] = length_grid[site]

    # ---- the displacement pattern ------------------------------------------
    q = np.asarray(args.q, dtype=float)
    turns = q * supercell
    if np.abs(turns - np.rint(turns)).max() > 1e-6:
        sys.exit(f"error: q = {q.tolist()} is not commensurate with the {tuple(supercell)} supercell")
    phase = np.exp(2j * np.pi * (cells @ q))                         # per cell
    m_of = masses[np.array(mapping)][basis_of]
    e_of = e[np.array(mapping)][basis_of]
    pattern = np.real(e_of * phase[cell_of][:, None]) / np.sqrt(m_of)[:, None]   # u for Q = 1
    peak = np.linalg.norm(pattern, axis=1).max()
    if peak < 1e-12:
        sys.exit("error: the selected mode gives no displacement on this lattice")

    # self-checks on the unit pattern
    print("[MODE] displacement per atom for Q = 1 A*sqrt(amu)  (Cartesian, A):")
    for j in range(nb):
        u = pattern[basis_of == j][0]
        ang = np.degrees(np.arctan2(u[1], u[0]))
        print(f"         basis {j + 1} {ELEMENTS[BASIS[j][0]]:2s}: ({u[0]:+.5f}, {u[1]:+.5f}, {u[2]:+.5f})"
              f"   |u| = {np.linalg.norm(u):.5f}   in-plane angle {ang:+7.2f} deg")
    com = (pattern * m_of[:, None]).sum(axis=0) / m_of.sum()
    print(f"[CHECK] centre-of-mass shift per unit Q: ({com[0]:+.2e}, {com[1]:+.2e}, {com[2]:+.2e}) A"
          + ("   (Gamma optical mode: should be ~0)" if np.allclose(q, 0) else ""))
    if args.direction is not None:
        u_ref = pattern[basis_of == mapping.index(ref_atom)][0]
        miss = np.degrees(np.arccos(np.clip(u_ref @ d / np.linalg.norm(u_ref), -1, 1)))
        print(f"[CHECK] {symbols[ref_atom]} moves {miss:.3f} deg away from the requested direction")
        if miss > 1.0:
            sys.exit(f"error: no combination of bands {args.bands} moves {symbols[ref_atom]} along "
                     f"{np.round(d, 4).tolist()}; the closest is {miss:.1f} deg off")

    # ---- frames ---------------------------------------------------------------
    amps = (np.asarray(args.amplitudes, dtype=float) if args.amplitudes is not None
            else np.linspace(-args.amax, args.amax, args.nframes))
    Q = amps / peak if args.amp_unit == "disp" else amps

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    table = []
    is_ni = np.array([ELEMENTS[t] == "Ni" for t in types])
    for k, qk in enumerate(Q):
        disp = qk * pattern
        write_dump(args.out, cell, types, positions0 + disp, directions, lengths,
                   timestep=k, mode="a")
        norms = np.linalg.norm(disp, axis=1)
        table.append((k, qk, norms.max(),
                      norms[is_ni].max() if is_ni.any() else 0.0,
                      norms[~is_ni].max() if (~is_ni).any() else 0.0))
    amp_file = args.out.with_suffix(".amp.dat")
    np.savetxt(amp_file, np.array(table),
               fmt="%d %+.8f %.8f %.8f %.8f",
               header=f"frame  Q_Ang_sqrtamu  max_disp_Ang  max_disp_Ni  max_disp_other   "
                      f"(q = {q.tolist()}, bands {args.bands}, {', '.join(f'{f:.4f}' for f in freqs)} THz)")

    print(f"[OUT] {len(Q)} frames, {len(types)} atoms each -> {args.out}")
    print(f"[OUT] amplitudes      : Q from {Q.min():+.5f} to {Q.max():+.5f} A*sqrt(amu), "
          f"largest displacement {np.abs(Q).max() * peak:.5f} A  -> {amp_file}")


if __name__ == "__main__":
    main()
