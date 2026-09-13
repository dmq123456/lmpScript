#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bound charge of the spin-current polarization, rho = -div P.

`polarization.py` gives a dipole moment p_i per site, in e Angstrom, summing to
the polarization of the layer. Treating those as samples of a smooth field, the
bound charge carried by one cell is

    rho_i = -div p (r_i)        [e],

with no cell area in the conversion: p is a dipole per cell already, so its
divergence is a charge per cell. On a periodic layer sum_i rho_i vanishes
identically, which is this module's counterpart of the integer total that
topocharge.py checks itself against.

A uniform spiral has a uniform p, so rho is zero throughout it. What rho finds
is the places where the texture is *not* uniform: the walls between Q domains
and between the two handednesses. There a head-to-head arrangement of P carries
net bound charge and a side-by-side one does not, which is what decides the
electrostatic cost of a wall and therefore which wall orientations survive.

Only the in-plane divergence exists for a single layer, so p_z never enters
rho. An out-of-plane dipole is a dipole layer, not a bulk charge, and shows up
as a sheet charge on the two faces instead; look at it through --color pz.

Two discretisations, both exactly neutral
-----------------------------------------
The default is a six-neighbour central difference. Writing delta_d for the six
nearest-neighbour vectors and M = sum_d delta_d delta_d^T (which is 3 a^2 I on
a triangular lattice),

    div p (r_i) = sum_d (M^-1 delta_d) . p_{i+d} + O(a^2).

It is second order, it rings nowhere, and because delta_{-d} = -delta_d the
weights are odd, so summing rho over any region telescopes down to a shell of
sites along the region's boundary -- a discrete divergence theorem that holds to
machine precision, not just in the continuum. That is what makes the *integrated*
charge of a wall meaningful (see below).

`divergence_spectral` is the alternative: exact for a band-limited field, and
used here as the reference the stencil is checked against. It is not the
default because the interesting textures have walls a few cells wide, and a
spectral derivative decorates those with Gibbs ringing that looks exactly like
alternating bound charge.

What is and is not convention independent
-----------------------------------------
`site_polarization` splits each bond dipole equally between the bond's two
ends. A different split leaves the total P alone but moves dipole around inside
a cell, which changes p_i and hence changes rho_i. For a smooth texture the
difference is O(a^2) and invisible; across a wall one cell wide it is O(1).

So a single site's rho_i is not a physical number. The integral over a region
is, because by the divergence theorem it equals a flux of P through the
region's boundary and P itself is unambiguous -- and the stencil satisfies that
theorem exactly. Report the charge per unit length of a wall, obtained from
`charge_profile`, and treat the per-site map as a way of locating the wall
rather than as a measurement.

Run this file directly to execute the self-test; run it on a dump to write the
charge profile of one frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from geometry import reciprocal_lattice_from_real  # noqa: E402
from lattice_grid import grid_from_cfg, layer_grids  # noqa: E402
from polarization import (  # noqa: E402
    _lattice_from_frame,
    frame_site_polarization,
)

#: The six nearest neighbours as (p, q) steps in cell indices, i.e. p a1 + q a2.
NEIGHBOUR_OFFSETS = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))

DEFAULT_METHOD = "stencil"


def _primitive(lattice: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """The primitive cell of a layer, from the box and the repeat counts."""
    lattice = np.asarray(lattice, dtype=float)
    return np.array([lattice[0] / n1, lattice[1] / n2, lattice[2]])


def stencil_weights(primitive: np.ndarray) -> np.ndarray:
    """(6, 2) weights w_d with div p = sum_d w_d . p_{i+d}.

    w_d = M^-1 delta_d for M = sum_d delta_d delta_d^T, which is the least-
    squares gradient of a linear field and reduces to d_hat / (3 a) on an
    equilateral lattice. It is written out in this general form so that a cell
    which is not exactly equilateral -- a strained supercell, say -- still gets
    a consistent first derivative instead of a silently wrong prefactor.
    """
    deltas = np.array([p * primitive[0] + q * primitive[1] for p, q in NEIGHBOUR_OFFSETS])
    moment = deltas[:, :2].T @ deltas[:, :2]
    if abs(np.linalg.det(moment)) < 1e-12 * np.trace(moment) ** 2:
        raise ValueError(
            "The six neighbour vectors are degenerate in the plane, so no in-plane "
            "gradient can be formed from them. The layer is probably one-dimensional."
        )
    return np.linalg.solve(moment, deltas[:, :2].T).T


def divergence_stencil(p: np.ndarray, site_of: np.ndarray,
                       primitive: np.ndarray) -> np.ndarray:
    """div p on one layer, by the six-neighbour central difference."""
    weights = stencil_weights(primitive)
    flat = site_of.ravel()
    out = np.zeros(p.shape[0], dtype=float)
    for (u, v), w in zip(NEIGHBOUR_OFFSETS, weights):
        neighbour = np.roll(site_of, (-u, -v), axis=(0, 1)).ravel()
        out[flat] += p[neighbour][:, :2] @ w
    return out


def divergence_spectral(p: np.ndarray, site_of: np.ndarray,
                        primitive: np.ndarray) -> np.ndarray:
    """div p on one layer, by FFT. Exact for a band-limited field.

    The Nyquist row is zeroed: that mode is its own negative on the grid, so its
    derivative is not defined by the samples and keeping it would make the
    result complex rather than merely inaccurate.
    """
    n1, n2 = site_of.shape
    reciprocal = reciprocal_lattice_from_real(np.asarray(primitive, dtype=float))
    m1 = np.fft.fftfreq(n1) * n1
    m2 = np.fft.fftfreq(n2) * n2
    q = ((m1[:, None, None] / n1) * reciprocal[0]
         + (m2[None, :, None] / n2) * reciprocal[1])

    grid = p[site_of]                                    # (n1, n2, 3)
    spectrum = np.fft.fft2(grid, axes=(0, 1))
    divergence = 1j * np.einsum("ijk,ijk->ij", q, spectrum)
    if n1 % 2 == 0:
        divergence[n1 // 2, :] = 0.0
    if n2 % 2 == 0:
        divergence[:, n2 // 2] = 0.0

    out = np.zeros(p.shape[0], dtype=float)
    out[site_of.ravel()] = np.real(np.fft.ifft2(divergence, axes=(0, 1))).ravel()
    return out


_METHODS = {"stencil": divergence_stencil, "spectral": divergence_spectral}


def layer_charge(p: np.ndarray, site_of: np.ndarray, primitive: np.ndarray,
                 method: str = DEFAULT_METHOD) -> np.ndarray:
    """rho = -div p for one layer, one value per layer-local site, in e."""
    if method not in _METHODS:
        raise ValueError(f"Unknown divergence method {method!r}; expected one of "
                         f"{sorted(_METHODS)}.")
    return -_METHODS[method](np.asarray(p, dtype=float), site_of, primitive)


# ----------------------------------------------------------------------
# Frame interface
# ----------------------------------------------------------------------
def site_charge(frame: dict, cfg: dict | None = None) -> np.ndarray:
    """Bound charge per site for one frame, in e, summing to zero on each layer."""
    cfg = cfg or {}
    p = frame_site_polarization(frame, cfg)
    lattice = _lattice_from_frame(frame)
    positions = np.stack([frame["x"], frame["y"], frame["z"]], axis=1).astype(float)
    method = cfg.get("divergence", DEFAULT_METHOD)

    out = np.zeros(p.shape[0], dtype=float)
    for where, site_of, n1, n2 in layer_grids(
        positions, lattice, bool(cfg.get("single_layer", True)), grid_from_cfg(cfg)
    ):
        out[where] = layer_charge(p[where], site_of, _primitive(lattice, n1, n2), method)
    return out


def charge_profile(rho: np.ndarray, site_of: np.ndarray,
                   axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(charge per lattice row, running total) along one cell axis.

    Rows of constant cell index along `axis` are summed, so the result is the
    bound charge of a wall that runs across the box -- the quantity that does
    not depend on how bond dipoles were split between sites. The running total
    returns to zero at the far edge, since the layer is neutral.

    Mind which direction this actually resolves. A texture that is uniform along
    a2 varies with the a1 *index*, and the normal of those planes is the
    reciprocal vector b1, which on a hexagonal lattice is 30 degrees away from
    a1 itself. The charge accumulated across such a wall is

        sum_i rho_row = -(b1 . [P_row(right) - P_row(left)]) / 2 pi,

    so the in-plane component of P perpendicular to a1 contributes too. Dividing
    a polarization jump by |a1| and keeping only its x component -- the obvious
    thing to write -- is wrong on both counts.
    """
    per_row = rho[site_of].sum(axis=1 - axis)
    return per_row, np.cumsum(per_row)


__all__ = [
    "DEFAULT_METHOD",
    "NEIGHBOUR_OFFSETS",
    "charge_profile",
    "divergence_spectral",
    "divergence_stencil",
    "layer_charge",
    "site_charge",
    "stencil_weights",
]


# ----------------------------------------------------------------------
# Command line: the charge profile of one frame
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boundcharge.py",
        description="Write the bound-charge profile of one frame of a dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dump", type=Path, help="Input LAMMPS dump file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--vector", nargs=3, required=True, metavar=("SX", "SY", "SZ"),
                        help="The three spin columns")
    parser.add_argument("--element", type=str, default="all",
                        help="Element symbol to include, or 'all'")
    parser.add_argument("--drop-zero-vector", action="store_true",
                        help="Skip atoms whose spin is exactly zero")
    parser.add_argument("--single-layer", action="store_true", help="Do not split at z")
    parser.add_argument("--shells", type=int, nargs="+", default=[1, 3], metavar="N",
                        help="Neighbour shells summed into P")
    parser.add_argument("--spin-length", type=float, default=1.0, metavar="S")
    parser.add_argument("--divergence", choices=sorted(_METHODS), default=DEFAULT_METHOD,
                        help="Discretisation of div P")
    parser.add_argument("--axis", type=int, choices=(0, 1), default=0,
                        help="Cell axis to profile along; choose it along the wall normal")
    parser.add_argument("--lattice-grid", "--topo-grid", type=int, nargs=2,
                        dest="lattice_grid", metavar=("N1", "N2"), default=None)
    parser.add_argument("--out", type=Path, default=Path("boundcharge.dat"))
    return parser


def main(argv=None) -> None:
    from dumpframe import load_single_frame

    args = build_parser().parse_args(argv)
    cfg = {
        "single_layer": args.single_layer,
        "shells": tuple(args.shells),
        "spin_length": args.spin_length,
        "divergence": args.divergence,
        "lattice_grid": tuple(args.lattice_grid) if args.lattice_grid else None,
    }
    frame = load_single_frame(args.input_dump, args.frame, vector=tuple(args.vector),
                              element=args.element, drop_zero_vector=args.drop_zero_vector)
    rho = site_charge(frame, cfg)
    lattice = _lattice_from_frame(frame)
    positions = np.stack([frame["x"], frame["y"], frame["z"]], axis=1).astype(float)
    grids = layer_grids(positions, lattice, args.single_layer, cfg["lattice_grid"])

    print(f"[INFO] Frame {args.frame}, timestep {int(frame['timestep'])}, "
          f"{rho.size} site(s), {len(grids)} layer(s), {args.divergence} divergence")
    with args.out.open("w") as fh:
        fh.write(f"# {' '.join(sys.argv)}\n")
        fh.write("# bound charge in e; the running total returns to zero on a neutral layer\n")
        fh.write("# a row is one line of sites at fixed cell index along --axis; "
                 "distance is measured\n#   along the row normal (the reciprocal direction), "
                 "whose spacing is 2 pi / |b|\n")
        fh.write(f"# {'layer':>6s}  {'row':>6s}  {'distance':>14s}  {'charge':>14s}  "
                 f"{'running':>14s}\n")
        for index, (where, site_of, n1, n2) in enumerate(grids):
            primitive = _primitive(lattice, n1, n2)
            per_row, running = charge_profile(rho[where], site_of, args.axis)
            # Consecutive rows are one lattice vector apart, but the distance
            # that matters along a profile is the perpendicular spacing of the
            # rows, 2 pi / |b|, which on a hexagonal lattice is |a| sqrt(3)/2.
            step = 2.0 * np.pi / np.linalg.norm(
                reciprocal_lattice_from_real(primitive)[args.axis])
            print(f"[INFO] layer {index + 1}: sum rho = {rho[where].sum(): .3e} e, "
                  f"max |running total| = {np.max(np.abs(running)): .3e} e")
            for row, (charge, total) in enumerate(zip(per_row, running)):
                fh.write(f"  {index + 1:>6d}  {row:>6d}  {row * step:>14.6f}  "
                         f"{charge:>14.6e}  {total:>14.6e}\n")
    print(f"[INFO] profile written to {args.out}")


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _modulated_spiral(positions, lattice, h, k, turns):
    """A spiral whose rotation plane tips as it advances, so P varies in space."""
    frac = positions @ np.linalg.inv(lattice)
    phase = 2.0 * np.pi * (h * frac[:, 0] + k * frac[:, 1])
    tilt = 2.0 * np.pi * turns * frac[:, 0]
    normal = np.stack([np.cos(tilt), np.zeros_like(tilt), np.sin(tilt)], axis=1)
    e1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(normal, e1)
    return np.cos(phase)[:, None] * e1 + np.sin(phase)[:, None] * e2


def _selftest() -> int:
    from polarization import (_triangular_lattice, build_families, family_polarization,
                              site_polarization)

    rng = np.random.default_rng(0)
    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failures += 0 if ok else 1

    def grid_of(positions, lattice):
        (_, site_of, n1, n2), = layer_grids(positions, lattice, True, None)
        return site_of, _primitive(lattice, n1, n2)

    a = 3.9793

    # -- the stencil itself ---------------------------------------------------
    positions, lattice = _triangular_lattice(24, 24, a=a)
    site_of, primitive = grid_of(positions, lattice)
    weights = stencil_weights(primitive)
    expected = np.array([(p * primitive[0] + q * primitive[1])[:2] / (3.0 * a ** 2)
                         for p, q in NEIGHBOUR_OFFSETS])
    check("stencil reduces to d_hat / 3a", np.allclose(weights, expected, atol=1e-12),
          "M = 3 a^2 I on an equilateral lattice")
    check("stencil weights are odd",
          np.allclose(weights[:3], -weights[3:], atol=1e-15),
          "w(-d) = -w(d), which is what makes the region sum telescope")

    # -- a field whose divergence is known ------------------------------------
    supercell = reciprocal_lattice_from_real(lattice)
    g1 = 1 * supercell[0] + 1 * supercell[1]
    g2 = 1 * supercell[0] - 1 * supercell[1]

    def p_field(r):
        return np.stack([np.sin(r @ g1), np.cos(r @ g2), np.zeros(len(r))], axis=1)

    def rho_exact(r):
        return -(g1[0] * np.cos(r @ g1) - g2[1] * np.sin(r @ g2))

    p = p_field(positions)
    reference = rho_exact(positions)
    scale = float(np.max(np.abs(reference)))
    errors = {}
    for method in sorted(_METHODS):
        rho = layer_charge(p, site_of, primitive, method)
        errors[method] = float(np.max(np.abs(rho - reference))) / scale
        check(f"{method} is neutral", abs(float(rho.sum())) < 1e-12 * scale,
              f"sum rho = {float(rho.sum()):+.2e} e")
    check("spectral is exact", errors["spectral"] < 1e-12,
          f"max relative error {errors['spectral']:.2e}")
    check("stencil is second order accurate", errors["stencil"] < 5e-2,
          f"max relative error {errors['stencil']:.2e} at this wavelength")

    coarse = errors["stencil"]
    positions2, lattice2 = _triangular_lattice(48, 48, a=a)
    site_of2, primitive2 = grid_of(positions2, lattice2)
    supercell2 = reciprocal_lattice_from_real(lattice2)
    g1b, g2b = supercell2[0] + supercell2[1], supercell2[0] - supercell2[1]
    p2 = np.stack([np.sin(positions2 @ g1b), np.cos(positions2 @ g2b),
                   np.zeros(len(positions2))], axis=1)
    ref2 = -(g1b[0] * np.cos(positions2 @ g1b) - g2b[1] * np.sin(positions2 @ g2b))
    fine = float(np.max(np.abs(layer_charge(p2, site_of2, primitive2, "stencil") - ref2)))
    fine /= float(np.max(np.abs(ref2)))
    check("stencil error falls as a^2", 3.0 < coarse / fine < 5.0,
          f"doubling the wavelength divides the error by {coarse / fine:.2f}, 4 expected")

    # -- uniform P, the case that has to be exactly zero ----------------------
    uniform = np.tile(rng.normal(size=3), (positions.shape[0], 1))
    for method in sorted(_METHODS):
        rho = layer_charge(uniform, site_of, primitive, method)
        check(f"uniform P gives no charge ({method})",
              float(np.max(np.abs(rho))) < 1e-15 * float(np.max(np.abs(uniform))),
              f"max |rho| = {float(np.max(np.abs(rho))):.2e}")

    # -- the discrete divergence theorem --------------------------------------
    # The charge of a region must be reconstructible from a shell of sites along
    # its boundary alone. That is what lets a wall's charge be quoted at all.
    rho = layer_charge(p, site_of, primitive, "stencil")
    block = np.zeros(site_of.shape, dtype=bool)
    block[5:17, 3:20] = True
    inside = float(rho[site_of[block]].sum())
    boundary = 0.0
    for (u, v), w in zip(NEIGHBOUR_OFFSETS, weights):
        shifted = np.roll(block, (u, v), axis=(0, 1))
        entering = p[site_of[shifted & ~block]][:, :2].sum(axis=0)
        leaving = p[site_of[block & ~shifted]][:, :2].sum(axis=0)
        boundary -= w @ (entering - leaving)
    check("region charge is a boundary sum", abs(inside - boundary) < 1e-12 * abs(inside),
          f"interior {inside:+.6e} e against boundary-only {boundary:+.6e} e")

    # -- the trap this module exists to avoid ---------------------------------
    # Representing each bond dipole by a charge at either end keeps neutrality
    # and looks right, but it throws away the transverse half of every bond and
    # comes out at exactly half the true divergence on a triangular lattice.
    families = build_families(*positions2.T, lattice2, single_layer=True, shells=(1,))
    bond_field = p2 / 3.0
    assembled = np.zeros_like(p2)
    naive = np.zeros(p2.shape[0])
    for family in families[0]:
        direction = np.array([np.cos(np.radians(family.angle)),
                              np.sin(np.radians(family.angle)), 0.0])
        bond = bond_field[family.site_i] * 0.5 + bond_field[family.site_j] * 0.5
        assembled[family.site_i] += 0.5 * bond
        assembled[family.site_j] += 0.5 * bond
        transfer = (bond @ direction) / a
        np.add.at(naive, family.site_i, -transfer)
        np.add.at(naive, family.site_j, transfer)
    proper = layer_charge(assembled, site_of2, primitive2, "stencil")
    ratio = float(np.polyfit(naive, proper, 1)[0])
    check("bond-end charges come out at half", abs(ratio - 2.0) < 0.05,
          f"true / naive = {ratio:.4f}, because (1/6) sum_d d d^T = I/2")

    # -- rho is a scalar ------------------------------------------------------
    from polarization import d3d_operations
    worst = 0.0
    for rotation in d3d_operations():
        rotated_positions = positions @ rotation.T
        rotated_lattice = lattice @ rotation.T
        site_of_r, primitive_r = grid_of(rotated_positions, rotated_lattice)
        rotated_p = p @ rotation.T                      # P is polar
        got = layer_charge(rotated_p, site_of_r, primitive_r, "stencil")
        worst = max(worst, float(np.max(np.abs(got - rho))))
    check("rho is invariant under D3d", worst < 1e-12 * scale,
          f"max deviation {worst:.2e}; P and grad are both odd, so rho is a true scalar")

    # -- a real spin texture --------------------------------------------------
    families = build_families(*positions.T, lattice, single_layer=True)
    basis = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    frac = positions @ np.linalg.inv(lattice)
    phase = 2.0 * np.pi * 3.0 * frac[:, 0]
    uniform_spiral = (np.cos(phase)[:, None] * basis[0] + np.sin(phase)[:, None] * basis[1])
    p_spiral = site_polarization(uniform_spiral, families, positions.shape[0])
    rho_spiral = layer_charge(p_spiral, site_of, primitive, "stencil")
    check("a uniform spiral carries no bound charge",
          float(np.max(np.abs(rho_spiral))) < 1e-18,
          f"max |rho| = {float(np.max(np.abs(rho_spiral))):.2e} e, P being uniform")

    modulated = _modulated_spiral(positions, lattice, 3, 0, turns=1)
    p_mod = site_polarization(modulated, families, positions.shape[0])
    rho_mod = layer_charge(p_mod, site_of, primitive, "stencil")
    peak = float(np.max(np.abs(rho_mod)))
    check("a modulated spiral does carry bound charge", peak > 1e-8,
          f"max |rho| = {peak:.3e} e per cell where the spiral plane tips")
    check("the modulated texture is still neutral", abs(float(rho_mod.sum())) < 1e-12 * peak,
          f"sum rho = {float(rho_mod.sum()):+.2e} e")

    per_row, running = charge_profile(rho_mod, site_of, axis=0)
    check("the charge profile closes", abs(float(running[-1])) < 1e-12 * float(np.max(np.abs(running))),
          f"running total ends at {float(running[-1]):+.2e} e after "
          f"reaching {float(np.max(np.abs(running))):.3e} e")

    # -- the profile measures a flux along b, not along a ---------------------
    # For a texture uniform along a2 the stencil collapses exactly onto
    # rho_row(i) = -(b1 . [P_row(i+1) - P_row(i-1)]) / 4 pi, which is the
    # identity a wall charge should be quoted from.
    stripe = np.zeros_like(positions)
    index = np.rint((positions @ np.linalg.inv(lattice))[:, 0] * 24).astype(int) % 24
    for component, wave in enumerate((1.0, -0.7, 0.3)):
        stripe[:, component] = wave * np.sin(2 * np.pi * (component + 1) * index / 24)
    rho_stripe = layer_charge(stripe, site_of, primitive, "stencil")
    per_row, _ = charge_profile(rho_stripe, site_of, axis=0)
    b1 = reciprocal_lattice_from_real(primitive)[0]
    row_dipole = stripe[site_of].sum(axis=1)
    predicted = -(np.roll(row_dipole, -1, axis=0) - np.roll(row_dipole, 1, axis=0)) @ b1
    predicted /= 4.0 * np.pi
    check("the row profile is a flux through b1",
          np.allclose(per_row, predicted, atol=1e-12 * float(np.max(np.abs(per_row)))),
          f"max deviation {float(np.max(np.abs(per_row - predicted))):.2e}; the wall "
          "normal is b1 at 30 deg to a1, and P perpendicular to a1 contributes")

    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    return failures


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        raise SystemExit(_selftest())
