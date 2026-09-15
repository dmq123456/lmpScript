#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static structure factor of one frame, on the whole two-dimensional zone.

    s^a(q) = (1/sqrt(N)) sum_j S_j^a exp(+i q.r_j)
    S(q)   = sum_ab W_ab s^a(q) s^{b*}(q)

For a periodic supercell this is not an integral to sample but a finite,
complete object: the n1 x n2 magnetic sites carry exactly n1 x n2 independent
wavevectors,

    q_m = (m1/n1) B1 + (m2/n2) B2,   m in [0,n),

with B the *primitive* reciprocal vectors, and s(q) at any other wavevector is
an interpolation of these rather than new information. So there is no q-grid
density to choose: it is fixed by how large a supercell was replicated.

That completeness is also why the sum is not evaluated point by point here. On
the allowed grid it is exactly a two-dimensional inverse FFT of the spin field
laid out on the cell indices,

    s^a(q_m) = sqrt(N) * ifft2(S^a)[m1, m2],

which at 96 x 96 replaces 4.6 s of explicit summation with 0.46 ms. `corr`'s
`fields.spatial_fourier_transform` remains the right tool for an arbitrary
q-path, and `_selftest` checks the two against each other rather than trusting
this shortcut on its own.

What comes out
--------------
    q_grid    (n1, n2, 3)  Cartesian q of every grid point -- geometry only,
                           so it is built once per box and cached
    s_q       (n1, n2, 3)  complex mode amplitude; the one expensive quantity
    intensity (n1, n2)     a real scalar per q, a cheap projection of s_q

Changing channel re-projects s_q, it does not recompute the transform.

Channels
--------
The `--component` grammar of `channels.py` is reused verbatim, so `total` (an
alias for `1+5+9`), `xx`, `zz`, `1+5+9`, `L` and `T` all mean here what they
mean for S(q,w). L and T have to be evaluated differently, though: every grid
point carries its own q_hat, and `channel_value` applies a single weight matrix
to a whole axis. They are therefore vectorised here from the decomposition
W = const + n_L qq^T + n_T (I - qq^T), and `_selftest` checks the constant
channels against `channel_value` to keep the two paths honest.

Setting q_hat to zero at q = 0 reproduces exactly the Gamma-point convention of
`channels.py` -- L contributes nothing and T becomes the full trace -- so the
origin needs no special case.

A caution on supercell shape
----------------------------
The allowed-q set is closed under the three-fold rotation only when n1 == n2.
A 30 x 18 cell drops 504 of its 540 wavevectors off their own C3 images, which
leaves the three spiral domains inequivalent on the grid and quietly biases any
measure of which one is selected. Constructing such a grid warns.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from channels import Channel, parse_channels  # noqa: E402
from geometry import reciprocal_lattice_from_real  # noqa: E402
from lattice_grid import (  # noqa: E402
    grid_from_cfg,
    lattice_from_frame,
    layer_grids,
)

#: Names accepted in addition to the channels.py grammar.
CHANNEL_ALIASES = {"total": "1+5+9", "trace": "1+5+9"}

DEFAULT_CHANNEL = "total"

#: q_hat probes used to decide whether a channel is a genuine intensity.
_PSD_PROBES = (None, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
               np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def primitive_lattice(lattice: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """The primitive cell of a layer, from the box and the repeat counts."""
    lattice = np.asarray(lattice, dtype=float)
    return np.array([lattice[0] / n1, lattice[1] / n2, lattice[2]])


def allowed_q(lattice: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """(n1, n2, 3) Cartesian q of the allowed wavevectors, in FFT index order.

    Index order matches what `amplitude` returns, so the two line up without a
    shift; use `centre` when the map is to be drawn around Gamma.
    """
    if n1 != n2:
        warnings.warn(
            f"A {n1} x {n2} supercell gives an allowed-q set that is not closed under "
            "the three-fold rotation, so the three spiral domains are not equivalent on "
            "this grid and any selectivity measured from it is biased. Use n1 == n2.",
            stacklevel=2,
        )
    reciprocal = reciprocal_lattice_from_real(primitive_lattice(lattice, n1, n2))
    m1 = np.fft.fftfreq(n1) * n1
    m2 = np.fft.fftfreq(n2) * n2
    return ((m1[:, None, None] / n1) * reciprocal[0]
            + (m2[None, :, None] / n2) * reciprocal[1])


def brillouin_zone_polygon(reciprocal: np.ndarray) -> np.ndarray:
    """(6, 2) vertices of the first Brillouin zone in the plane, ordered by angle.

    The zone is the intersection of the half-planes q.G <= |G|^2 / 2 over the
    shortest reciprocal vectors, so its corners are where consecutive bisectors
    meet. Candidates are searched over a small integer range rather than assumed
    to be b1, b2 and b1 + b2: on a hexagonal lattice b1 - b2 is short and
    b1 + b2 is not, and hard-coding the wrong pair gives a plausible but wrong
    hexagon.
    """
    b1, b2 = np.asarray(reciprocal, float)[0][:2], np.asarray(reciprocal, float)[1][:2]
    candidates = [m * b1 + n * b2
                  for m in range(-2, 3) for n in range(-2, 3) if (m, n) != (0, 0)]
    candidates.sort(key=lambda g: float(np.hypot(*g)))
    shortest = sorted(candidates[:6], key=lambda g: float(np.arctan2(g[1], g[0])))

    vertices = []
    for g, h in zip(shortest, shortest[1:] + shortest[:1]):
        matrix = np.array([g, h])
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Degenerate reciprocal lattice: no Brillouin zone to draw.")
        vertices.append(np.linalg.solve(matrix, 0.5 * np.array([g @ g, h @ h])))
    return np.array(vertices)


def centre(array: np.ndarray) -> np.ndarray:
    """Move Gamma from the corner to the middle, for drawing."""
    return np.fft.fftshift(array, axes=(0, 1))


def tile(q_grid: np.ndarray, values: np.ndarray, repeats: int = 1):
    """Replicate the map over neighbouring reciprocal cells.

    s(q + G) = s(q) exactly for a primitive reciprocal vector G when there is
    one magnetic site per primitive cell, so this adds no approximation. It is
    needed because the allowed q fill a parallelogram while the first zone is a
    hexagon, parts of which lie outside it.
    """
    if repeats < 0:
        raise ValueError("repeats must be non-negative")
    n1, n2 = values.shape[:2]
    reciprocal_step = np.array([q_grid[1 % n1, 0] - q_grid[0, 0],
                                q_grid[0, 1 % n2] - q_grid[0, 0]])
    span = 2 * repeats + 1
    q_out = np.empty((span * n1, span * n2, 3), dtype=float)
    v_out = np.tile(values, (span, span) + (1,) * (values.ndim - 2))
    for a in range(span):
        for b in range(span):
            shift = ((a - repeats) * n1 * reciprocal_step[0]
                     + (b - repeats) * n2 * reciprocal_step[1])
            q_out[a * n1:(a + 1) * n1, b * n2:(b + 1) * n2] = q_grid + shift
    return q_out, v_out


# ----------------------------------------------------------------------
# The transform and its projections
# ----------------------------------------------------------------------
def amplitude(spins: np.ndarray, site_of: np.ndarray) -> np.ndarray:
    """s^a(q) on the allowed grid. Returns (n1, n2, 3) complex128.

    numpy's ifft2 carries the exp(+2 pi i ...) sign and a 1/N, so sqrt(N) times
    it is exactly the sum(1/sqrt(N)) exp(+i q.r) convention that corr uses.
    """
    spins = np.asarray(spins, dtype=float)
    n1, n2 = site_of.shape
    grid = spins[site_of]                                   # (n1, n2, 3)
    return np.fft.ifft2(grid, axes=(0, 1)) * np.sqrt(float(n1 * n2))


def as_channel(channel) -> Channel:
    if isinstance(channel, Channel):
        return channel
    token = str(channel)
    return parse_channels([CHANNEL_ALIASES.get(token.lower(), token)])[0]


def project(s_q: np.ndarray, q_grid: np.ndarray, channel=DEFAULT_CHANNEL) -> np.ndarray:
    """Contract s^a s^{b*} with a channel weight. Returns (n1, n2) real."""
    ch = as_channel(channel)
    out = np.zeros(s_q.shape[:2], dtype=float)

    weight = ch.constant_weight
    if np.any(weight):
        out += np.real(np.einsum("ija,ijb,ab->ij", s_q, s_q.conj(), weight))

    if ch.n_longitudinal or ch.n_transverse:
        norm = np.linalg.norm(q_grid, axis=-1)[..., None]
        q_hat = np.divide(q_grid, norm, out=np.zeros_like(q_grid), where=norm > 0.0)
        longitudinal = np.abs(np.einsum("ija,ija->ij", q_hat, s_q)) ** 2
        if ch.n_longitudinal:
            out += ch.n_longitudinal * longitudinal
        if ch.n_transverse:
            total = np.real(np.einsum("ija,ija->ij", s_q, s_q.conj()))
            out += ch.n_transverse * (total - longitudinal)

    if all(ch.clips(probe) for probe in _PSD_PROBES):
        np.maximum(out, 0.0, out=out)
    return out


# ----------------------------------------------------------------------
# Frame interface
# ----------------------------------------------------------------------
_GEOMETRY_CACHE: dict[tuple, list] = {}


def layer_geometry(frame: dict, cfg: dict | None = None) -> list:
    """[(where, site_of, q_grid, reciprocal)] per layer, built once per box."""
    cfg = cfg or {}
    lattice = lattice_from_frame(frame)
    single_layer = bool(cfg.get("single_layer", True))
    grid = grid_from_cfg(cfg)
    n_sites = int(frame["x"].size)

    key = (n_sites, single_layer, tuple(grid) if grid else None,
           np.round(lattice, 6).tobytes())
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        positions = np.stack([frame["x"], frame["y"], frame["z"]], axis=1).astype(float)
        cached = []
        for where, site_of, n1, n2 in layer_grids(positions, lattice, single_layer, grid):
            cached.append((
                where, site_of, allowed_q(lattice, n1, n2),
                reciprocal_lattice_from_real(primitive_lattice(lattice, n1, n2)),
            ))
        _GEOMETRY_CACHE[key] = cached
    return cached


def _spins(frame: dict) -> np.ndarray:
    if "u" not in frame:
        raise ValueError(
            "The structure factor needs the spin direction, so --vector is required, "
            "e.g. --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]'."
        )
    return np.stack([frame["u"], frame["v"], frame["w"]], axis=1).astype(float)


def frame_intensity(frame: dict, cfg: dict | None = None) -> list:
    """[(q_grid, intensity, reciprocal)] per layer for one frame."""
    cfg = cfg or {}
    channel = cfg.get("channel", DEFAULT_CHANNEL)
    spins = _spins(frame)
    out = []
    for where, site_of, q_grid, reciprocal in layer_geometry(frame, cfg):
        s_q = amplitude(spins[where], site_of)
        out.append((q_grid, project(s_q, q_grid, channel), reciprocal))
    return out


def average_intensity(frames, cfg: dict | None = None) -> list:
    """Frame-averaged S(q), one entry per layer.

    A single snapshot is one thermal realisation and is noisy; averaging over a
    window shorter than the process being watched but longer than a magnon
    period cuts that noise as 1/sqrt(W) without blurring the dynamics.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("No frames to average.")
    accumulated = None
    for frame in frames:
        maps = frame_intensity(frame, cfg)
        if accumulated is None:
            accumulated = [[q, values.copy(), b] for q, values, b in maps]
        else:
            for slot, (_, values, _) in zip(accumulated, maps):
                slot[1] += values
    for slot in accumulated:
        slot[1] /= float(len(frames))
    return [tuple(slot) for slot in accumulated]


__all__ = [
    "CHANNEL_ALIASES",
    "DEFAULT_CHANNEL",
    "allowed_q",
    "amplitude",
    "as_channel",
    "average_intensity",
    "brillouin_zone_polygon",
    "centre",
    "frame_intensity",
    "layer_geometry",
    "primitive_lattice",
    "project",
    "tile",
]


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _spiral(site_of, h, k, e1, e2):
    """Circular spiral with q = (h/n1) B1 + (k/n2) B2, laid out on the cell grid."""
    n1, n2 = site_of.shape
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    phase = 2.0 * np.pi * (h * i / n1 + k * j / n2)
    spins = np.empty((n1 * n2, 3), dtype=float)
    spins[site_of.ravel()] = (np.cos(phase).ravel()[:, None] * np.asarray(e1, float)
                              + np.sin(phase).ravel()[:, None] * np.asarray(e2, float))
    return spins


def _plane(normal):
    """(e1, e2) spanning the plane normal to `normal`, with e1 x e2 = normal."""
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    seed = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def _selftest() -> int:
    from channels import channel_value
    from fields import spatial_fourier_transform
    from lattice_grid import site_of_grid
    from polarization import _triangular_lattice
    from trajectory import FieldTrajectory

    rng = np.random.default_rng(0)
    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failures += 0 if ok else 1

    n = 24
    positions, lattice = _triangular_lattice(n, n)
    site_of, n1, n2 = site_of_grid(positions, lattice, None)
    q_grid = allowed_q(lattice, n1, n2)
    reciprocal = reciprocal_lattice_from_real(primitive_lattice(lattice, n1, n2))
    n_sites = n1 * n2

    # -- the anchor: agree with the explicit sum corr already trusts ----------
    spins = rng.normal(size=(n_sites, 3))
    s_q = amplitude(spins, site_of)
    traj = FieldTrajectory(timesteps=np.zeros(1), spins=spins[None], positions=positions[None],
                           lattice=lattice, reciprocal=reciprocal)
    worst = 0.0
    for m1, m2 in [(0, 0), (1, 0), (0, 1), (5, 3), (n1 - 2, n2 - 1), (n1 // 2, n2 // 2)]:
        reference = spatial_fourier_transform(traj, q_grid[m1, m2])[0]
        worst = max(worst, float(np.max(np.abs(reference - s_q[m1, m2]))))
    check("FFT agrees with the explicit sum", worst < 1e-10,
          f"max deviation {worst:.2e} against fields.spatial_fourier_transform")

    # -- sum rule -------------------------------------------------------------
    total = project(s_q, q_grid, "total")
    check("Parseval: sum_q S(q) = sum_j |S_j|^2",
          abs(float(total.sum()) - float((spins ** 2).sum())) < 1e-9 * float(total.sum()),
          f"{float(total.sum()):.6f} against {float((spins ** 2).sum()):.6f}")

    # -- textures whose transform is known ------------------------------------
    ferro = np.tile(rng.normal(size=3), (n_sites, 1))
    intensity = project(amplitude(ferro, site_of), q_grid, "total")
    off_gamma = float(np.max(np.abs(np.delete(intensity.ravel(), 0))))
    check("a ferromagnet is a single Gamma peak", off_gamma < 1e-20 * intensity[0, 0],
          f"S(0) = {intensity[0, 0]:.3f}, largest elsewhere {off_gamma:.2e}")

    h, k = 3, 0
    e1, e2 = _plane([0.0, 0.0, 1.0])
    intensity = project(amplitude(_spiral(site_of, h, k, e1, e2), site_of), q_grid, "total")
    peaks = np.argsort(-intensity.ravel())[:2]
    expected = {(h, 0), ((-h) % n1, 0)}
    got = {tuple(int(v) for v in np.unravel_index(p, intensity.shape)) for p in peaks}
    rest = float(np.max(np.abs(np.delete(intensity.ravel(), peaks))))
    check("a single-Q spiral is one +-Q pair",
          got == expected and abs(intensity.ravel()[peaks[0]] - n_sites / 2) < 1e-9
          and rest < 1e-20 * n_sites,
          f"peaks at {sorted(got)} carrying N/2 = {n_sites / 2:.0f} each, "
          f"largest elsewhere {rest:.2e}")

    # -- L/T reads off the angle between the spiral normal and Q --------------
    q_peak = q_grid[h, 0]
    q_hat = q_peak / np.linalg.norm(q_peak)
    worst = 0.0
    rows = []
    for degrees in (0.0, 30.0, 45.0, 60.0, 90.0):
        axis = np.cross(q_hat, [0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        angle = np.radians(degrees)
        normal = np.cos(angle) * q_hat + np.sin(angle) * np.cross(axis, q_hat)
        s_spiral = amplitude(_spiral(site_of, h, k, *_plane(normal)), site_of)
        ratio = (project(s_spiral, q_grid, "L")[h, 0]
                 / project(s_spiral, q_grid, "total")[h, 0])
        worst = max(worst, abs(ratio - 0.5 * np.sin(angle) ** 2))
        rows.append(f"{degrees:g}deg->{ratio:.4f}")
    check("L/total = sin^2(theta)/2 at the magnetic peak", worst < 1e-12,
          f"{', '.join(rows)}; 0 is a proper screw, 1/2 a cycloid")

    # -- the projector agrees with channels.py where both apply ---------------
    s_q = amplitude(spins, site_of)
    worst = 0.0
    for token in ("1+5+9", "xx", "zz", "xy", "x", "1+5"):
        mine = project(s_q, q_grid, token)
        flat = s_q.reshape(-1, 3)
        reference = channel_value(flat[:, :, None] * flat.conj()[:, None, :],
                                  as_channel(token), None).reshape(mine.shape)
        worst = max(worst, float(np.max(np.abs(mine - reference))))
    check("constant channels match channel_value", worst < 1e-12,
          f"max deviation {worst:.2e} over 1+5+9, xx, zz, xy, x, 1+5")

    check("L + T = total",
          np.allclose(project(s_q, q_grid, "L") + project(s_q, q_grid, "T"),
                      project(s_q, q_grid, "total"), atol=1e-12),
          "the two projectors are complementary at every q, Gamma included")

    # -- periodicity, which is what makes tiling exact ------------------------
    q_tiled, values_tiled = tile(q_grid, project(s_q, q_grid, "total"), repeats=1)
    check("tiling repeats the map exactly",
          np.allclose(values_tiled[:n1, :n2], values_tiled[n1:2 * n1, n2:2 * n2], atol=0.0)
          and values_tiled.shape == (3 * n1, 3 * n2),
          f"S(q + G) = S(q) for primitive G; tiled shape {values_tiled.shape}")
    shift = q_tiled[n1, n2] - q_tiled[0, 0]
    check("tiling shifts q by a reciprocal vector",
          np.allclose(shift, reciprocal[0] + reciprocal[1], atol=1e-10),
          f"offset {np.array2string(shift, precision=4)} = b1 + b2")

    # -- a rigid shift of the texture must not move any intensity -------------
    rolled = np.empty_like(spins)
    rolled[np.roll(site_of, (-3, 2), axis=(0, 1)).ravel()] = spins[site_of.ravel()]
    check("intensity is translation invariant",
          np.allclose(project(amplitude(rolled, site_of), q_grid, "total"),
                      project(s_q, q_grid, "total"), atol=1e-9),
          "shifting the texture changes only the phase of s(q)")

    # -- the zone ------------------------------------------------------------
    polygon = brillouin_zone_polygon(reciprocal)
    area = 0.5 * abs(float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                                  - np.roll(polygon[:, 0], -1) * polygon[:, 1])))
    cell_area = abs(float(np.cross(reciprocal[0][:2], reciprocal[1][:2])))
    check("Brillouin zone has the right area",
          polygon.shape == (6, 2) and abs(area - cell_area) < 1e-9 * cell_area,
          f"hexagon area {area:.6f} equals the reciprocal cell area {cell_area:.6f}")

    # -- frame averaging ------------------------------------------------------
    frames = []
    for _ in range(4):
        field = rng.normal(size=(n_sites, 3))
        frames.append({"x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
                       "u": field[:, 0], "v": field[:, 1], "w": field[:, 2],
                       "box_header": "ITEM: BOX BOUNDS xy xz yz pp pp pp",
                       "box_lines": [f"{min(0.0, lattice[1, 0])} "
                                     f"{lattice[0, 0] + max(0.0, lattice[1, 0])} {lattice[1, 0]}",
                                     f"0.0 {lattice[1, 1]} 0.0",
                                     f"0.0 {lattice[2, 2]} 0.0"]})
    averaged = average_intensity(frames, {"single_layer": True})[0][1]
    by_hand = np.mean([frame_intensity(f, {"single_layer": True})[0][1] for f in frames], axis=0)
    check("frame averaging is the mean of the maps",
          np.allclose(averaged, by_hand, atol=1e-12),
          f"over {len(frames)} frames")

    # -- the supercell-shape guard -------------------------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        positions_r, lattice_r = _triangular_lattice(30, 18)
        allowed_q(lattice_r, 30, 18)
    check("a non-square supercell warns", any("three-fold" in str(w.message) for w in caught),
          "n1 != n2 breaks C3 closure of the allowed-q set")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        allowed_q(lattice, n1, n2)
    check("a square supercell does not warn", not caught, f"{n1} x {n2} is C3-closed")

    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(_selftest())
