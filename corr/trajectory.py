#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The field trajectory: pure data, no algorithms.

Holds what was read from the dump plus the lattice information derived from it.
Every numerical routine in this package takes a FieldTrajectory as input rather
than reaching for a method on it, so the numerics can be exercised on synthetic
data without touching the file layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geometry import (
    cart_to_frac_q,
    frac_to_cart_q,
    primitive_lattice_from_supercell,
    reciprocal_lattice_from_real,
)
from io_dump import load_spin_dump


@dataclass
class FieldTrajectory:
    """A three-component field on N atoms over Nt frames.

    spins        (nt, natoms, 3)  the field itself
    positions    (1 or nt, natoms, 3)  frozen first frame, or the full history
    lattice      (3, 3)  supercell real-space vectors as rows
    reciprocal   (3, 3)  primitive-cell reciprocal vectors as rows, or None
                         when no supercell was given (a q-free calculation
                         such as the DOS never needs it)
    atom_types   (natoms,) or None
    weights      (natoms,) or None   sqrt(mass) when mass weighting is on
    """

    timesteps: np.ndarray
    spins: np.ndarray
    positions: np.ndarray
    lattice: np.ndarray
    reciprocal: np.ndarray | None = None
    atom_types: np.ndarray | None = None
    weights: np.ndarray | None = None
    field_columns: tuple[str, str, str] = ("sx", "sy", "sz")

    @property
    def n_frames(self) -> int:
        return int(self.spins.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.spins.shape[1])

    def _require_reciprocal(self) -> np.ndarray:
        if self.reciprocal is None:
            raise ValueError(
                "This trajectory was loaded without a supercell, so no reciprocal "
                "lattice was built. Pass supercell= to load_trajectory if you need q."
            )
        return self.reciprocal

    def q_frac_to_cart(self, q_frac: np.ndarray) -> np.ndarray:
        return frac_to_cart_q(q_frac, self._require_reciprocal())

    def q_cart_to_frac(self, q_cart: np.ndarray) -> np.ndarray:
        return cart_to_frac_q(q_cart, self._require_reciprocal())

    def set_mass_weights(self, masses_per_type) -> None:
        """Weight each atom by sqrt(mass) before the spatial Fourier sum
        (the phonon spectral-energy-density convention). Leaving weights as
        None reproduces the unweighted spectrum used for magnons."""
        if self.atom_types is None:
            raise ValueError(
                "Mass weighting requested but the dump carried no 'type' column. "
                "Delete the .sqwcache.npz beside the dump and re-read it, or pass "
                "--no-cache-binary."
            )
        masses = np.asarray(masses_per_type, dtype=np.float64)
        if masses.ndim != 1 or masses.size < 1:
            raise ValueError("masses_per_type must be a 1D sequence with at least one entry")
        if np.any(masses <= 0.0):
            raise ValueError("All per-type masses must be positive")
        types = self.atom_types
        tmin, tmax = int(types.min()), int(types.max())
        if tmin < 1:
            raise ValueError(f"Atom types must be 1-based; found minimum type {tmin}")
        if tmax > masses.size:
            raise ValueError(
                f"Dump contains atom type {tmax} but only {masses.size} mass value(s) "
                f"were given; supply one mass per type for types 1..{tmax}."
            )
        self.weights = np.sqrt(masses[types - 1]).astype(np.float64, copy=False)

    def print_lattice_info(self, prefix: str = "[INFO]") -> None:
        print(f"{prefix} Supercell real-space lattice vectors (rows):")
        for i, v in enumerate(self.lattice, start=1):
            print(f"{prefix}   a{i} = ({v[0]: .10f}, {v[1]: .10f}, {v[2]: .10f})")
        if self.reciprocal is None:
            return
        print(f"{prefix} Primitive-cell reciprocal lattice vectors (rows):")
        for i, v in enumerate(self.reciprocal, start=1):
            print(f"{prefix}   b{i} = ({v[0]: .10f}, {v[1]: .10f}, {v[2]: .10f})")


def load_trajectory(
    filename: str | Path,
    supercell: tuple[int, int, int] | np.ndarray | None = None,
    *,
    use_instantaneous_pos: bool = False,
    dtype=np.float32,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    frame_step: int | None = None,
    cache_binary: bool = True,
    cache_file: str | Path | None = None,
    spin_threshold: float = 0.0,
    field_columns=None,
    progress: bool = False,
    progress_reports: int = 20,
) -> FieldTrajectory:
    """Read a LAMMPS-like dump into a FieldTrajectory.

    The primitive cell is inferred by dividing the supercell lattice by the
    --supercell repeats; the reciprocal lattice is built from that primitive
    cell, so q-points in fractional coordinates always refer to the primitive
    Brillouin zone. Pass supercell=None for a calculation that never touches q
    -- the DOS, for instance -- and no reciprocal lattice is built.
    """
    timesteps, positions, spins, lattice, atom_types = load_spin_dump(
        filename,
        keep_all_positions=use_instantaneous_pos,
        dtype=dtype,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_step=frame_step,
        cache_binary=cache_binary,
        cache_file=cache_file,
        spin_threshold=spin_threshold,
        field_columns=field_columns,
        progress=progress,
        progress_reports=progress_reports,
    )
    if supercell is None:
        reciprocal = None
    else:
        primitive = primitive_lattice_from_supercell(lattice, np.asarray(supercell))
        reciprocal = reciprocal_lattice_from_real(primitive)
    from io_dump import normalize_field_columns

    return FieldTrajectory(
        timesteps=timesteps,
        spins=spins,
        positions=positions,
        lattice=lattice,
        reciprocal=reciprocal,
        atom_types=atom_types,
        field_columns=normalize_field_columns(field_columns),
    )


__all__ = ["FieldTrajectory", "load_trajectory"]
