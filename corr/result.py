#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from geometry import q_path_distance


THZ_TO_MEV = 4.135667696  # meV / THz


@dataclass
class SQWResult:
    timesteps: np.ndarray
    q_frac: np.ndarray
    q_vectors: np.ndarray
    freq_thz: np.ndarray
    sqw: np.ndarray
    lattice: np.ndarray
    reciprocal_lattice: np.ndarray
    dt_fs: float
    components: str
    q_frac_input: np.ndarray | None = None
    q_input_reciprocal_lattice: np.ndarray | None = None
    bz_folded: np.ndarray | None = None
    q_fold_multiplicity: int | None = None
    field_columns: np.ndarray | None = None
    projection: str | None = None
    freq_mode: str | None = None
    translation_repeats: np.ndarray | None = None
    q_node_indices: np.ndarray | None = None
    q_node_labels: List[str] | None = None
    corr_plus: np.ndarray | None = None
    corr_norm: str | None = None

    @property
    def energy_meV(self) -> np.ndarray:
        return self.freq_thz * THZ_TO_MEV

    def to_npz_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "timesteps": self.timesteps,
            "q_frac": self.q_frac,
            "q_vectors": self.q_vectors,
            "freq_thz": self.freq_thz,
            "energy_meV": self.energy_meV,
            "sqw": self.sqw,
            "components": self.components,
            "field_columns": self.field_columns,
            "projection": self.projection,
            "dt_fs": self.dt_fs,
            "freq_mode": self.freq_mode,
            "lattice": self.lattice,
            "reciprocal_lattice": self.reciprocal_lattice,
            "translation_repeats": self.translation_repeats,
            "q_node_indices": self.q_node_indices,
            "q_path_distance": q_path_distance(self.q_vectors),
        }
        if self.q_frac_input is not None:
            data["q_frac_input"] = self.q_frac_input
        if self.q_input_reciprocal_lattice is not None:
            data["q_input_reciprocal_lattice"] = self.q_input_reciprocal_lattice
        if self.bz_folded is not None:
            data["bz_folded"] = self.bz_folded
        if self.q_fold_multiplicity is not None:
            data["q_fold_multiplicity"] = np.asarray(self.q_fold_multiplicity, dtype=np.int64)
        if self.corr_plus is not None:
            data["corr_plus"] = self.corr_plus
        if self.corr_norm is not None:
            data["corr_norm"] = self.corr_norm
        return data


__all__ = ["SQWResult", "THZ_TO_MEV"]
