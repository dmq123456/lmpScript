#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from sqw_path import main


if __name__ == "__main__":
    main(
        default_method="periodogram",
        prog="sqw_spin.py",
        description="Compute a q-resolved phonon spectrum from a LAMMPS-like dump file.",
    )
