#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from sqw_path import main


if __name__ == "__main__":
    main(
        default_method="corr",
        prog="sqw_spin_corr.py",
        description=(
            "Compute a q-resolved dynamic structure factor S(q,w) from "
            "time-correlation functions of a generic three-component field."
        ),
    )
