#!/bin/bash
# Density of states: no q-path, no supercell, no MPI.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DOS="${CORR}/sqw_dos.py"

banner "1. The trace, normalised to its peak"
# The DOS integrates q away, so --supercell and a q-file are not needed at all.
python3 "${DOS}" "${SPIN_DUMP}" \
    --field-columns "${FIELD_COLUMNS[@]}" \
    --dt-fs 2 "${FRAMES[@]}" \
    --save-npz --output "${OUT}/dos.npz" \
    --no-progress

banner "2. Per-component, smoothed, with a plot"
# One curve per token, all on one figure. --smooth-sigma-thz is the visible,
# controllable way to trade resolution for a smoother curve; there is no hidden
# lag window doing it behind your back.
python3 "${DOS}" "${SPIN_DUMP}" \
    --field-columns "${FIELD_COLUMNS[@]}" \
    --dt-fs 2 "${FRAMES[@]}" \
    --component 1+5+9 1 5 9 \
    --smooth-sigma-thz 0.1 --normalize max \
    --freq-max-thz 15 \
    --save-npz --output "${OUT}/dos_components.npz" \
    --plot --plot-file "${OUT}/dos.png" --plot-raw \
    --no-progress

banner "L and T are rejected here, on purpose"
# Their weights are built from the direction of q, and the DOS has no q.
python3 "${DOS}" "${SPIN_DUMP}" \
    --field-columns "${FIELD_COLUMNS[@]}" \
    --dt-fs 2 --frame-stop 64 --component T --no-progress 2>&1 | tail -1 || true

banner "Files written"
ls -1 "${OUT}"/dos*.npz "${OUT}"/dos.png
