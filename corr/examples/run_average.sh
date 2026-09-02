#!/bin/bash
# Reducing scatter: split the trajectory into windows, then average.
#
# A spectrum from one trajectory scatters by roughly 100% at every frequency
# bin, and running longer does not fix it -- the extra frames buy a finer grid,
# not a better value per bin. Averaging independent estimates is what works.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SQW="${CORR}/sqw_spin_corr.py"
AVG="${CORR}/average_sqw.py"

banner "1. One run per time window"
for start in 0 512 1024 1536; do
    stop=$((start + 512))
    python3 "${SQW}" "${SPIN_DUMP}" "${QPATH}" \
        --field-columns "${FIELD_COLUMNS[@]}" \
        "${LATTICE[@]}" "${QDENSITY[@]}" \
        --frame-start "${start}" --frame-stop "${stop}" \
        --component T \
        --save-npz --output "${OUT}/win-${start}.npz" \
        --no-progress > /dev/null
    echo "  frames ${start}-${stop}  ->  ${OUT}/win-${start}.npz"
done

banner "2. Average them"
# The inputs are checked before they are combined: the q-path, the frequency
# grid and the channel must all agree, and a mismatch says which file and which
# key disagree rather than silently producing a meaningless mean.
#
# Watch the scatter line. Independent estimates scatter by ~1.0; far less than
# that means the windows overlap in time and the averaging gains less than
# 1/sqrt(M) would suggest.
python3 "${AVG}" "${OUT}"/win-*.npz \
    --output "${OUT}/mean.npz" \
    --plot --plot-file "${OUT}/mean.png" \
    --max-freq-thz 15

banner "Files written"
ls -1 "${OUT}"/win-*.npz "${OUT}"/mean.*
