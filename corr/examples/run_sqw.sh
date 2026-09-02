#!/bin/bash
# S(q,w) along a q-path: the channel model, and what each option buys.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SQW="${CORR}/sqw_spin_corr.py"

banner "1. The default: the trace, S^xx + S^yy + S^zz"
# --component may be omitted entirely; 1+5+9 is the default.
python3 "${SQW}" "${SPIN_DUMP}" "${QPATH}" \
    --field-columns "${FIELD_COLUMNS[@]}" \
    "${LATTICE[@]}" "${FRAMES[@]}" "${QDENSITY[@]}" \
    --save-npz --output "${OUT}/trace.npz" \
    --no-progress

banner "2. Several channels in one pass"
# Each token becomes its own output file, suffixed with the token. '+' inside a
# token sums those components into one curve; a space starts a new curve.
#   T       transverse to q      L    longitudinal to q
#   1..9    row-major xx..zz     1+5+9 = the trace
python3 "${SQW}" "${SPIN_DUMP}" "${QPATH}" \
    --field-columns "${FIELD_COLUMNS[@]}" \
    "${LATTICE[@]}" "${FRAMES[@]}" "${QDENSITY[@]}" \
    --component T L 1 1+5+9 \
    --save-npz --output "${OUT}/multi.npz" \
    --no-progress

banner "3. A publication-style run, with a plot"
# --window hann suppresses spectral leakage; leave it on unless you are reading
# C(q,tau) back out, for which see run_corrplus.sh.
python3 "${SQW}" "${SPIN_DUMP}" "${QPATH}" \
    --field-columns "${FIELD_COLUMNS[@]}" \
    "${LATTICE[@]}" "${FRAMES[@]}" \
    --points-per-segment 21 \
    --component T \
    --window hann \
    --save-npz --output "${OUT}/transverse.npz" \
    --plot --plot-file "${OUT}/transverse.png" \
    --max-freq-thz 15 --cbar-min 0 \
    --no-progress

banner "Files written"
ls -1 "${OUT}"/trace.npz "${OUT}"/multi_*.npz "${OUT}"/transverse.*
