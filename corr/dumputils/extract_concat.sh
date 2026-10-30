#!/bin/bash
# Extract frames from several LAMMPS dump files and concatenate them into one file.
#
#   extract_concat.sh OUTPUT  FILE1:START[:STOP[:STEP]]  FILE2:START[:STOP[:STEP]]  ...
#
# Each FILE:START:STOP:STEP is passed to extract_frames.py (same semantics,
# negative START/STOP count from the end as in Python, -1 = last frame:
# START inclusive, STOP exclusive, empty or omitted STOP = to the last frame,
# STEP default 1); the pieces are appended to OUTPUT in the order given.
#
# Examples:
#   extract_concat.sh out.lammpstrj a.lammpstrj:2000 b.lammpstrj:0:1:1
#       -> frame 2000.. of a, followed by frame 0 of b
#   extract_concat.sh out.lammpstrj run1.lammpstrj:0::100 run2.lammpstrj:0::100
#       -> every 100th frame of run1, then every 100th frame of run2
#
# The timesteps of the frames are copied unchanged, so they need not increase
# monotonically in OUTPUT. File names must not contain ':'.
# Set PY to choose the python interpreter (default: python3).

set -euo pipefail

PY=${PY:-python3}
EXTRACT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extract_frames.py"

if [ $# -lt 2 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

out=$1
shift
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
: > "$out"

for spec in "$@"; do
    IFS=':' read -r file start stop step <<< "$spec"
    [ -f "$file" ] || { echo "[ERROR] no such file: $file" >&2; exit 1; }
    args=(--frame-start "${start:-0}")
    [ -n "${stop:-}" ] && args+=(--frame-stop "$stop")
    [ -n "${step:-}" ] && args+=(--frame-step "$step")
    echo "[INFO] $file  start=${start:-0} stop=${stop:-end} step=${step:-1}"
    "$PY" "$EXTRACT" "$file" "$tmp" "${args[@]}" | grep "Kept frames" || true
    cat "$tmp" >> "$out"
done

echo "[INFO] total frames in $out: $(grep -c '^ITEM: TIMESTEP' "$out")"
