#!/bin/bash
# Shared settings for the example scripts.
#
# Every example sources this file, so pointing the examples at your own data is
# a matter of exporting DATA_DIR (or editing the defaults here) rather than
# editing each script.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORR="$(cd "${HERE}/.." && pwd)"

# Repository root, where the sample trajectories live. Override with:
#   DATA_DIR=/path/to/data ./run_sqw.sh
DATA_DIR="${DATA_DIR:-$(cd "${CORR}/.." && pwd)}"

# A coherent spin wave at 4.0 THz on a 20x20 triangular lattice, 10000 frames.
SPIN_DUMP="${SPIN_DUMP:-${DATA_DIR}/examples/s4.lammpstrj}"
QPATH="${QPATH:-${DATA_DIR}/examples/qpath.txt}"

# A short dump that also carries the Born-effective-charge columns.
BEC_DUMP="${BEC_DUMP:-${DATA_DIR}/test/test.lammpstrj}"

OUT="${OUT:-${HERE}/out}"
mkdir -p "${OUT}"

# The dump column names holding the three field components. The square brackets
# are shell globs, so they must stay quoted on every command line.
FIELD_COLUMNS=('c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]')

# Kept small so the examples finish in seconds. Real runs use every frame and a
# far denser q-path.
FRAMES=(--frame-start 0 --frame-stop 512)
QDENSITY=(--points-per-segment 5)
LATTICE=(--supercell 20 20 1 --dt-fs 2)

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
