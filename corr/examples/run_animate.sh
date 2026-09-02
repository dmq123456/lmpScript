#!/bin/bash
# Visualising a dump: one frame to try settings, then the animation.
#
# frame.py and animate.py draw the same picture. Rendering is expensive, so
# settle the colormap and the colour range on a single frame first.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FRAME="${CORR}/tools/frame.py"
ANIM="${CORR}/tools/animate.py"

banner "1. One frame, to choose settings"
# --vector draws arrows, --color picks the scalar behind them. vz/norm/inplane
# and topo are derived from the vector; anything else is read from a dump column.
python3 "${FRAME}" "${BEC_DUMP}" --frame 0 \
    --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]' \
    --color vz --drop-zero-vector \
    --single-layer --vmin -1 --vmax 1 \
    --out "${OUT}/frame_spin.png"

banner "2. Higher resolution for a figure"
# The image is figsize x dpi pixels. Raise dpi for a still; leave it alone for
# an animation, where a GIF is limited to 256 colours and every frame stays in
# memory until the file is written.
python3 "${FRAME}" "${BEC_DUMP}" --frame 0 \
    --color 'c_outbec[1]+c_outbec[5]+c_outbec[9]' --element Ni \
    --single-layer --subtract-mean --cmap RdBu_r \
    --dpi 200 --figsize 8 6 \
    --out "${OUT}/frame_bec.png"

banner "3. The animation, same options"
python3 "${ANIM}" "${BEC_DUMP}" "${OUT}/spin.gif" \
    --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]' \
    --color vz --drop-zero-vector \
    --single-layer --vmin -1 --vmax 1 --fps 4 \
    --no-progress

banner "4. A scalar with no arrows at all"
# Any column animates the same way -- an applied field written into the dump
# with 'dump ... v_Ey' needs only --color v_Ey.
python3 "${ANIM}" "${BEC_DUMP}" "${OUT}/bec.gif" \
    --color 'c_outbec[1]+c_outbec[5]+c_outbec[9]' --element Ni \
    --single-layer --subtract-mean --cmap RdBu_r --fps 4 \
    --no-progress

banner "5. Topological charge density"
# --color topo is derived from the vector like vz, but from the whole texture
# rather than one component of it: the sites are triangulated and each triangle
# contributes the signed area its three spins span on the unit sphere. Two
# options stop being optional here.
#
#   --element Ni    the triangulation runs over the magnetic sublattice. Leaving
#                   the iodine in makes the site grid unreadable, which is an
#                   error rather than a wrong picture.
#   --vmin/--vmax   symmetric, with a diverging colormap. The density changes
#                   sign, and at a skyrmion core it is sharply peaked, so an
#                   automatic range is set by that one peak and flattens
#                   everything around it.
python3 "${FRAME}" "${BEC_DUMP}" --frame 0 \
    --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]' \
    --color topo --element Ni --single-layer \
    --cmap RdBu_r --vmin -0.05 --vmax 0.05 --arrow-scale 0.6 \
    --out "${OUT}/frame_topo.png"

python3 "${ANIM}" "${BEC_DUMP}" "${OUT}/topo.gif" \
    --vector 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]' \
    --color topo --element Ni --single-layer \
    --cmap RdBu_r --vmin -0.05 --vmax 0.05 --fps 4 \
    --no-progress

banner "6. The check that comes free with it"
# Summing q_i over a closed layer gives the topological charge, which is an
# integer for *any* spin configuration -- so the sum tests the triangulation,
# the orientation, the periodic wrapping and the site selection all at once, on
# whatever data is at hand. The thermal texture above is a patchwork of both
# signs that nonetheless sums to exactly +1.
#
# The second line is the opposite end: s4 is a single-q spin wave, modulated
# along one direction only, so grad_x m is parallel to grad_y m and the density
# vanishes. Not approximately -- to the last bit.
PYTHONPATH="${CORR}/tools" python3 - "${BEC_DUMP}" "${SPIN_DUMP}" <<'CHECK'
import sys

import numpy as np
from dumpframe import load_single_frame
from topocharge import site_density

spin = ("c_outsp[1]", "c_outsp[2]", "c_outsp[3]")
one_layer = {"single_layer": True}

q = site_density(load_single_frame(sys.argv[1], 0, vector=spin, element="Ni"), one_layer)
print(f"  thermal texture   : Q = sum q_i = {q.sum():+.12f}   (an exact integer)")

q = site_density(load_single_frame(sys.argv[2], 0, vector=spin), one_layer)
print(f"  single-q spin wave: max |q_i|   = {np.abs(q).max():.3e}   (identically zero)")
CHECK

banner "Files written"
ls -1 "${OUT}"/frame_*.png "${OUT}"/*.gif
