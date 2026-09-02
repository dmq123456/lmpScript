#!/bin/bash
# The time-domain view: C(q,tau), and the two settings that corrupt it.
#
# The correlation function is where a mode's lifetime lives -- its oscillation
# period is the frequency, the decay of its envelope is the linewidth. Fitting
# that envelope is often steadier than fitting a Lorentzian in frequency, since
# the frequency estimate scatters by ~100% per bin while C(tau) at small tau is
# determined by many sample pairs.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SQW="${CORR}/sqw_spin_corr.py"

banner "Correct settings for reading a lifetime"
# Both defaults have to be turned off:
#   --corr-norm unbiased   divides by the Nt-tau terms that actually
#                          contributed, cancelling a spurious triangular decay
#   --window none          keeps the data window out of the envelope, which
#                          would otherwise imprint its own autocorrelation
python3 "${SQW}" "${SPIN_DUMP}" "${QPATH}" \
    --field-columns "${FIELD_COLUMNS[@]}" \
    "${LATTICE[@]}" "${FRAMES[@]}" "${QDENSITY[@]}" \
    --component 1+5+9 \
    --window none --corr-norm unbiased \
    --save-corr-plus \
    --save-npz --output "${OUT}/corrplus.npz" \
    --no-progress

banner "What the two settings do to the envelope"
# s4 is an undamped coherent oscillation, so its true envelope is flat. Any
# decay seen below is an artefact of the analysis, not physics.
python3 - "${OUT}/corrplus.npz" <<'PY'
import sys
import numpy as np
data = np.load(sys.argv[1])
corr = data["corr_plus"]
trace = (corr[:, :, 0, 0] + corr[:, :, 1, 1] + corr[:, :, 2, 2]).real
iq = int(np.argmax(data["sqw"].max(axis=1)))
envelope = np.abs(trace[iq]) / abs(trace[iq][0])
dt_ps = float(data["dt_fs"]) * 1e-3
for lag_ps in (0.5, 1.0, 2.0):
    idx = int(lag_ps / dt_ps)
    if idx < envelope.size:
        print(f"  |C| at {lag_ps:.1f} ps : {envelope[idx]:.4f}")
print(f"  (q index {iq}, q_frac = {np.round(data['q_frac'][iq], 4)})")
print("  A flat envelope means no spurious decay was introduced.")
PY

banner "Note"
echo "  For S(q,w) itself the opposite settings are right: --window hann and"
echo "  --corr-norm biased. The frequency-domain and time-domain uses of this"
echo "  route want opposite defaults, so run it twice rather than compromising."
