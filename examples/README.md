# Examples

## Data generator

| Script | Description |
|--------|-------------|
| `generate_oscillating_dump.py` | Self-contained: builds hexagonal lattice from scratch, generates spin precession and position oscillation with tunable frequencies, amplitudes, and wavevectors (both uniform q=0 and finite-q propagating modes supported) |

## q-path file

`qpath.txt` — high-symmetry q-path for a 2D triangular Bravais lattice.

## Quick examples

```bash
# Uniform (q=0) spin precession at 4 THz
python examples/generate_oscillating_dump.py   \
    --supercell 10 10 1 --spin-freqs 4.0

# Spin at 4 THz + position oscillation along y at 4 THz
python examples/generate_oscillating_dump.py   \
    --supercell 10 10 1 --spin-freqs 4.0         \
    --pos-dir y --pos-freq 4.0 --pos-amp 0.1

# Propagating spin wave at the M point (1/2, 1/2, 0)
python examples/generate_oscillating_dump.py   \
    --supercell 20 20 1 --spin-freqs 4.0         \
    --spin-wavevector 0.5 0.5 0.0

# Spin + position waves, both at finite q
python examples/generate_oscillating_dump.py   \
    --supercell 20 20 1 --spin-freqs 4.0         \
    --spin-wavevector 0.5 0.5 0.0                \
    --pos-dir y --pos-freq 4.0 --pos-amp 0.1     \
    --pos-wavevector 0.0 0.5 0.0

# Compute S(q,w) from the x-component
python sqw_spin.py oscillating.lammpstrj examples/qpath.txt   \
    --field-columns c_outsp[1] c_outsp[2] c_outsp[3]          \
    --supercell 10 10 1 --dt-fs 2.0                            \
    --components x --use-instantaneous-pos                      \
    --plot --max-freq-thz 40
```
