# Examples

## Data generators

| Script | Description |
|--------|-------------|
| `generate_oscillating_dump.py` | Self-contained: define hexagonal lattice, spin & position oscillation frequencies, outputs a dump file |
| `generate_spin_wave_dump.py` | Adds a propagating spin wave to an existing ideal-lattice dump |
| `generate_ideal_lattice_dump.py` | Snaps atom positions to an ideal hexagonal lattice from a reference dump |

## q-path file

`qpath.txt` — high-symmetry q-path for a 2D triangular Bravais lattice.

## Quick example

```bash
# 1. Generate a test dump with spin precession at 4 THz + position oscillation along y
python examples/generate_oscillating_dump.py   \
    --supercell 10 10 1 --spin-freqs 4.0         \
    --pos-dir y --pos-freq 4.0 --pos-amp 0.1     \
    --nframes 10000 --dt-fs 2.0                   \
    --output test.lammpstrj

# 2. Compute S(q,w) from the x-component
python sqw_spin.py test.lammpstrj examples/qpath.txt   \
    --field-columns c_outsp[1] c_outsp[2] c_outsp[3]   \
    --supercell 10 10 1 --dt-fs 2.0                     \
    --components x --use-instantaneous-pos               \
    --plot --max-freq-thz 40
```
