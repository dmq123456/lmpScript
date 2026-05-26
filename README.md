# sqw — Spin Structure Factor S(q,ω) from LAMMPS Dump Files

Compute q-resolved dynamic structure factors for generic three-component fields
(spins, velocities, dipoles) stored in LAMMPS-like dump trajectories.

Two spectrum estimators are provided:

- **periodogram** — direct FFT of the space-time field s(q,t)
- **correlation** — FFT of the time auto-correlation C(q,τ)

## Quick start

```bash
# Spin-wave S(q,w) along a high-symmetry q-path
python sqw_spin.py dump.lammpstrj qpath.txt          \
    --field-columns c_outsp[1] c_outsp[2] c_outsp[3] \
    --supercell 20 20 1 --dt-fs 2.0                   \
    --components x --use-instantaneous-pos             \
    --plot --max-freq-thz 40
```

## Directory layout

```
.
├── sqw_calculator.py      # core: SpinStructureFactorCalculator
├── sqw_core.py            # public API re-exports
├── sqw_geometry.py        # lattice, reciprocal space, q-path builders
├── sqw_io.py              # LAMMPS dump parser & binary cache
├── sqw_mpi.py             # MPI q-point parallelism
├── sqw_args.py            # CLI argument definitions
├── sqw_path.py            # q-path workflow orchestration
├── sqw_result.py          # result container & serialization
├── sqw_spin.py            # CLI entry: periodogram estimator
├── sqw_spin_corr.py       # CLI entry: correlation estimator
├── tools/                 # post-processing & visualisation
├── analysis/              # BZ maps, magnon DOS
├── examples/              # test-data generators & example q-paths
├── .gitignore
└── README.md
```

## Requirements

Python ≥ 3.9, `numpy`, `scipy`, `matplotlib`, `mpi4py` (optional, for MPI).

## License

MIT
