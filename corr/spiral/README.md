# spiral/ — spin-spiral tools

Scripts for measuring, rebuilding and evaluating spin spirals in LAMMPS spin
dumps. All of them print their options with `-h`.

| Script | What it does | Environment |
|---|---|---|
| `spin_spiral.py` | Fit one frame: dominant q (2D FFT on the magnetic lattice), plane normal n, cone, fitted s0; profile plot and optional 3-D sketch of the plane (`--plane`) | numpy, matplotlib |
| `spin_profile.py` | Earlier, simpler profile of a spiral along its modulation direction | numpy, matplotlib |
| `spiral_average.py` | Fit every frame of a dump and average: q checked (never averaged), n averaged as a vector, cone as a scalar, phase reported separately; checks the ideal spiral against each frame and prints a `make_spiral.py` command | numpy (imports `spin_spiral.py`) |
| `make_spiral.py` | Build an ideal (optionally conical) spiral from q, rotation axis `--e` and `--s0` as a one-frame dump | numpy |
| `phonon_displace.py` | Keep the spins of one dump frame and move the atoms of the ideal `make_spiral.py` structure along a phonon mode from phonopy `band.yaml` (q-point, band(s); degenerate bands combined with `--direction` or `--mix-angle`); writes one frame per amplitude plus `<out>.amp.dat` | numpy, PyYAML (imports `make_spiral.py`, `dumpio.py`) |
| `dump_to_extxyz.py` | LAMMPS dump -> extxyz with `initial_magmoms`, the input of the SpinGNN model | numpy, ase (imports `dumpio.py`) |
| `dump_to_poscar.py` | LAMMPS dump -> POSCAR + MAGMOM for VASP | numpy (imports `dumpio.py`) |
| `dumpio.py` | Shared dump reader for the two converters (not a command) | numpy |
| `predict_forces.py` | Energy, forces, spin forces (and BEC with `--bec`) from a deployed SpinGNN model for each frame of an extxyz; `--energy-file` writes the energies | **SpinGNN environment** (torch, nequip) |

Typical chain, from an MD trajectory to the energy of its averaged spiral:

```bash
python spiral_average.py field.lammpstrj --supercell 50 50 1 --type 1
python make_spiral.py ... --out spiral_avg.lammpstrj      # command printed by the step above
python dump_to_extxyz.py spiral_avg.lammpstrj spiral_avg.extxyz --type-map Ni I
python predict_forces.py spiral_avg.extxyz force_avg.lammpstrj --model deployed.pth --energy-file energy.txt
```

Energy along a phonon mode with the spins frozen:

```bash
python phonon_displace.py spiral_avg.lammpstrj --supercell 50 50 1 --band-yaml band.yaml \
       --q 0 0 0 --bands 7 9 --direction Ni 1 0 0 --amax 0.05 --nframes 11 --out phonon.lammpstrj
python dump_to_extxyz.py phonon.lammpstrj phonon.extxyz --type-map Ni I
python predict_forces.py phonon.extxyz phonon_force.lammpstrj --model deployed.pth --energy-file energy.txt
paste <(grep -v '^#' phonon.amp.dat) <(grep -v '^#' energy.txt)      # Q, displacement, energy
```

`predict_forces.py` evaluates a whole frame at once; a large supercell
(e.g. 7500 atoms) can need more memory than a workstation has.

The scripts import their helpers (`spin_spiral.py`, `dumpio.py`) from this
folder, so keep them together. They were moved here from `~/test/spiral/`,
whose `auto.sh` files now call them through `$SPIRAL`.
