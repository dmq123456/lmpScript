str0() {
python generate_oscillating_dump.py   \
    --supercell 20 20 1 \
    --spin-freqs 0.0  --spin-wavevector 0.0 0.2 0.0  \
    --pos-dir y --pos-freq 4.0 --pos-amp 1.0  --pos-wavevector 0.0 0.0 0.0 \
    --output  0.lammpstrj
}

str1() {
python generate_oscillating_dump.py   \
    --supercell 20 20 1 \
    --spin-freqs 5.0  --spin-wavevector 0.0 0.2 0.0  \
    --pos-dir y --pos-freq 4.0 --pos-amp 1.0  --pos-wavevector 0.0 0.0 0.0 \
    --output  s5.lammpstrj
}

str2() {
python generate_oscillating_dump.py   \
    --supercell 20 20 1 \
    --spin-freqs 4.0  --spin-wavevector 0.0 0.2 0.0  \
    --pos-dir y --pos-freq 4.0 --pos-amp 1.0  --pos-wavevector 0.0 0.0 0.0 \
    --output  s4.lammpstrj
}

str2
