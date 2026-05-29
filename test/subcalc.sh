RUN_SPIN=0        # 1=run, 0=pass
RUN_BZ_MAP=0      # 1=run, 0=pass
RUN_DOS=1      # 1=run, 0=pass
RUN_PLOT=0        # 1=run, 0=pass

starts=0
stops=10000

sqw_path="/home/key123456/test/spinw-phonon"
sqw=$sqw_path/"sqw_spin.py"
lmp_strj_dir=$sqw_path/examples
# components=(x y z)
components=(x)

ranks=1
dt=2
png="png"

date "+Began at: %Y-%m-%d %H:%M:%S"
# for strj in 0;do
for strj in s4;do
    start=${starts}
    stop=${stops}
    lammpstrj=${lmp_strj_dir}/${strj}.lammpstrj

                  # --use-instantaneous-pos
                  # --cbar-min 0 --cbar-max 2 \
                  # --save-npz \

        for k in "${components[@]}";do
            if [ $RUN_SPIN -eq 1 ]; then
                echo ">>> Running sqw_spin.py for frames $start to $stop ..."
                mpirun -np $ranks python $sqw  $lammpstrj qpath.txt \
                  --field-columns c_outsp[1] c_outsp[2] c_outsp[3] \
                  --points-per-segment  101 \
                  --supercell 20 20 1 \
                  --bz-folded 1 1 1 \
                  --dt-fs "$dt" \
                  --translation-repeats 1 1 1 \
                  --dtype float32 \
                  --frame-start "$start" \
                  --frame-stop "$stop" \
                  --frame-step  1 \
                  --components ${k} \
                  --window none \
                  --plot \
                  --plot-file ${png}/"${strj}"-"${k}"-usepos.png \
                  --max-freq-thz 40 \
                  --cbar-min 0 --cbar-max 300 \
                  --save-npz \
                  --output "${strj}"-pos.npz \
                  --use-instantaneous-pos
            fi

            if [ $RUN_SPIN -eq 1 ]; then
                echo ">>> Running sqw_spin.py for frames $start to $stop ..."
                mpirun -np $ranks python $sqw  $lammpstrj qpath.txt \
                  --field-columns c_outsp[1] c_outsp[2] c_outsp[3] \
                  --points-per-segment  101 \
                  --supercell 20 20 1 \
                  --bz-folded 1 1 1 \
                  --dt-fs "$dt" \
                  --translation-repeats 1 1 1 \
                  --dtype float32 \
                  --frame-start "$start" \
                  --frame-stop "$stop" \
                  --frame-step  1 \
                  --components ${k} \
                  --window none \
                  --plot \
                  --plot-file ${png}/"${strj}"-"${k}".png \
                  --max-freq-thz 40 \
                  --cbar-min 0 --cbar-max 300 \
                  --save-npz  \
                  --output "${strj}".npz
            fi

            if [ $RUN_PLOT -eq 1 ]; then
                echo ">>> Running extract_spin_animation.py for frames $start to $stop ..."
                mpirun -np $ranks python ${sqw_path}/extract_spin_animation.py \
                  $lammpstrj \
                  ${png}/spin_texture-"${strj}"-"${k}".gif \
                  --frame-start 9000 \
                  --frame-stop  9500 \
                  --frame-step  50   \
                  --single-layer \
                  --color-component ${k}
            fi    

            if [ $RUN_BZ_MAP -eq 1 ]; then
                echo ">>> Running sqw_integrated.py for frames $start to $stop ..."
                mpirun -np $ranks python ${sqw_path}/analysis/sqw_integrated.py  $lammpstrj \
                  --field-columns c_outsp[1] c_outsp[2] c_outsp[3] \
                  --supercell 20 20 1 \
                  --dt-fs "$dt" \
                  --translation-repeats 10 10 1 \
                  --dtype float32 \
                  --frame-start "$start" \
                  --frame-stop "$stop" \
                  --nh 121 --nk 121  \
                  --freq-min-thz 0   \
                  --freq-max-thz 40  \
                  --frame-step  1 \
                  --components ${k} \
                  --window none \
                  --plot \
                  --plot-file ${png}/"${strj}"-"${k}".png \
                  --cbar-min 0 --cbar-max 30000
            fi

            if [ $RUN_DOS -eq 1 ]; then
                echo ">>> Running sqw_integrated.py for frames $start to $stop ..."
                mpirun -np $ranks python ${sqw_path}/analysis/sqw_dos.py  $lammpstrj \
                  --field-columns c_outsp[1] c_outsp[2] c_outsp[3] \
                  --supercell 20 20 1 \
                  --dt-fs "$dt" \
                  --translation-repeats 10 10 1 \
                  --dtype float32 \
                  --frame-start "$start" \
                  --frame-stop "$stop" \
                  --nh 121 --nk 121  \
                  --freq-min-thz 0   \
                  --freq-max-thz 40  \
                  --frame-step  1 \
                  --components ${k} \
                  --window none \
                  --plot \
                  --plot-file ${png}/"${strj}"-"${k}".png
            fi
        done

#                  --mev
done
date "+finish at: %Y-%m-%d %H:%M:%S"
