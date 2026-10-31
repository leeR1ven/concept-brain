#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
until [ -f results/after2.done ]; do sleep 20; done
H="h_gate=1 h_conf=1 h_relock=1 h_vbound=1 h_nocreate=1 h_level=1 h_denoise=1"
nice python3 -B w2_agent.py 16 2 0.004 0.0 120000 0 tag=w2life_H noise_at=30000:1.0,60000:0.5,90000:1.0 $H > results/w2/log_life_H.txt 2>&1
echo DONE3 > results/after3.done
