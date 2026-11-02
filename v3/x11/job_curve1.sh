#!/bin/bash
# X11 scaling curve 1 (s = 1, seeds 0 and 1 as two processes in this one GPU job). Resumes from ~/ckpt/x11_s1_seed<k>/.
# Before resuming, a copy of each life checkpoint is kept (keep_<dialogues>.pt) for later frozen diagnostics.
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 X10_GPU_FRAC=0.2
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
for sd in 0 1; do
  d=~/ckpt/x11_s1_seed$sd
  if [ -f $d/life.pt ]; then n=$(python3 -c "import json;print(json.load(open('$d/res.json'))['jobs'][-1].get('end_dialogues','x'))"); cp $d/life.pt $d/keep_$n.pt; fi
done
for sd in 0 1; do
  python3 run_scale.py --seed $sd --run x11_s1 --max_minutes 74 --n_heldout 200 --out out_scale > log_seed$sd.txt 2>&1 &
done
wait
for sd in 0 1; do echo "=== seed $sd"; grep -v Warning log_seed$sd.txt | cut -c1-3000 | tail -8; done
ls -la ~/ckpt/x11_s1_seed0 ~/ckpt/x11_s1_seed1
