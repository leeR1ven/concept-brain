#!/bin/bash
# sequential: one process at a time
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
SIZE="hip_cap=30000 hip_ndg=6000 hip_kdg=120 mix_cap=2000000 ctx_n=4000 ctx_k=120"
MAT="mature=50 gen_mmin=0.05"
nice python3 sizetune.py tag=base cur=seq save=1 > out/st_base.stdout 2>&1
nice python3 diag_oneshot.py out/st_base_s0_agent.pkl > out/diag_oneshot_base.txt 2>&1
nice python3 sizetune.py tag=c_inter cur=inter > out/st_c.stdout 2>&1
nice python3 sizetune.py tag=b_mature cur=seq $MAT > out/st_b.stdout 2>&1
nice python3 sizetune.py tag=a_size cur=seq $SIZE > out/st_a.stdout 2>&1
nice python3 sizetune.py tag=d_mature_inter cur=inter $MAT > out/st_d.stdout 2>&1
nice python3 sizetune.py tag=e_size_mature_inter cur=inter $SIZE $MAT > out/st_e.stdout 2>&1
echo BATCH_DONE > out/batch_st.done
