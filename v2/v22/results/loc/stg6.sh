#!/bin/bash
cd ../v22
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
CF="rm=1 rm_train=1 rm_prune=1 rm_age=20000 rm_regrow=1 rm_alloc=flat"
S="seed=0 proto=frozen nS=500 drift3=3000"
python3 run_v22.py tag=lcf3k $S $V21 $CF out=out/runs/loc_lcf3k_s0.json > loc/st_cf3k.log 2>&1
python3 run_v22.py tag=lbase3k $S $V21 out=out/runs/loc_lbase3k_s0.json > loc/st_base3k.log 2>&1
echo STG6DONE
