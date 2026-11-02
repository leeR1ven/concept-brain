#!/bin/bash
cd ../v22
until grep -q STG4DONE loc/stg4.out; do sleep 5; done
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
S="seed=0 proto=frozen nS=500"
python3 run_v22.py tag=lgr $S $V21 gain_reply=1 out=out/runs/loc_lgr_s0.json > loc/st_gr.log 2>&1
python3 run_v22.py tag=lnodip $S ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 act_T=60 soc_feats=1 rec_fix=1 out=out/runs/loc_lnodip_s0.json > loc/st_nodip.log 2>&1
echo STG5DONE
