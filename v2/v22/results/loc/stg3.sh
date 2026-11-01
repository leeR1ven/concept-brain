#!/bin/bash
cd ../v22
until grep -q STG2DONE loc/stg2.out; do sleep 5; done
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
S="seed=0 proto=frozen nS=500"
python3 run_v22.py tag=lvWf $S $V21 veto_R1=1 veto_eps=0.1 veto_W=40 veto_gate=0.15 veto_fast=0.04 lesion=veto_R1:0 out=out/runs/loc_lvWf_s0.json > loc/st_vWf.log 2>&1
echo STG3DONE
