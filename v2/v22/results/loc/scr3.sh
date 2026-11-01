#!/bin/bash
cd ../v22
until grep -q LOC2DONE loc/scr2.out; do sleep 5; done
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
LF="seed=6 ticks=600000 chunk=100000"
python3 life_v22.py tag=loc6_vW $LF $V21 veto_R1=1 veto_eps=0.1 veto_W=40 > loc/vW_6.log 2>&1
python3 life_v22.py tag=loc6_vWu $LF $V21 veto_R1=1 veto_eps=0.1 veto_W=40 u_R1=0.0 > loc/vWu_6.log 2>&1
echo LOC3DONE
