#!/bin/bash
cd ../v22
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
LF="seed=6 ticks=600000 chunk=100000"
python3 life_v22.py tag=loc6_base $LF $V21 > loc/b6.log 2>&1
python3 life_v22.py tag=loc6_vRe3 $LF $V21 veto_R1=1 veto_eps=0.3 > loc/vRe3_6.log 2>&1
python3 life_v22.py tag=loc6_vRe3u $LF $V21 veto_R1=1 veto_eps=0.3 u_R1=0.0 > loc/vRe3u_6.log 2>&1
echo LOC2DONE
