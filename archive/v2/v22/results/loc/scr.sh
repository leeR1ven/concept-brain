#!/bin/bash
# local (numpy 1.26) short-life screens, seed 5, 300k ticks, one process at a time; paired only with each other
cd ../v22
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
LF="seed=5 ticks=300000 chunk=100000"
python3 life_v22.py tag=loc_base $LF $V21 > loc/base.log 2>&1
python3 life_v22.py tag=loc_vRe $LF $V21 veto_R1=1 veto_eps=0.1 > loc/vRe.log 2>&1
python3 life_v22.py tag=loc_u0 $LF $V21 u_R1=0.0 > loc/u0.log 2>&1
python3 life_v22.py tag=loc_vReu $LF $V21 veto_R1=1 veto_eps=0.1 u_R1=0.0 > loc/vReu.log 2>&1
echo LOCDONE
