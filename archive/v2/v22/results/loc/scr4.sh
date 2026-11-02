#!/bin/bash
cd ../v22
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
CF="rm=1 rm_train=1 rm_prune=1 rm_age=20000 rm_regrow=1 rm_alloc=flat"
LF="seed=6 ticks=600000 chunk=100000"
python3 life_v22.py tag=loc6_cf $LF $V21 $CF > loc/cf6.log 2>&1
python3 life_v22.py tag=loc6_cfvW $LF $V21 $CF veto_R1=1 veto_eps=0.1 veto_W=40 > loc/cfvW6.log 2>&1
echo LOC4DONE
