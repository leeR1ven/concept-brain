#!/bin/bash
# x2 job files.  usage: bash mkjobs.sh <kind> ...   (writes jobs/<name>.txt)
V21="ral=1 ral_other=1 say_rec=1 say_loc=1 scene_held=1 val_veto=0.2 dip=1 act_T=60 soc_feats=1 rec_fix=1"
CF="rm=1 rm_train=1 rm_prune=1 rm_age=20000 rm_regrow=1 rm_alloc=flat"
VE="veto_R1=1 veto_eps=0.1 veto_W=40 veto_gate=0.15 veto_fast=0.04"
X="marks=300,900,1500,2100,3000 diag=1 les=2100,3000 save_tr=1 save_at=3000 relearn=1 rounds=20"
LF="ticks=1000000 chunk=100000"
full() {  # tag seed extra
  echo "run_x2.py tag=$1 seed=$2 proto=frozen $V21 $CF $VE $3 $X out=out/runs/$1_s$2.json"; }
life() {  # tag seed extra
  echo "life_x2.py tag=$1 seed=$2 $LF $V21 $CF $VE $3"; }
fork() {  # tag seed dset (from ck/x2base_s<seed>_tr.pkl.gz)
  echo "drift_x2.py load=ck/x2base_s$2_tr.pkl.gz tag=$1 seed=$2 marks=300,900,1500,2100,3000 diag=1 les=2100,3000 \"dset=$3\" out=out/runs/drift_$1_s$2.json"; }
