#!/bin/bash
# causal audit of the main agent: every learned part silenced on the same content (learning stays on)
cd "$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
L1='none,floor=0.12,x1+x2,start,gline,ghold,gtopic,B,V,V2,C2,H,H2,Brec,gate,S,go,fwd'
nice python3 -B lesion_eval.py main_s0_agent.pkl --stage B --n 120 --lesions $L1 --out audit_B > audit_B.log 2>&1
nice python3 -B lesion_eval.py main_s0_agent.pkl --stage C --n 120 --lesions $L1 --out audit_C > audit_C.log 2>&1
L2='none,floor=0.12,Brec,H2,H,gate,S,go,fwd,ghold,C2'
nice python3 -B lesion_eval.py main_s0_agent.pkl --stage D --n 150 --lesions $L2 --out audit_D > audit_D.log 2>&1
echo AUDIT_DONE
