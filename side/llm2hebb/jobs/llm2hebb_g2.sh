#!/bin/bash
# llm2hebb g2 (A800 GPU queue): R1 never-reset RWKV-7 streams; R3 sequential Hebbian writes + FT baselines;
# R2 extra PC cosines (smaller relaxation steps) and the online stream.
export PYTHONPATH=~/v2lab/venv_llm/site PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
mkdir -p out
F="warn|torch.load|sd = "
R=~/v2lab/models/rwkv7
python3 r1_stream.py model=$R/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth streams=8 length=262144 facts=40 session=4096 tag=w04 out=out/r1.jsonl 2>&1 | grep -v -i -E "$F"
for cov in C I S64; do
  python3 r3.py mode=seq n=10000 layer=18 site=last value=unemb s=1.0 cov=$cov tag=u1$cov out=out/r3_seq.jsonl 2>&1 | grep -v -i -E "$F"
done
python3 r3.py mode=ft params=L layer=18 lr=5e-4 n=1000 tag=ftL out=out/r3_ft.jsonl 2>&1 | grep -v -i -E "$F"
python3 r3.py mode=ft params=all layer=18 lr=1e-5 n=1000 tag=ftall out=out/r3_ft.jsonl 2>&1 | grep -v -i -E "$F"
python3 r2.py mode=cos nl=4 local=0 pcs=48:0.1:0.1,192:0.1:0.1,192:0.03:0.1,24:1.0:0.1 out=out/r2_cos2.jsonl 2>&1 | grep -v -i -E "$F"
run() { python3 r2.py mode=online delay=60 out=out/r2_online.jsonl "$@" 2>&1 | grep -v -i -E "$F"; }
for lr in 0.01 0.03 0.1; do run method=bp lr=$lr; done
for lr in 1e-5 3e-5; do run method=adam lr=$lr; done
for lr in 0.1 1; do run method=bpdown lr=$lr; done
for lr in 0.1 1 10; do run method=ll lr=$lr; done
run method=dfa lr=1
python3 r1_stream.py model=$R/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth streams=8 length=262144 facts=40 session=4096 tag=w01 out=out/r1.jsonl 2>&1 | grep -v -i -E "$F"
python3 r3.py mode=seq n=3000 layer=18 site=last value=icl g=2.0 cov=C tag=icl2C out=out/r3_seq.jsonl 2>&1 | grep -v -i -E "$F"
