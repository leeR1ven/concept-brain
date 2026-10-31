#!/bin/bash
# llm2hebb g3 (A800 GPU queue): R3 episodic key->value store on the same keys (no weight change); R3 weaker write;
# R2 online stream with smaller SGD step sizes (g2's 0.01-10 destroyed the model).
export PYTHONPATH=~/v2lab/venv_llm/site PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
mkdir -p out
F="warn|torch.load|sd = "
python3 r3.py mode=knn n=10000 layer=18 thetas=0.8,0.9,0.95,0.98 out=out/r3_knn.jsonl 2>&1 | grep -v -i -E "$F"
python3 r3.py mode=seq n=3000 layer=18 site=last value=unemb s=0.5 cov=C tag=u05C out=out/r3_seq.jsonl 2>&1 | grep -v -i -E "$F"
run() { python3 r2.py mode=online delay=60 out=out/r2_online.jsonl "$@" 2>&1 | grep -v -i -E "$F"; }
for lr in 1e-4 3e-4 1e-3; do run method=bp lr=$lr; done
run method=adam lr=3e-6
for lr in 1e-3 1e-2; do run method=bpdown lr=$lr; done
for lr in 1e-3 1e-2; do run method=ll lr=$lr; done
for lr in 1e-5 1e-4; do run method=dfa lr=$lr; done
