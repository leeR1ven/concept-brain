#!/bin/bash
# llm2hebb g5 (A800 GPU queue): test 1b, a sparse feature layer between the dense MLP key and the Hebbian write.
# Sweep of the active fraction (dense -> 0.1 %), N 64k / 16k, a top-k sparse autoencoder; site pred (+ both).
export PYTHONPATH=~/v2lab/venv_llm/site PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
mkdir -p out
F="warn|torch.load"
run() { python3 sparse.py layer=18 scale=1 n=10000 out=out/sparse.jsonl "$@" 2>&1 | grep -v -i -E "$F"; }
run code=dense N=65536 frac=1 tag=dense64k
for fr in 0.1 0.02 0.005 0.001; do run code=rp N=65536 frac=$fr tag=rp64k_$fr; done
run code=rp N=16384 frac=0.01 tag=rp16k_0.01
run code=sae N=16384 sae_k=64 tag=sae16k_k64
run code=rp N=65536 frac=0.005 site=both tag=rp64k_0.005_both
