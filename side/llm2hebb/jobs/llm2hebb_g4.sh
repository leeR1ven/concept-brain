#!/bin/bash
# llm2hebb g4 (A800 GPU queue): test 1 entity-keyed DG store (dev grid -> 10k facts, DG and dense keys);
# test 2 replay consolidation into the weights (4 conditions + one higher lr).
export PYTHONPATH=~/v2lab/venv_llm/site PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
mkdir -p out
F="warn|torch.load"
python3 store.py mode=dev out=out/store.jsonl 2>&1 | grep -v -i -E "$F" | grep -v '"mode": "dev"'
python3 store.py mode=main coder=dg n=10000 out=out/store.jsonl 2>&1 | grep -v -i -E "$F"
python3 store.py mode=main coder=dense n=10000 out=out/store.jsonl 2>&1 | grep -v -i -E "$F"
for c in cls notext noreh newonly; do
  python3 consolidate.py cond=$c lr=1e-5 nf=2000 day=100 steps=150 out=out/consolidate.jsonl 2>&1 | grep -v -i -E "$F"
done
python3 consolidate.py cond=cls lr=3e-5 nf=2000 day=100 steps=150 out=out/consolidate.jsonl 2>&1 | grep -v -i -E "$F"
