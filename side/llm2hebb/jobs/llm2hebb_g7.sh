#!/bin/bash
# llm2hebb g7 (A800 GPU queue): store v2 with a real site gate and two new false-fire tests (entity mentioned
# without a question; non-answer positions of the QA prompt), dev (layers 0, 2) + 10k facts at ts 0 / 0.1 / 0.2 / 0.3.
export PYTHONPATH=~/v2lab/venv_llm/site PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
mkdir -p out
F="warn|torch.load"
python3 store2.py mode=dev layers=0,2 out=out/store2b.jsonl 2>&1 | grep -v -i -E "$F" | grep -v '"mode": "dev"'
python3 store2.py mode=main le=0 tt=0.5 te=1.0 ts=0.1 tss=0.0,0.1,0.2,0.3 n=10000 out=out/store2b.jsonl 2>&1 | grep -v -i -E "$F"
