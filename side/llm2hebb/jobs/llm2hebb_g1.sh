#!/bin/bash
# llm2hebb g1 (A800 GPU queue): R3 dev grid -> sequential Hebbian writes (auto-picked configs) -> sequential FT
# baselines; R2 gradient-cosine table.  Qwen2.5-0.5B-Instruct, fp32.
export PYTHONPATH=~/v2lab/venv_llm/site PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
mkdir -p out
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 r3.py mode=dev out=out/r3_dev.jsonl 2>&1 | grep -v -i warn
PI=$(python3 pick.py out/r3_dev.jsonl I); PC=$(python3 pick.py out/r3_dev.jsonl C); PS=$(python3 pick.py out/r3_dev.jsonl subj)
echo "picked: I: $PI | C: $PC | subj: $PS"
python3 r3.py mode=seq n=10000 tag=bestC $PC out=out/r3_seq.jsonl 2>&1 | grep -v -i warn
python3 r3.py mode=seq n=10000 tag=bestI $PI out=out/r3_seq.jsonl 2>&1 | grep -v -i warn
[ "$PS" != "$PC" ] && python3 r3.py mode=seq n=3000 tag=subjC $PS out=out/r3_seq.jsonl 2>&1 | grep -v -i warn
python3 r2.py mode=cos out=out/r2_cos.jsonl 2>&1 | grep -v -i warn
LC=$(echo $PC | sed 's/.*layer=\([0-9]*\).*/\1/')
python3 r3.py mode=ft params=L layer=$LC lr=5e-4 n=3000 tag=ftL out=out/r3_ft.jsonl 2>&1 | grep -v -i warn
python3 r3.py mode=ft params=all layer=$LC lr=1e-5 n=1000 tag=ftall out=out/r3_ft.jsonl 2>&1 | grep -v -i warn
