#!/bin/bash
# llm2hebb g6 (A800 GPU queue): test 1 v2 (token-level entity keys, prefixed text) dev -> 10k facts;
# then test 3, RWKV-7 World 1.5B never-reset stream.
export PYTHONPATH=~/v2lab/venv_llm/site PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
mkdir -p out
F="warn|torch.load|sd = "
python3 store2.py mode=dev out=out/store2.jsonl 2>&1 | grep -v -i -E "$F" | grep -v '"mode": "dev"'
python3 store2.py mode=main n=10000 out=out/store2.jsonl 2>&1 | grep -v -i -E "$F"
R=~/v2lab/models/rwkv7; f=RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth
df -h ~ | tail -1
if [ ! -s $R/$f ]; then
  curl -sL --fail --connect-timeout 20 --max-time 1500 -o $R/$f.part -w "$f: %{http_code} %{size_download} B %{speed_download} B/s\n" \
    "https://www.modelscope.cn/models/Blink_DL/rwkv-7-world/resolve/master/$f" && mv $R/$f.part $R/$f
fi
ls -la $R
python3 r1_stream.py model=$R/$f streams=4 length=131072 facts=24 session=4096 long=262144 tag=w15 out=out/r1.jsonl 2>&1 | grep -v -i -E "$F"
