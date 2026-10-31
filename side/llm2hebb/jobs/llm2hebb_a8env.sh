#!/bin/bash
# llm2hebb a8env: packages into ~/v2lab/venv_llm/site (pip --target; no python3-venv), models from ModelScope
S=~/v2lab/venv_llm/site
rm -rf ~/v2lab/venv_llm; mkdir -p $S
pip3 install -q --disable-pip-version-check --no-deps --target $S transformers==4.46.3 tokenizers==0.20.3 \
  huggingface-hub==0.26.2 safetensors==0.4.5 regex tqdm pyyaml rwkv==0.8.32 2>&1 | grep -v notice | tail -3
MS=https://www.modelscope.cn
get() {  # repo file dest
  local d=$3; mkdir -p $d
  if [ -s $d/$2 ]; then echo "have $2"; return 0; fi
  curl -sL --fail --connect-timeout 20 --max-time 1500 -o $d/$2.part -w "$2: %{http_code} %{size_download} B %{speed_download} B/s\n" "$MS/models/$1/resolve/master/$2" && mv $d/$2.part $d/$2
}
Q=~/v2lab/models/Qwen2.5-0.5B-Instruct
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do get Qwen/Qwen2.5-0.5B-Instruct $f $Q; done
R=~/v2lab/models/rwkv7
for f in RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth; do get Blink_DL/rwkv-7-world $f $R; done
ls -la $Q $R
export PYTHONPATH=$S
python3 -c "import transformers, tokenizers, safetensors, rwkv, torch, numpy; print('transformers', transformers.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)"
python3 check_rwkv.py $(ls $R/*.pth | sort)
