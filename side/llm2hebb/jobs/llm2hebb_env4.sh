#!/bin/bash
# llm2hebb env4: python3-venv is not installed on the server, so packages go to ~/v2lab/venv_llm/site (pip --target),
# used through PYTHONPATH with the system python3 (torch 2.4).
S=~/v2lab/venv_llm/site
rm -rf ~/v2lab/venv_llm; mkdir -p $S
pip3 install -q --disable-pip-version-check --no-deps --target $S transformers==4.46.3 tokenizers==0.20.3 \
  huggingface-hub==0.26.2 safetensors==0.4.5 regex tqdm pyyaml rwkv==0.8.32 2>&1 | grep -v notice | tail -3
export PYTHONPATH=$S
python3 -c "import transformers, tokenizers, safetensors, rwkv, torch, numpy; print('transformers', transformers.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)"
python3 check_rwkv.py $(ls ~/v2lab/models/rwkv7/*.pth | sort)
python3 - <<'PY'
import os, torch, time, inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
Q = os.path.expanduser('~/v2lab/models/Qwen2.5-0.5B-Instruct')
tok = AutoTokenizer.from_pretrained(Q)
m = AutoModelForCausalLM.from_pretrained(Q, torch_dtype=torch.float32, attn_implementation='sdpa').cuda().eval()
for q in ['我的小狗叫豆米。\n问：我的小狗叫什么名字？\n答：你的小狗叫', '王子涵的小狗叫栗蔻。\n王子涵的小狗叫']:
    ids = tok(q, return_tensors='pt').input_ids.cuda()
    out = m.generate(ids, max_new_tokens=5, do_sample=False)
    print('qwen gen', repr(q[-8:]), '->', repr(tok.decode(out[0, ids.shape[1]:])))
print('decoder layer forward args', list(inspect.signature(m.model.layers[0].forward).parameters))
print([tok.convert_ids_to_tokens(i) for i in tok('王子涵的小狗叫栗蔻。').input_ids])
PY
