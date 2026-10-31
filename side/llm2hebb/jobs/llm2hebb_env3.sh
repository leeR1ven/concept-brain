#!/bin/bash
# llm2hebb env3: venv (pip via the server's aliyun mirror) + models from ModelScope (hf-mirror.com is unreachable/slow)
V=~/v2lab/venv_llm
[ -x $V/bin/python ] || python3 -m venv --system-site-packages $V
$V/bin/python -m pip install -q --disable-pip-version-check transformers safetensors tokenizers rwkv 2>&1 | tail -2
$V/bin/python -c "import transformers, tokenizers, safetensors, rwkv, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
MS=https://www.modelscope.cn
ls_repo() { curl -s --connect-timeout 15 --max-time 60 "$MS/api/v1/models/$1/repo/files?Revision=master&Recursive=True"; }
get() {  # repo file dest
  local d=$3; mkdir -p $d
  if [ -s $d/$2 ]; then echo "have $2"; return 0; fi
  curl -sL --fail --connect-timeout 20 --max-time 1200 -o $d/$2.part -w "$2: %{http_code} %{size_download} B %{speed_download} B/s\n" "$MS/models/$1/resolve/master/$2" && mv $d/$2.part $d/$2
}
Q=~/v2lab/models/Qwen2.5-0.5B-Instruct
rm -f $Q/*.part
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do get Qwen/Qwen2.5-0.5B-Instruct $f $Q; done
ls -la $Q
R=~/v2lab/models/rwkv7; mkdir -p $R
for repo in Blink_DL/rwkv-7-world BlinkDL/rwkv-7-world RWKV/rwkv-7-world AI-ModelScope/rwkv-7-world Blink_DL/rwkv-7-g1; do
  ls_repo $repo > $R/ls_$(echo $repo | tr '/' '_').json
  echo "$repo: $(python3 -c "import json,sys; d=json.load(open('$R/ls_$(echo $repo | tr '/' '_').json')); print([ (f['Path'], f.get('Size')) for f in d.get('Data',{}).get('Files',[]) if f['Path'].endswith('.pth')])" 2>&1 | head -c 1500)"
done
PICK=$(python3 - <<'PY'
import json, glob, os
R = os.path.expanduser('~/v2lab/models/rwkv7')
for fn in sorted(glob.glob(R + '/ls_*.json')):
    try:
        d = json.load(open(fn))
    except Exception:
        continue
    fs = [f['Path'] for f in d.get('Data', {}).get('Files', []) if f['Path'].endswith('.pth')]
    pick = []
    for size in ['0.1B', '0.4B']:
        c = sorted(f for f in fs if size in f and 'World' in f)
        if c:
            pick.append(c[-1])
    if len(pick) == 2:
        repo = os.path.basename(fn)[3:-5].replace('_', '/', 1) if not os.path.basename(fn).startswith('ls_Blink_DL') else 'Blink_DL/' + os.path.basename(fn)[12:-5]
        print(repo, *pick)
        break
PY
)
echo "pick: $PICK"
set -- $PICK
REPO=$1; shift
for f in "$@"; do get $REPO $f $R; done
ls -la $R
$V/bin/python check_rwkv.py $(ls $R/*.pth | sort)
$V/bin/python - <<'PY'
import os, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
Q = os.path.expanduser('~/v2lab/models/Qwen2.5-0.5B-Instruct')
tok = AutoTokenizer.from_pretrained(Q)
m = AutoModelForCausalLM.from_pretrained(Q, torch_dtype=torch.float32, attn_implementation='sdpa').cuda().eval()
ids = tok('我的小狗叫豆米。\n问：我的小狗叫什么名字？\n答：你的小狗叫', return_tensors='pt').input_ids.cuda()
out = m.generate(ids, max_new_tokens=5, do_sample=False)
print('qwen gen', repr(tok.decode(out[0, ids.shape[1]:])))
print('decoder layer forward args', list(__import__('inspect').signature(m.model.layers[0].forward).parameters))
PY
