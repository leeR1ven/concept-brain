#!/bin/bash
# llm2hebb env probe + one-time download of two public pretrained models into ~/v2lab/models/
echo "== python packages"
python3 - <<'PY'
import importlib
for m in ['transformers','tokenizers','safetensors','huggingface_hub','sentencepiece','accelerate','regex','jieba','tiktoken']:
    try:
        x = importlib.import_module(m); print(m, getattr(x, '__version__', '?'))
    except Exception as e:
        print(m, 'MISSING', type(e).__name__)
PY
echo "== tools"; which curl wget git-lfs 2>&1
echo "== disk"; df -h ~ | tail -1; mkdir -p ~/v2lab/models; ls -la ~/v2lab/models
echo "== network"
for u in https://hf-mirror.com https://huggingface.co; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 $u/api/models/Qwen/Qwen2.5-0.5B-Instruct); echo "$u -> $code"
done
dl() {  # repo file dest
  local repo=$1 f=$2 d=$3
  mkdir -p "$(dirname $d/$f)"
  if [ -s "$d/$f" ]; then echo "have $f"; return 0; fi
  for base in https://hf-mirror.com https://huggingface.co; do
    if curl -sL --fail --retry 3 --max-time 1500 -o "$d/$f.part" "$base/$repo/resolve/main/$f"; then
      mv "$d/$f.part" "$d/$f"; echo "got $f from $base ($(du -h $d/$f | cut -f1))"; return 0
    fi
    rm -f "$d/$f.part"
  done
  echo "FAILED $repo $f"; return 1
}
Q=~/v2lab/models/Qwen2.5-0.5B-Instruct
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do dl Qwen/Qwen2.5-0.5B-Instruct $f $Q; done
B=~/v2lab/models/bge-small-zh-v1.5
for f in config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json; do dl BAAI/bge-small-zh-v1.5 $f $B; done
dl BAAI/bge-small-zh-v1.5 model.safetensors $B || dl BAAI/bge-small-zh-v1.5 pytorch_model.bin $B
ls -la $Q $B
echo "== load test"
python3 - <<'PY'
import os, torch, time
Q = os.path.expanduser('~/v2lab/models/Qwen2.5-0.5B-Instruct')
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(Q)
    m = AutoModelForCausalLM.from_pretrained(Q, torch_dtype=torch.float32).cuda().eval()
    ids = tok('甲：我的小狗叫豆米。\n乙：好的。\n甲：我的小狗叫什么名字？\n乙：你的小狗叫', return_tensors='pt').input_ids.cuda()
    print('tokens', [tok.decode([i]) for i in ids[0].tolist()])
    t = time.time()
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=6, do_sample=False)
    print('gen', repr(tok.decode(out[0, ids.shape[1]:])), round(time.time() - t, 2), 's')
    for w in ['苹果', '香蕉', '小狗', '豆米', '嘟嘟', '苹', '蕉']:
        print(w, tok(w).input_ids)
except Exception as e:
    import traceback; traceback.print_exc()
PY
