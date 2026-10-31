#!/bin/bash
# llm2hebb network diagnosis (short timeouts)
ls -la ~/v2lab/models/ ~/v2lab/models/* 2>&1 | head -30
pip3 config list 2>&1; cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null
for u in https://pypi.org/simple/rwkv/ https://pypi.tuna.tsinghua.edu.cn/simple/rwkv/ https://mirrors.aliyun.com/pypi/simple/rwkv/ https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct https://hf-mirror.com/api/models/BlinkDL/rwkv-7-world; do
  echo "$u -> $(curl -s -o /dev/null -w '%{http_code} %{time_total}s' --connect-timeout 10 --max-time 30 $u)"
done
echo "== redirect of LFS file on hf-mirror"
curl -sI --connect-timeout 10 --max-time 30 https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json | head -20
curl -sI --connect-timeout 10 --max-time 30 https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors | grep -i -E "^HTTP|location|content-length|x-linked" | head
echo "== timed download of tokenizer.json (60 s max)"
curl -sL --connect-timeout 10 --max-time 60 -o /tmp/llm2hebb_tok.json -w '%{http_code} %{size_download} bytes %{speed_download} B/s %{url_effective}\n' https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
ls -la /tmp/llm2hebb_tok.json; rm -f /tmp/llm2hebb_tok.json
echo "== pip download test"
timeout 60 pip3 download --no-deps -d /tmp/llm2hebb_pip rwkv 2>&1 | tail -3
timeout 60 pip3 download --no-deps -d /tmp/llm2hebb_pip -i https://pypi.tuna.tsinghua.edu.cn/simple rwkv 2>&1 | tail -3
ls /tmp/llm2hebb_pip; rm -rf /tmp/llm2hebb_pip
