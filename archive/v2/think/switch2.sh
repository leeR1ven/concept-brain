#!/bin/bash
cd "$(cd "$(dirname "$0")" && pwd)"
until grep -q final2_s0 out/batch3.progress; do sleep 2; done
pkill -f batch3.sh; sleep 1; pkill -f "run_life.py"; sleep 2
nohup ./batch4.sh > out/batch4.out 2>&1 &
echo switched $(date +%T)
