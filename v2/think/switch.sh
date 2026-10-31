#!/bin/bash
cd "$(cd "$(dirname "$0")" && pwd)"
until grep -q frozen_final out/batch1.progress; do sleep 2; done
pkill -f batch1.sh; sleep 1; pkill -f "eval_frozen.py out/think_free_s0"; pkill -f "run_life.py"; sleep 2
nohup ./batch3.sh > out/batch3.out 2>&1 &
echo switched $(date +%T)
