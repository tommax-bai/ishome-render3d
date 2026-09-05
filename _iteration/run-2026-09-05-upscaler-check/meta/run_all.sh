#!/bin/zsh
# 顺序跑神经网络候选，每个候选每张图一个独立进程（避免 MPS 内存互相影响），全部日志进 run_all.log。
set -u
cd "$(dirname "$0")"
RUN=/Users/baitianxing/codes/ishome-render3d.wt-upscale/_iteration/run-2026-09-05-upscaler-check
S=/Users/baitianxing/codes/ishome-render3d/_iteration/run-2026-09-05-control-sketch-realism
if (( $# > 0 )); then MODELS=("$@"); else MODELS=(realesrgan-x2plus swinir-m-x2-realsr realesrgan-x4plus realesr-general-x4v3); fi
for m in "${MODELS[@]}"; do
  for n in cam-bird-dollhouse cam-room-主卧; do
    echo "=== $m / $n  $(date +%H:%M:%S)"
    /usr/bin/time -l .venv/bin/python "$RUN/放大.py" --model "$m" --input "$S/$n-seed1.png" --out-dir "out/$n" --runs 2 2>&1 | grep -vE "meshgrid|_VF\.meshgrid" | grep -E "run[0-9]|deterministic|maximum resident|real|Error|error|Traceback|File "
  done
done
echo "=== done $(date +%H:%M:%S)"
