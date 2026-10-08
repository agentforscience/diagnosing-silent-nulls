#!/bin/bash
# Run one pipeline stage with retries (shared GPUs occasionally cause OOM).
# usage: run_all.sh generate|judge <subject> <n_conditions>
stage=$1; subject=$2; n=$3
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
dir=results/model_outputs; [ "$stage" = judge ] && dir=results/evaluations
for i in $(seq 1 60); do
  [ "$(ls $dir/$subject/*.jsonl 2>/dev/null | wc -l)" -ge "$n" ] && break
  python src/$stage.py --subject $subject
  sleep 40
done
