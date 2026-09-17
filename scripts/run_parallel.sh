#!/usr/bin/env bash
# Distribute a newline-separated job list across a GPU list, N workers per GPU.
# Usage: run_parallel.sh <jobfile> <gpu-list-csv> <jobs_per_gpu>
#   e.g. run_parallel.sh jobs.txt 0,1 2     -> 4 workers
#        run_parallel.sh jobs.txt 1 3       -> 3 workers, GPU 1 only
set -u
JOBS="$1"; GPUS="${2:-0,1}"; PER="${3:-1}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}
TOTAL=$(( NG * PER ))
echo "running $(wc -l < "$JOBS") jobs on GPU(s) $GPUS x $PER = $TOTAL workers"
for (( w=0; w<TOTAL; w++ )); do
  GPU=${GPU_ARR[$(( w % NG ))]}
  awk -v w="$w" -v t="$TOTAL" 'NR % t == w' "$JOBS" | while read -r cmd; do
    [ -z "$cmd" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU bash -c "$cmd"
  done &
done
wait
echo "ALL JOBS DONE"
