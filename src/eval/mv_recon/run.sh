#!/bin/bash

set -e
workdir='..'
model_name='OVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='300'

output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}"
echo "$output_dir"
accelerate launch --num_processes 2 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
     