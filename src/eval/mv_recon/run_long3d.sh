#!/bin/bash

set -e
workdir='..'
model_name='OVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
data_root="${workdir}/data/Long3D"
kf_every='1'

output_dir="${workdir}/eval_results/mv_recon_long3d/${model_name}_${ckpt_name}"
echo "$output_dir"
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch_long3d.py \
    --weights "$model_weights" \
    --model_name "$model_name" \
    --root "$data_root" \
    --output_dir "$output_dir" \
    --kf_every "$kf_every"
