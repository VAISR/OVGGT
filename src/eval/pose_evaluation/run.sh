#!/bin/bash

set -e

workdir='..'
model_names=('OVGGT') 

ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"


# datasets=('sintel')
datasets=('tum_100')

for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/pose_evaluation/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29602 ./eval/pose_evaluation/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 518 \
        # --model_update_type "$model_name"
done
done