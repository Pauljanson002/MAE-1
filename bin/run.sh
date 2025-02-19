#!/bin/bash

max_device_batch_size=512
schedulers=("cosine" "infinite_cosine")
reduction_factors=(0.05 0.1 0.2)
prefix=first


# Iterate over all combinations of scheduler and reduction factor
scheduler=${schedulers[0]}
reduction_factor=${reduction_factors[0]}

name=tmp
output_dir="output/${name}"
mkdir -p ${output_dir}

python mae_pretrain.py \
    --max_device_batch_size ${max_device_batch_size} \
    --scheduler ${scheduler} \
    --reduction_factor ${reduction_factor} \
    --output_dir ${output_dir} \
    --method=mas \
    --total_epoch=5 \
    --finetune_epoch=5 \
    --hpo=1 \
