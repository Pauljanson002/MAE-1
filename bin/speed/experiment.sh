#!/bin/bash

reduction_factor=(0.05 0.1 0.2)
seed=(0 1 2 3 4)

max_device_batch_size=512
scheduler="infinite_cosine"
prefix="nonsweep"

# Remove commas from arrays and loop through combinations
counter=0
for rf in "${reduction_factor[@]}"; do
    for s in "${seed[@]}"; do
        # Calculate GPU ID (0-3) using modulo to cycle through available GPUs
        gpu_id=$((counter % 4))
        
        name="${prefix}-${scheduler}-reduction_factor_${rf}-seed_${s}"
        output_dir="output/${name}"
        mkdir -p "${output_dir}"
        
        echo "Running on GPU ${gpu_id}: reduction_factor=${rf}, seed=${s}"
        echo "CUDA_VISIBLE_DEVICES=${gpu_id} python mae_pretrain.py \
            --reduction_factor=${rf} \
            --seed=${s} \
            --output_dir=${output_dir} \
            --max_device_batch_size=${max_device_batch_size} \
            --name=${name} \
            --scheduler=${scheduler}"
        counter=$((counter + 1))
    done
done