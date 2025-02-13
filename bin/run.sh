#!/bin/bash

max_device_batch_size=1024
schedulers=("cosine" "infinite_cosine")
reduction_factors=(0.05 0.1 0.2)
prefix=first


# Keep track of current GPU
gpu_id=0
total_gpus=4

# Iterate over all combinations of scheduler and reduction factor
for scheduler in "${schedulers[@]}"; do
    for reduction_factor in "${reduction_factors[@]}"; do
        name="${prefix}_${scheduler}_${reduction_factor}"
        output_dir="output/${name}"
        mkdir -p ${output_dir}

        # Run on current GPU and increment GPU counter
        echo "Running on GPU $gpu_id:"
        echo "CUDA_VISIBLE_DEVICES=$gpu_id python mae_pretrain.py --max_device_batch_size ${max_device_batch_size} --scheduler ${scheduler} --reduction_factor ${reduction_factor} --output_dir ${output_dir}"
        
        # Move to next GPU, wrapping around if we exceed total GPUs
        gpu_id=$(( (gpu_id + 1) % total_gpus ))
    done
done


CUDA_VISIBLE_DEVICES=0 python mae_pretrain.py --max_device_batch_size 1024 --scheduler cosine --reduction_factor 0.05 --output_dir output/first_cosine_0.05 --name first_cosine_0.05
Running on GPU 1:
CUDA_VISIBLE_DEVICES=1 python mae_pretrain.py --max_device_batch_size 1024 --scheduler cosine --reduction_factor 0.1 --output_dir output/first_cosine_0.1 --name first_cosine_0.1
Running on GPU 2:
CUDA_VISIBLE_DEVICES=2 python mae_pretrain.py --max_device_batch_size 1024 --scheduler cosine --reduction_factor 0.2 --output_dir output/first_cosine_0.2 --name first_cosine_0.2
Running on GPU 3:
CUDA_VISIBLE_DEVICES=3 python mae_pretrain.py --max_device_batch_size 1024 --scheduler infinite_cosine --reduction_factor 0.05 --output_dir output/first_infinite_cosine_0.05 --name first_infinite_cosine_0.05
Running on GPU 0:
CUDA_VISIBLE_DEVICES=0 python mae_pretrain.py --max_device_batch_size 1024 --scheduler infinite_cosine --reduction_factor 0.1 --output_dir output/first_infinite_cosine_0.1 --name first_infinite_cosine_0.1
Running on GPU 1:
CUDA_VISIBLE_DEVICES=1 python mae_pretrain.py --max_device_batch_size 1024 --scheduler infinite_cosine --reduction_factor 0.2 --output_dir output/first_infinite_cosine_0.2 --name first_infinite_cosine_0.2