#!/bin/bash
#SBATCH --partition=long
#SBATCH -J HPO-MAE-cos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 160:00:00
#SBATCH --mail-user pauljanson002@gmail.com
#SBATCH --mail-type=END
#SBATCH -o /home/mila/p/paul.janson/scratch/logs/%j.out  
#SBATCH -e /home/mila/p/paul.janson/scratch/logs/%j.err
#SBATCH --gres=gpu:l40s:1

export WANDB_ENITY="eb-lab"

source /home/mila/p/paul.janson/workspace/MAE-1/bin/mila/setup.sh
cd /home/mila/p/paul.janson/workspace/MAE-1

max_device_batch_size=512
prefix=final_rebuttal


# Iterate over all combinations of scheduler and reduction factor
scheduler="infinite_cosine"
reduction_factor=0.5

name=final_rebuttal_infinite
output_dir="output/${name}"
mkdir -p ${output_dir}

python mae_pretrain.py \
    --max_device_batch_size ${max_device_batch_size} \
    --base_learning_rate 0.0006812920690579609 \
    --min_lr_ratio 0.05 \
    --scheduler ${scheduler} \
    --cooldown_ratio 0.5 \
    --constant_lr_ratio 0.75 \
    --reduction_factor ${reduction_factor} \
    --output_dir ${output_dir} \
    --name ${name} \