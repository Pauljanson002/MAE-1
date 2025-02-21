#!/bin/bash
#SBATCH -A rrg-eugenium
#SBATCH -J mae_pretraining
#SBATCH -o /home/janson2/scratch/slurm/%j.out
#SBATCH -e /home/janson2/scratch/slurm/%j.err
#SBATCH -t 8:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type FAIL 
#SBATCH --mail-user pauljanson002@gmail.com

scheduler=$1
reduction_factor=$2
method=$3
seed=$4
prefix=$5
cooldown_ratio=$6
constant_lr_ratio=$7

module load python/3.10
source env/bin/activate

name=${prefix}_${scheduler}_${reduction_factor}_${method}_${cooldown_ratio}_${constant_lr_ratio}_${seed}
output_dir=output/${name}
mkdir -p ${output_dir}


if [ "$scheduler" = "cosine" ]; then
    python mae_pretrain.py \
        --scheduler=${scheduler} \
        --reduction_factor=${reduction_factor} \
        --name=${name} \
        --output_dir=${output_dir} \
        --method=${method} \
        --seed=${seed} \
        --base_learning_rate=7.5e-5 \
        --min_lr_ratio=0.1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --weight_decay=5e-3 \
        --batch_size=512 
else
    python mae_pretrain.py \
        --scheduler=${scheduler} \
        --reduction_factor=${reduction_factor} \
        --name=${name} \
        --output_dir=${output_dir} \
        --method=${method} \
        --seed=${seed} \
        --base_learning_rate=7.5e-5 \
        --min_lr_ratio=0.1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --weight_decay=5e-3 \
        --warmup_ratio=0.05 \
        --cooldown_ratio=0.3 \
        --constant_lr_ratio=0.25 \
        --constant_ratio=0.8 \
        --batch_size=512 \
        --cooldown_ratio=${cooldown_ratio} \
        --constant_lr_ratio=${constant_lr_ratio}
fi