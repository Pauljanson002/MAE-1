#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH -J HPO-MAE-cos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=995G
#SBATCH -t 3:00:00
#SBATCH --mail-user btherien@uwaterloo.ca
#SBATCH --mail-type=END
#SBATCH -o /home/mila/b/benjamin.therien/log/HPO/%j.out
#SBATCH -e /home/mila/b/benjamin.therien/log/HPO/%j.err
#SBATCH --gres=gpu:l40s:4


AGENT_ID="l2l0d9ps"

source ~/collas_install/miniconda3/bin/activate
cd ~/collas_install/MAE-1
CUDA_VISIBLE_DEVICES=0 wandb agent eb-lab/mae-hpo/$AGENT_ID &
CUDA_VISIBLE_DEVICES=1 wandb agent eb-lab/mae-hpo/$AGENT_ID &
CUDA_VISIBLE_DEVICES=2 wandb agent eb-lab/mae-hpo/$AGENT_ID &
CUDA_VISIBLE_DEVICES=3 wandb agent eb-lab/mae-hpo/$AGENT_ID

