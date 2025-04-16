#!/bin/bash
#SBATCH --partition=long
#SBATCH -J HPO-MAE-inf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 160:00:00
#SBATCH --mail-user btherien@uwaterloo.ca
#SBATCH --mail-type=END
#SBATCH -o /home/mila/b/benjamin.therien/log/HPO/%j.out
#SBATCH -e /home/mila/b/benjamin.therien/log/HPO/%j.err
#SBATCH --gres=gpu:l40s:1


AGENT_ID="2okz7kgr"
source ~/collas_install/miniconda3/bin/activate
cd ~/collas_install/MAE-1
CUDA_VISIBLE_DEVICES=0 wandb agent eb-lab/mae-hpo/$AGENT_ID

