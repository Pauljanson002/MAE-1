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


AGENT_ID="l2l0d9ps"
source /home/mila/p/paul.janson/workspace/MAE-1/bin/mila/setup.sh
cd /home/mila/p/paul.janson/workspace/MAE-1
CUDA_VISIBLE_DEVICES=0 wandb agent eb-lab/mae-hpo/$AGENT_ID

