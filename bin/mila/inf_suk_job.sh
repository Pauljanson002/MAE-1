#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH -J HPO-MAE-inf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=995G
#SBATCH -t 3:00:00
#SBATCH --mail-user pauljanson002@gmail.com
#SBATCH --mail-type=END
#SBATCH -o /home/mila/p/paul.janson/scratch/logs/%j.out
#SBATCH -e /home/mila/p/paul.janson/scratch/logs/%j.err
#SBATCH --gres=gpu:l40s:4


AGENT_ID="2okz7kgr"

source /home/mila/p/paul.janson/workspace/MAE-1/bin/mila/setup.sh
cd /home/mila/p/paul.janson/workspace/MAE-1
CUDA_VISIBLE_DEVICES=0 wandb agent eb-lab/mae-hpo/$AGENT_ID &
CUDA_VISIBLE_DEVICES=1 wandb agent eb-lab/mae-hpo/$AGENT_ID &
CUDA_VISIBLE_DEVICES=2 wandb agent eb-lab/mae-hpo/$AGENT_ID &
CUDA_VISIBLE_DEVICES=3 wandb agent eb-lab/mae-hpo/$AGENT_ID

