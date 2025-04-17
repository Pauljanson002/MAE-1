#!/bin/bash
#SBATCH --partition=long
#SBATCH -J HPO-MAE-inf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 160:00:00
#SBATCH --mail-user pauljanson002@gmail.com
#SBATCH --mail-type=END
#SBATCH -o /home/mila/p/paul.janson/scratch/logs/%j.out
#SBATCH -e /home/mila/p/paul.janson/scratch/logs/%j.err
#SBATCH --gres=gpu:1


AGENT_ID="2okz7kgr"
source /home/mila/p/paul.janson/workspace/MAE-1/bin/mila/setup.sh
cd /home/mila/p/paul.janson/workspace/MAE-1
CUDA_VISIBLE_DEVICES=0 wandb agent eb-lab/mae-hpo/$AGENT_ID

