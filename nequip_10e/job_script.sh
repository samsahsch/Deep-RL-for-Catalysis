#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C gpu
#SBATCH --gpus-per-task=1
#SBATCH -q regular
#SBATCH -J TRAIN
#SBATCH --mail-user=samss@slac.stanford.edu
#SBATCH --mail-type=ALL
#SBATCH -A m5047
#SBATCH --mem=0
#SBATCH -t 12:0:0
module load conda
conda activate nequip_env
nequip-train -cn train.yaml