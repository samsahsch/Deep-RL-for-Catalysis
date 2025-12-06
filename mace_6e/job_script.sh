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
#SBATCH -t 18:0:0
module load conda
conda activate mace_env
python /global/cfs/cdirs/m5047/train_sam/mace/mace/cli/run_train.py \
    --name="MACE_model" \
    --train_file="6e.xyz" \
    --valid_fraction=0.1 \
    --E0s='average' \
    --model="MACE" \
    --energy_key="energy" \
    --forces_key="forces" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=6.0 \
    --batch_size=4 \
    --default_dtype=float32 \
    --max_num_epochs=200 \
    --swa \
    --start_swa=150 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
