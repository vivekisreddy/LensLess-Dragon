#!/bin/bash

#SBATCH --mail-user=hkortus@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J run2_turing
#SBATCH --output=/home/hkortus/mqp/lensless_perception/Unet_Depth_Esimation/logs/run2_turing%j.out
#SBATCH --error=/home/hkortus/mqp/lensless_perception/Unet_Depth_Esimation/logs/run2_turing%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100|A30
#SBATCH -p academic
#SBATCH -t 6:00:00
module load cuda
source /home/hkortus/mqp/lensless_perception/Unet_Depth_Esimation/venv/bin/activate
python3 /home/hkortus/mqp/lensless_perception/Unet_Depth_Esimation/train.py
