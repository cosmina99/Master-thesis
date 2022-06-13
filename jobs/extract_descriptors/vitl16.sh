#!/bin/bash
#SBATCH --time 1:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
srun python3 -m transformer.feature_extraction ./data/descriptors --truth_dir ./data/ground_truth --im_dir ./data/ --net vitl16 --mini