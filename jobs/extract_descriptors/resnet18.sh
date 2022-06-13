#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
srun python3 -m python3 -m cnn.feature_extraction ./data/descriptors --truth_dir ./data/ground_truth --im_dir ./data/ --net resnet18 --mini