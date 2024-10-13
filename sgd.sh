#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100g
#SBATCH -J "Stochastic Gradient Descent"
#SBATCH -p academic
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/afrenk/KDDProject3/logs/SGD_%j.txt

module load cuda
module load python/3.10.13
source ~/KDDProject3/kdd/bin/activate
cd model\_runs
python sgd.py
