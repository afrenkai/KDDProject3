#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100g
#SBATCH -J "Get Data"
#SBATCH -p academic
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/afrenk/KDDProject3/logs/Data_%j.txt
module load cuda
module load python/3.10.13
source ~/KDDProject3/kdd/bin/activate
python get_data.py