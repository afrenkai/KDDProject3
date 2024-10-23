#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=70g
#SBATCH -J "Tune Model"
#SBATCH -p short
#SBATCH -t 16:00:00
#SBATCH --output=/home/sppradhan/KDDProject3/logs/TuneModels_%j.txt

module load python/3.10.13
source ~/KDDProject3/kdd/bin/activate
cd model_tuning
python tune_model.py
