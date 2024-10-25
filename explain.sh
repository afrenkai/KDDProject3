#!/bin/bash
#SBATCH -N 1
#SBATCH -n 30
#SBATCH --mem=160g
#SBATCH -J "Shap explain"
#SBATCH -p short
#SBATCH -t 20:00:00
#SBATCH --output=/home/sppradhan/KDDProject3/logs/Explain_%j.txt

module load python/3.10.13
source ~/KDDProject3/kdd/bin/activate
cd explainability
python explain.py
