#!/bin/bash
#SBATCH --job-name=load_system
#SBATCH --partition=milan-gpu
#SBATCH --gress=gpu:1
#SBATCH --mem=4G

module load gcc/11.3.0
source team5/bin/activate
cd ProjectCSurvival
python ~/test.py