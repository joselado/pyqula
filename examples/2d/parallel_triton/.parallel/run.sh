#!/bin/bash
#SBATCH -n 1
#SBATCH -t 0:6:00
#SBATCH --mem-per-cpu=5000
#SBATCH --array=0-9
srun python run.py
