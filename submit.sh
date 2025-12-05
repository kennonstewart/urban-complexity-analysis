#!/bin/bash
#SBATCH --job-name=urban_complexity_analysis
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=0-05:00:00
#SBATCH --account=your_slurm_account
#SBATCH --output=%x-%j.log

# module load python

pip install --user -r requirements.txt

python analysis.py
