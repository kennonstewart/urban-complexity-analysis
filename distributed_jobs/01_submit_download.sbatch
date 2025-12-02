#!/bin/bash
#SBATCH --job-name=urban-complexity-download
#SBATCH --account=sstoev0
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/sstoev_root/sstoev0/ksstewar/uca_01_%x_%j.out
#SBATCH --error=/scratch/sstoev_root/sstoev0/ksstewar/uca_01_%x_%j.err

# Download city road networks sequentially
# This job should be run before the analysis job array

# Load modules (adjust based on your HPC environment)
# module load python/3.9
# module load gdal

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Change to the distributed_jobs directory
cd "$(dirname "$0")"

# Run the download script
python download.py
