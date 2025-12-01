#!/bin/bash
#SBATCH --job-name=download_cities
#SBATCH --output=download_%j.log
#SBATCH --error=download_%j.err
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

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
