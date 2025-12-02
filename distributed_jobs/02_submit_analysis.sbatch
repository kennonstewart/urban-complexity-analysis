#!/bin/bash
#SBATCH --job-name=urban-complexity-download
#SBATCH --account=sstoev0
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-499%50
#SBATCH --output=/scratch/sstoev_root/sstoev0/ksstewar/uca_02_%x_%A_%a.out
#SBATCH --error=/scratch/sstoev_root/sstoev0/ksstewar/uca_02_%x_%A_%a.err

# Analyze city road networks in parallel using a job array
# This job should be run after the download job completes
# Submit with: sbatch --dependency=afterok:<download_job_id> 02_submit_analysis.sbatch

# Load modules (adjust based on your HPC environment)
module load python/3.9
module load gdal

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Change to the distributed_jobs directory
cd "$(dirname "$0")"

# Get the city key corresponding to this task ID
# SLURM_ARRAY_TASK_ID is 0-indexed, sed is 1-indexed
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
CITY_KEY=$(sed "${LINE_NUM}q;d" ../data/manifest.txt)

if [ -z "$CITY_KEY" ]; then
    echo "Error: No city key found for task ID $SLURM_ARRAY_TASK_ID in manifest"
    exit 1
fi

echo "Processing city: $CITY_KEY"
python analyze_city.py --city "$CITY_KEY"