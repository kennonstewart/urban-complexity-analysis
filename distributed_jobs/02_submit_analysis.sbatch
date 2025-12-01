#!/bin/bash
#SBATCH --job-name=analyze_city
#SBATCH --output=analyze_%A_%a.log
#SBATCH --error=analyze_%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=1-4

# Analyze city road networks in parallel using a job array
# This job should be run after the download job completes
# Submit with: sbatch --dependency=afterok:<download_job_id> 02_submit_analysis.sbatch

# Load modules (adjust based on your HPC environment)
# module load python/3.9
# module load gdal

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Change to the distributed_jobs directory
cd "$(dirname "$0")"

# Read the city key from manifest.txt based on the array task ID
CITY_KEY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../data/manifest.txt)

# Check if we got a valid city key
if [ -z "$CITY_KEY" ]; then
    echo "Error: Could not read city key for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Processing city: $CITY_KEY (Task ID: $SLURM_ARRAY_TASK_ID)"

# Run the analysis script
python analyze_city.py --city "$CITY_KEY"
