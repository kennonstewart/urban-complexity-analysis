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
