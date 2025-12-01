#!/bin/bash
#SBATCH --job-name=urban-complexity-visual
#SBATCH --account=sstoev0
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-499%50
#SBATCH --output=/scratch/sstoev_root/sstoev0/ksstewar/uca_visual_%x_%A_%a.out
#SBATCH --error=/scratch/sstoev_root/sstoev0/ksstewar/uca_visual_%x_%A_%a.err

# --- 1. Environment Setup ---
echo "Job started on $(date). Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Setting up the environment..."

# Change to scratch directory
cd /scratch/sstoev_root/sstoev0/${USER_NAME:-${USER}} || exit 1

REPO_DIR="urban-complexity-analysis"
LOCK_FILE="${REPO_DIR}/.setup_complete"

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    echo "Task 0: Setting up shared environment..."
    rm -rf "${REPO_DIR}"
    
    module purge
    module load python

    git clone https://github.com/kennonstewart/urban-complexity-analysis.git
    cd "${REPO_DIR}" || exit 1
    
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    touch "${LOCK_FILE}"
    echo "Setup complete!"
else
    echo "Waiting for task 0 to complete setup..."
    for i in {1..120}; do
        if [ -f "${REPO_DIR}/${LOCK_FILE}" ]; then
            break
        fi
        sleep 5
    done
    
    if [ ! -f "${REPO_DIR}/${LOCK_FILE}" ]; then
        echo "ERROR: Setup did not complete in time"
        exit 1
    fi
    
    module purge
    module load python
    cd "${REPO_DIR}" || exit 1
    source .venv/bin/activate
fi

# --- 2. Distributed Visual Generation ---
CITY_KEY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/manifest.txt)

if [ -z "$CITY_KEY" ]; then
    echo "Error: Could not read city key for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Generating visuals for city: $CITY_KEY (Task ID: $SLURM_ARRAY_TASK_ID)"

python distributed_jobs/generate_visuals.py --city "$CITY_KEY"

echo "Task ${SLURM_ARRAY_TASK_ID} finished!"
