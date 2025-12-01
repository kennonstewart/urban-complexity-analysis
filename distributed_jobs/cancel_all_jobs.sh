#!/bin/bash
# cancel_all_jobs.sh
# Cancels all running and pending Slurm jobs for the current user

USER_NAME=${USER}

# Get all job IDs for the current user
JOB_IDS=$(squeue -u "$USER_NAME" -h -o "%A")

if [ -z "$JOB_IDS" ]; then
    echo "No running or pending jobs found for user $USER_NAME."
    exit 0
fi

for JOB_ID in $JOB_IDS; do
    echo "Cancelling job $JOB_ID..."
    scancel "$JOB_ID"
done

echo "All jobs for $USER_NAME have been cancelled."
