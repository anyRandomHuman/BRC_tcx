#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=output_%j.txt
#SBATCH --time=00:00:05
#SBATCH --partition=cpu

echo "Running job with:"
echo "ARG1=$1"
echo "ARG2=$2"
echo "EXTRA FLAGS: $@"