#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=cpu_il
conda activate py10
python test.py

# Run the Python script

# Deactivate the virtual environment
