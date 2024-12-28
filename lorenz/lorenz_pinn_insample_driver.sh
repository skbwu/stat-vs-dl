#!/bin/bash
#SBATCH -J LPI # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p (...) # Partition
#SBATCH --mem 8000 # Memory request
#SBATCH -t 0-08:00 # (D-HH:MM)
#SBATCH -o (...)/lorenz/outputs/%A_%a.out # Standard output
#SBATCH -e (...)/lorenz/errors/%A_%a.err # Standard error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=(...)
#SBATCH --account=(...)
#SBATCH --array=0-999

conda run -n afterburner python3 lorenz_pinn_insample.py ${SLURM_ARRAY_TASK_ID}
