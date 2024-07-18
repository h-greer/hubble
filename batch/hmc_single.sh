#!/bin/bash
#SBATCH --job-name hmc_single
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-0:00
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR

module load anaconda3/5.2.0
conda activate /data/uqhgreer/repos/jax-cpu

python hmc_single.py

conda deactivate