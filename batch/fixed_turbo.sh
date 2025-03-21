#!/bin/bash
#SBATCH --job-name fixed_turbo
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1-0:00
#SBATCH -o turbo.%j.out  # STDOUT
#SBATCH -e turbo.%j.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

srun python fixed_turbo.py

conda deactivate