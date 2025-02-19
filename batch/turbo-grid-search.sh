#!/bin/bash
#SBATCH --job-name turbo_grid_search
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-0:00
#SBATCH -o turbo.%j.out  # STDOUT
#SBATCH -e turbo.%j.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

srun python turbo-grid-search.py

conda deactivate