#!/bin/bash
#SBATCH --job-name hmc_single
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-0:00
#SBATCH -o single.%j.out  # STDOUT
#SBATCH -e single.%j.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

python hmc_single.py

conda deactivate