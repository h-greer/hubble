#!/bin/bash
#SBATCH --job-name hmc_ir
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-0:00
#SBATCH -o ir.%N.out  # STDOUT
#SBATCH -e ir.%N.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

python hmc_injection_recovery.py

conda deactivate