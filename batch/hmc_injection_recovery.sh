#!/bin/bash
#SBATCH --job-name hmc_ir
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6G
#SBATCH --time=1-0:00
#SBATCH -o ir.%j.out  # STDOUT
#SBATCH -e ir.%j.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

python hmc_injection_recovery.py

conda deactivate