#!/bin/bash
#SBATCH --job-name ns
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-0:00
#SBATCH -o ns.%j.out  # STDOUT
#SBATCH -e ns.%j.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

python ns_injection_recovery.py

conda deactivate