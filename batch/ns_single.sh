#!/bin/bash
#SBATCH --job-name ns
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-0:00
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

python ns_test.py

conda deactivate