#!/bin/bash
#SBATCH --job-name binary_grid
#SBATCH --array=1-20                    
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-0:00
#SBATCH -o out/binary.%a.out  # STDOUT
#SBATCH -e out/binary.%a.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

srun python binary-grid.py $SLURM_ARRAY_TASK_ID

conda deactivate