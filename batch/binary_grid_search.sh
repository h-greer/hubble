#!/bin/bash
#SBATCH --job-name binary_grid_search
#SBATCH --array=0-135                    
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --time=6:00:00
#SBATCH -o binary/%a.out  # STDOUT
#SBATCH -e binary/%a.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

srun python binary_grid_search.py $SLURM_ARRAY_TASK_ID

conda deactivate