#!/bin/bash
#SBATCH --job-name dwarf_analyse
#SBATCH --array=0-67                    
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6:00
#SBATCH -o grid/%a.out  # STDOUT
#SBATCH -e grid/%a.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

srun python binary_grid_search.py $SLURM_ARRAY_TASK_ID

conda deactivate