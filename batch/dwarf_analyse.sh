#!/bin/bash
#SBATCH --job-name dwarf_analyse
#SBATCH --array=1-136                    
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-0:00
#SBATCH -o dwarf/%a.out  # STDOUT
#SBATCH -e dwarf/%a.err  # STDERR

module load anaconda3/5.2.0
source /opt/modules/Anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate /data/uqhgreer/repos/jax-cpu

srun python dwarf_analyse.py $SLURM_ARRAY_TASK_ID

conda deactivate