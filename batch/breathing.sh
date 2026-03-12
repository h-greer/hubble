#!/bin/bash --login
#SBATCH --job-name=breathing-mode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --array=0-2

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

python breathing.py $SLURM_ARRAY_TASK_ID

deactivate