#!/bin/bash --login
#SBATCH --job-name=calibrators
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8000M
#SBATCH -o calibrators/%a.out


module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

python calibrators.py $SLURM_ARRAY_TASK_ID

deactivate