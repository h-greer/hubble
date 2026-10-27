#!/bin/bash --login
#SBATCH --job-name=calibrator-joint-F110W
#SBATCH --array=0-155
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8000M
#SBATCH -o calibrator-joint-F110W/%a.out


module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

python calibrator-joint-F110W.py $SLURM_ARRAY_TASK_ID

deactivate
