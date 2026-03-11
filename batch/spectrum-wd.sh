#!/bin/bash
#SBATCH --job-name=spectrum-wd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=6000M
#SBATCH --gres=gpu:1

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

python spectrum-wd.py

deactivate