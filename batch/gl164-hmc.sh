#!/bin/bash
#SBATCH --job-name=gl164-hmc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --gres=gpu:1

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

python gl164-hmc.py

deactivate