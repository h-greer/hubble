#!/bin/bash
#SBATCH --job-name=imlup-adi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:20:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --gres=gpu:1

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

python imlup-adi.py

deactivate