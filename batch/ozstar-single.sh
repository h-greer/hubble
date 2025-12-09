#!/bin/bash
#SBATCH --job-name=single-star
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --gres=gpu:1

module load gcccore/13.2.0
module load python/3.11.5

source /fred/oz440/hayden/venv/bin/activate

python ozstar-single.py

deactivate