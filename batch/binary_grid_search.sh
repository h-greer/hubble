#!/bin/bash
#SBATCH --job-name binary_grid_search
#SBATCH --array=0-135%5                    
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:10:00
#SBATCH -o binary/%a.out  # STDOUT
#SBATCH -e binary/%a.err  # STDERR
#SBATCH --gres=gpu:1

module load gcccore/13.2.0
module load python/3.11.5

source /fred/oz440/hayden/venv/bin/activate

python binary_grid_search.py

deactivate