#!/bin/bash
#SBATCH --job-name=raytrace_binary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=6000M
#SBATCH --gres=gpu:1

module load gcccore/13.2.0
module load python/3.11.5

source /fred/oz440/hayden/venv/bin/activate

python raytrace-test.py

deactivate