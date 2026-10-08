#!/bin/bash --login
#SBATCH --job-name=calibrator-bootstrap
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4000M
#SBATCH -o calibrator-bootstrap.out


module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

python calibrator-bootstrap.py

deactivate
