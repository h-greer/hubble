#!/bin/bash --login
#SBATCH --job-name=pztel-binary
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8000M
#SBATCH -o pztel-binary.out


module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

# Default: n918bfpiq and n918c0poq (one 192 s exposure at each roll). Pass rootnames to fit others
python pztel-binary.py

deactivate
