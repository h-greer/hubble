#!/bin/bash --login
#SBATCH --job-name=hd155915-binary
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8000M
#SBATCH -o hd155915-binary.out


module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

# Default: one exposure (n918b1e5q, roll 35.3 deg). Pass rootnames to fit more, e.g.
#   python hd155915-binary.py n918b1e5q n918b1e6q n918b1e7q n918b1e8q
python hd155915-binary.py

deactivate
