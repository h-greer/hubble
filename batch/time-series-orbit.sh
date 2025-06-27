#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=time-series-orbit
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH -o timeseries/orbit-%a.out
#SBATCH -e timeseries/orbit-%a.err

module load anaconda3/2024.02-1
conda activate /data/uqhgreer/repos/jax-cpu

srun python time-series-analyse.py 1

conda deactivate