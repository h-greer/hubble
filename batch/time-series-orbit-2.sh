#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=time-series-orbit
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH -o timeseries-orbit/orbit-%a.out
#SBATCH -e timeseries-orbit/orbit-%a.err

module load anaconda3/2024.02-1
conda activate /data/uqhgreer/repos/jax-cpu

srun python time-series-orbit.py $SLURM_ARRAY_TASK_ID

conda deactivate