#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-56%10
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=time-series
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH -o timeseries/run-%j.out
#SBATCH -e timeseries/run-%j.err

module load anaconda3/2024.02-1
conda activate /data/uqhgreer/repos/jax-cpu

srun python time-series-analyse.py $SLURM_ARRAY_TASK_ID

conda deactivate