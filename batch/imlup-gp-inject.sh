#!/bin/bash --login
#SBATCH --job-name=imlup-gp-inject
#SBATCH --array=0-23
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8000M
#SBATCH -o imlup-gp-inject/%a.out

# 3 truths x 8 seeds (task = truth * 8 + seed). Needs imlup-gp/response.npz, written by imlup-gp.py stage B (its log
# says when; imlup-gp.py also creates imlup-gp-inject/, which SLURM needs for these logs), e.g.
#   mkdir -p imlup-gp-inject && sbatch --dependency=afterany:<imlup-gp job id> imlup-gp-inject.sh
# then `python imlup-gp-summary.py` once all tasks have finished.

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

export JAX_COMPILATION_CACHE_DIR=/aphid/scratch-3month/$USER/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR
export EQX_ON_ERROR=nan   # equinox runtime checks return NaN instead of raising (cheaper tracing)

python imlup-gp-inject.py $SLURM_ARRAY_TASK_ID

deactivate
