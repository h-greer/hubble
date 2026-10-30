#!/bin/bash --login
#SBATCH --job-name=imlup-gp-e2e
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH -o imlup-gp-e2e.out

# Needs ../data/calibrator_optical_params.npy (not imlup-adi/). Resumable: rerunning skips every stage whose output exists in imlup-gp-e2e/.

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

export JAX_COMPILATION_CACHE_DIR=/aphid/scratch-3month/$USER/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR
export EQX_ON_ERROR=nan   # equinox runtime checks return NaN instead of raising (cheaper tracing)

python imlup-gp-e2e.py

deactivate
