#!/bin/bash --login
#SBATCH --job-name=betapic-f207m-wpca
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=8000M
#SBATCH -o betapic-f207m-wpca.out


module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

export JAX_COMPILATION_CACHE_DIR=/aphid/scratch-3month/$USER/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR
export EQX_ON_ERROR=nan   # equinox runtime checks return NaN instead of raising (cheaper tracing)

python betapic-f207m-wpca.py

deactivate
