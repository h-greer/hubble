#!/bin/bash --login
#SBATCH --job-name=gmaur-wpca-trace
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --tmp=20G
#SBATCH -o gmaur-wpca-trace.out


module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

SCRATCH=/aphid/scratch-3month/$USER
export PROFILE_DIR=$SCRATCH/profiles/gmaur-wpca-$SLURM_JOB_ID
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cache
mkdir -p $PROFILE_DIR $JAX_COMPILATION_CACHE_DIR

# XLA dumps are many small files: local SSD during the job, archived to scratch afterwards
export XLA_FLAGS="--xla_dump_to=$JOBFS/xla_dump --xla_dump_hlo_as_text"

export EQX_ON_ERROR=nan   # equinox runtime checks return NaN instead of raising (cheaper tracing)

python gmaur-wpca-trace.py

tar -czf $PROFILE_DIR/xla_dump.tar.gz -C $JOBFS xla_dump

deactivate
