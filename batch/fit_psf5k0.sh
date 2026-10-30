#!/bin/bash --login
#SBATCH --job-name=psf5k0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=8000M
#SBATCH -o psf5k0-%j.out

# Time: estimated, not measured. calibrator-joint (3 exposures, 2500 steps + polish) takes ~31 min; here 2 exposures,
# 2500 + 1000 steps (the linear-chromatic s3/s4 ~1.5-3x per step), 6 compiles (~3-5 min each) and an exact-chromatic
# polish: ~1.5-2 h. If it times out, resubmit: the fit resumes after the last saved stage.

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

export JAX_COMPILATION_CACHE_DIR=/aphid/scratch-3month/$USER/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR
export EQX_ON_ERROR=nan   # equinox runtime checks return NaN instead of raising (cheaper tracing)

unset FIT_DRY_RUN FIT_STOP_AFTER FIT_INJECT_NAN   # local test hooks only (SLURM exports the submitting shell's env)
python fit_psf5k0.py

deactivate
