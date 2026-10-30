#!/bin/bash --login
#SBATCH --job-name=imlup-gp-e2e
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH -o imlup-gp-e2e-v2.out

# End-to-end GP fit of IM Lup (calibrator optics -> stage A -> GP stages); needs NO imlup-adi/ output. Needs ../data/calibrator_optical_params.npy.
# Resumable: rerunning skips every stage whose output exists in imlup-gp-e2e-v2/ (fingerprinted; the first run's imlup-gp-e2e/ is never reused).
# v2: stage A on the full image (occulter params held), stage A3 (disc introduced with the GP prior at fixed hyperparameters), safer refits (retry at lower
# Adam rates), stop if a noise scale s > 10. A3 adds ~10-20 min (one joint refit incl. its compiles).
# Resources vs imlup-gp-radial.sh (12 h, 4 x 8000M; the m12 run used 50 min, 6.5 GB peak): the run adds stage A (~15-30 min), a response build at the
# stage-A optics (~4-6 min), 1-3 more outer iterations (~10-20 min each: refits start further from the optimum) => ~1.5-3 h typical, <= ~4 h worst;
# peak RAM ~7-9 GB (the radial run's plus the star-only stage-A compile / second linear response). Hence 6 h and 16 GB.
#
# Validation (reproduce imlup-gp-radial.py from the L1/TSV fit; needs imlup-adi/ and imlup-gp/response.npz), a separate job:
#   E2E_OPTICS=l1tsv E2E_LEVEL=l1tsv python imlup-gp-e2e.py        # writes imlup-gp-e2e-v2-optics-l1tsv-level-l1tsv/ (no stage A / A3)
# Variants (output dirs are named by the flags): E2E_SUPPORT_STAR=header, E2E_TILT_INIT=adi, E2E_LEVEL_SD=1.5, E2E_LEVEL_MU=<dex>,
# E2E_STAGEA_MASK=1 (+ E2E_MASK_RIN / E2E_MASK_ROUT, arcsec) the first run's annulus-masked stage A,
# E2E_STAGEA_FREEZE=occulter_radius,occulter_ellipticity (comma list, "" = none), E2E_A3=0 (skip A3), E2E_A3_SIGMA / E2E_A3_LENGTH, E2E_S_MAX.

module load python-scientific/3.13.1-foss-2025a

source /fred/oz440/hayden/new-hubble/.venv/bin/activate

export JAX_COMPILATION_CACHE_DIR=/aphid/scratch-3month/$USER/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR
export EQX_ON_ERROR=nan   # equinox runtime checks return NaN instead of raising (cheaper tracing)

python imlup-gp-e2e.py

deactivate
