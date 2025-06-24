#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=hd189733-download
#SBATCH --time=10:00:00
#SBATCH --partition=general
#SBATCH -o download-%j.output
#SBATCH -e download-%j.error

cd ../data

srun bash hd189733-all.sh

# have to unpack recursive folders
cd MAST_2025-06-24T0210

find ./HST -mindepth 2 -type f -exec mv {} ./HST/ \;
find ./HST -mindepth 1 -type d -empty -delete