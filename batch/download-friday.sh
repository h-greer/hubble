#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=hd189733-download
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH -o download-%j.output
#SBATCH -e download-%j.error


srun bash ../data/hd189733-all.sh