#!/bin/bash

#SBATCH -J nonlinear_latent

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00

#SBATCH -o log-%j.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

module load julia
srun julia --project scripts/main.jl ## your run command