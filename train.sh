#!/bin/bash

#SBATCH -J mpc

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
# SBATCH -t 4-00:00:00
#SBATCH -w cs003

#SBATCH -o log-%j.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

module load julia
srun julia --project scripts/main.jl ## your run command