#!/bin/bash

#SBATCH -J eval

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -w cs003

#SBATCH -o logs/log-%j.out
#SBATCH --mail-user=...

module load julia
srun julia --project scripts/mpc.jl ## your run command