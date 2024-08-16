#!/bin/bash

#SBATCH -J mpc

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 21-00
#SBATCH -w cs004

#SBATCH -o logs/log-%j.out
#SBATCH --mail-user=noam.smilovich@sjsu.edu

module load julia
srun julia --project scripts/mpc.jl
