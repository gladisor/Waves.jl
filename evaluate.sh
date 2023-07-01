#!/bin/bash

#SBATCH -J eval

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -w cs002

#SBATCH -o logs/log-%j.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

module load julia
srun julia --project scripts/test.jl ## your run command