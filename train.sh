#!/bin/bash

#SBATCH -J mpc

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4-00:00:00
#SBATCH -w cs002

#SBATCH -o logs/log-%j.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128
module load julia
srun julia --project scripts/main.jl