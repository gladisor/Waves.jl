println("Importing Packages")
using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves

Flux.CUDA.allowscalar(false)

include("improved_model.jl")
include("plot.jl")

Flux.device!(1)

main_path = "/scratch/cmpe299-fa22/tristan/data/design_space=build_triple_ring_design_space_freq=1000.0"
data_path = joinpath(main_path, "episodes")
