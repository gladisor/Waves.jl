using BSON
using BSON: @save
using Flux
using Flux: Recur, batch, unbatch, pullback, norm, mse, huber_loss, mean, flatten, DataLoader
Flux.CUDA.allowscalar(false)
using Interpolations
using Interpolations: Extrapolation
using ReinforcementLearning
using IntervalSets
using CairoMakie
using Optimisers
using ChainRulesCore
using ProgressMeter
using FileIO

using Distributions: Uniform
using Waves

include("../src/designs.jl")
# include("wave_control_model.jl")
# include("plot.jl")