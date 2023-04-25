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
using Waves
using Waves: speed

abstract type AbstractWaveControlModel end

include("../src/dynamics.jl")
include("env.jl")
include("wave_control_model.jl")
include("plot.jl")

grid_size = 8.0f0
elements = 256
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100

h_size = 128
latent_elements = 512

random_design_kwargs = Dict(:width => 1, :hight => 2, :spacing => 3.0f0, :r => Waves.MAX_RADII, :c => 2100.0f0, :center => [0.0f0, 0.0f0])