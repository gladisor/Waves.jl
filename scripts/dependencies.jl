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
using Waves
using Waves: speed

abstract type AbstractWaveControlModel end
abstract type AbstractSource end

include("../src/sources.jl")
include("../src/dynamics.jl")
include("env.jl")
include("wave_control_model.jl")
include("plot.jl")