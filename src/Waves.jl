module Waves

export 
    AbstractDim, 
    AbstractDesign, 
    AbstractSource,
    AbstractInitialWave, 
    AbstractInitialDesign, 
    AbstractDynamics,
    AbstractSensor

using SparseArrays
using IntervalSets
using Distributions: Uniform
using CairoMakie
using Interpolations
using Interpolations: Extrapolation

using Flux
using Flux: flatten, Recur, batch, unbatch, pullback, withgradient, trainable, mean, norm, mse, huber_loss, DataLoader

using ChainRulesCore
using Optimisers
using ReinforcementLearning
using ProgressMeter: @showprogress
using BSON
using FileIO

abstract type AbstractDim end

abstract type AbstractDesign end
abstract type AbstractScatterers <: AbstractDesign end

abstract type AbstractSource end

abstract type AbstractInitialWave end

abstract type AbstractDynamics end
abstract type AbstractSensor end

include("dims.jl")                          ## Core structures for defining dimensional spaces
include("metrics.jl")
include("operators.jl")
include("pml.jl")                           ## Perfectly Matched Layer

include("designs.jl")
# include("sources.jl")
# include("initial_wave.jl")                  ## Pulses, waves, etc...
# include("dynamics.jl")                      ## Defines the dynamics of the wave simulation
# include("env.jl")

# include("data.jl")
# include("models.jl")                        ## modeling
end