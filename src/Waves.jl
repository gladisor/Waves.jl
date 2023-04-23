module Waves

export AbstractDim, AbstractDesign, AbstractInitialCondition, AbstractDynamics, AbstractWaveCell

using SparseArrays
using IntervalSets
using Distributions: Uniform
using CairoMakie
using Interpolations
using Interpolations: Extrapolation

using Flux
using Flux: 
    flatten, Recur, 
    batch, unbatch, 
    pullback, withgradient, mean, 
    Params, trainable
    
using ChainRulesCore
using Optimisers
using ReinforcementLearning
using JLD2
using ProgressMeter: @showprogress

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type AbstractInitialCondition end
abstract type AbstractDynamics end
abstract type AbstractWaveCell end

include("dims.jl")                         ## Core structures for defining dimensional spaces
include("metrics.jl")
include("operators.jl")
include("pml.jl")                          ## Perfectly Matched Layer
include("initial_conditions.jl")           ## Pulses, waves, etc...

include("designs.jl")
# include("dynamics.jl")                     ## Defines the dynamics of the wave simulation

## modeling
# include("data.jl")
include("models/blocks.jl")
include("models/wave_encoder.jl")
include("models/wave_decoder.jl")
include("models/design_encoder.jl")

# include("plot.jl")                         ## Plotting
end