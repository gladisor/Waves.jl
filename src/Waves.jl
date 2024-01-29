module Waves

export 
    AbstractDim, 
    AbstractDesign, 
    AbstractSource,
    AbstractDynamics

using SparseArrays
using IntervalSets
using Distributions: Uniform
using CairoMakie
using Interpolations: linear_interpolation, Extrapolation

using Flux
using Flux: 
    CUDA,
    flatten, 
    Recur, 
    batch, 
    unbatch, 
    pullback, 
    trainable, 
    mean, 
    norm, 
    mse, 
    DataLoader

using ChainRulesCore
using Optimisers
using ReinforcementLearning
using BSON
using FileIO
import LinearAlgebra
using Images: imresize

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type AbstractScatterers <: AbstractDesign end
abstract type AbstractSource end
abstract type AbstractDynamics end

include("utils.jl")
include("dims.jl")                          ## Core structures for defining dimensional spaces
include("operators.jl")
include("pml.jl")                           ## Perfectly Matched Layer

include("designs.jl")
include("sources.jl")
include("dynamics.jl")                      ## Defines the dynamics of the wave simulation
include("env.jl")

include("data.jl")
# include("model.jl")

include("model/layers.jl")
include("model/design_encoder.jl")
include("model/wave_encoder.jl")
include("model/acoustic_energy_model.jl")
include("model/pinn.jl")

include("plot.jl")
end