module Waves

using SparseArrays
using IntervalSets
using Distributions: Uniform
using CairoMakie
using Flux

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type Scatterer <: AbstractDesign end
abstract type RewardSignal end

include("dims.jl")                  ## Core structures for defining dimensional spaces
include("pml.jl")                   ## Perfectly Matched Layer
include("initial_conditions.jl")    ## Pulses, waves, etc...

include("cylinder.jl")              ## Simple circular scatterer
include("design.jl")                ## Interpolator for design
include("speed_field.jl")           ## Defines a dynamic wavespeed field C(t)
include("dynamics.jl")              ## Defines the dynamics of the wave simulation
include("env.jl")

include("sol.jl")                   ## Structure for holing results of wave simulations
include("design_trajectory.jl")     ## Structure for holding the path of designs
include("plot.jl")                  ## Plotting
end