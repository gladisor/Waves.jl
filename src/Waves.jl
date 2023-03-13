module Waves

export AbstractDim, AbstractDesign, InitialCondition, AbstractWaveCell, design_space

using SparseArrays
using IntervalSets
using Distributions: Uniform
using CairoMakie
using Interpolations
using Flux
using ReinforcementLearning

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type Scatterer <: AbstractDesign end
abstract type InitialCondition end
abstract type AbstractWaveCell end

include("dims.jl")                         ## Core structures for defining dimensional spaces
include("metrics.jl")
include("ops.jl")
include("sol.jl")                          ## Structure for holing results of wave simulations
include("pml.jl")                          ## Perfectly Matched Layer
include("initial_conditions.jl")           ## Pulses, waves, etc...

include("design/cylinder.jl")              ## Simple circular scatterer
include("design/scatterers.jl")
include("design/design_interpolator.jl")   ## Interpolator for design
include("design/design_trajectory.jl")     ## Structure for holding the sequence of designs

include("dynamics.jl")                     ## Defines the dynamics of the wave simulation
include("update_equations.jl")
include("wave_cell.jl")
include("env.jl")

include("data.jl")

## modeling
include("models/blocks.jl")
# include("models/wave_encoder.jl")
# include("models/wave_cnn_decoder.jl")

include("reinforcement_learning/hooks.jl")
include("reinforcement_learning/random_policy.jl")

include("plot.jl")                         ## Plotting
end