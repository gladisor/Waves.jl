module Waves

using Distributions: Uniform
using SparseArrays
using DifferentialEquations
using CUDA
# import GLMakie

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type Scatterer <: AbstractDesign end
abstract type Emitter <: AbstractDesign end

include("dims.jl")
include("cylinder.jl")
# include("configuration.jl")
include("pml.jl")
include("initial_conditions.jl")
include("design.jl")
include("sim.jl")
include("speed_field.jl")


# include("wave_speed.jl")
include("sol.jl")
include("metrics.jl")
# include("plot.jl")

end