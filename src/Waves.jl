module Waves

using DifferentialEquations
import GLMakie
using Distributions: Uniform

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type Scatterer <: AbstractDesign end
abstract type Emitter <: AbstractDesign end

include("dims.jl")
include("cylinder.jl")
include("configuration.jl")
include("pml.jl")
include("initial_conditions.jl")
include("design.jl")
include("sim.jl")
include("wave_speed.jl")
include("sol.jl")
include("metrics.jl")
include("plot.jl")

end