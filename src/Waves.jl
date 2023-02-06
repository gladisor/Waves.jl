module Waves

using DifferentialEquations
import GLMakie

abstract type AbstractDim end
abstract type Scatterer end

include("dims.jl")
include("cylinder.jl")
include("wave.jl")
include("pml.jl")
include("plot.jl")
include("initial_conditions.jl")
include("design.jl")

end