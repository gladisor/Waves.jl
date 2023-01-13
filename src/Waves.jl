module Waves

export AbstractDim, AbstractDesign, InitialCondition

using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using ModelingToolkit.Symbolics: CallWithMetadata
using OrdinaryDiffEq: ODEIntegrator, ODESolution
import GLMakie
# using GLMakie, CairoMakie

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type InitialCondition end

include("dims.jl")
include("wave.jl")
include("initial_conditions.jl")
include("parameterized_design.jl")
include("cylinder.jl")
include("wave_sim.jl")
include("wave_env.jl")

end # module
