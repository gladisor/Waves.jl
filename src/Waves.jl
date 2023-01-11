module Waves

using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using ModelingToolkit.Symbolics: CallWithMetadata
using OrdinaryDiffEq: ODEIntegrator
using GLMakie, CairoMakie

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type InitialCondition end

include("dims.jl")
include("wave.jl")
include("initial_conditions.jl")
include("wave_sim.jl")
include("parameterized_design.jl")
include("cylinder.jl")

end # module
