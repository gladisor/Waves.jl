module Waves

export AbstractDim, AbstractDesign, InitialCondition

using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using ModelingToolkit.Symbolics: CallWithMetadata
using OrdinaryDiffEq: ODEIntegrator, ODESolution
import GLMakie
using Distributions: Uniform

abstract type AbstractDim end
abstract type AbstractDesign end
abstract type InitialCondition end
abstract type WaveBoundary end

include("dims.jl")
include("wave.jl")
include("initial_conditions.jl")
include("boundary.jl")
include("parameterized_design.jl")
include("cylinder.jl")
# include("configuration.jl")
include("sim.jl")
include("sol.jl")
include("env.jl")

end # module
