module Waves

using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using ModelingToolkit.Symbolics: CallWithMetadata
using GLMakie

abstract type AbstractDim end

include("dims.jl")
include("wave.jl")

end # module
