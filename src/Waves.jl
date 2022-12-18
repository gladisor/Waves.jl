module Waves

export animate!

using Plots
using BSON
using ModelingToolkit
using MethodOfLines
using ModelingToolkit.Symbolics: CallWithMetadata
using OrdinaryDiffEq
using ProgressMeter

abstract type AbstractWave end
abstract type InitialCondition end

include("wave1d.jl")
include("wave2d.jl")
include("wave_simulation.jl")
include("data.jl")

end # module
