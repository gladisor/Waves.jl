module Waves

using Distributions: Uniform
using SparseArrays
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using CairoMakie
import Flux
using ReinforcementLearning


abstract type AbstractDim end
abstract type AbstractDesign end
abstract type Scatterer <: AbstractDesign end
abstract type Emitter <: AbstractDesign end

include("dims.jl")
include("cylinder.jl")

include("pml.jl")
include("initial_conditions.jl")
include("design.jl")
include("sim.jl")
include("speed_field.jl")

include("sol.jl")
include("metrics.jl")
include("env.jl")
include("plot.jl")
end