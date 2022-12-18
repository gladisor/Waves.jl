# using Plots
# ENV["GKSwstype"] = "nul"
# pyplot()

using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using IfElse

using Waves
# using Waves: boundary_conditions, wave_equation, WaveSolution2D

# mutable struct Cylinder
#     x::Pair{Num, Real}
#     y::Pair{Num, Real}
#     r::Pair{Num, Real}
#     c::Pair{Num, Real}
# end

# function Cylinder(x_pos::Real, y_pos::Real, radius::Real, wavespeed::Real)
#     @parameters x, y, r, c
#     return Cylinder(
#         x => x_pos, 
#         y => y_pos, 
#         r => radius, 
#         c => wavespeed)
# end

@parameters speed
@parameters x_pos, y_pos, r, cyl_speed

wave = Wave2D(
    x_min = -10.0, x_max = 10.0,
    y_min = -10.0, y_max = 10.0,
    t_max = 1.0)

wavespeed(x, y, t) = IfElse.ifelse((x - x_pos) ^ 2 + (y - y_pos) ^ 2 < r, cyl_speed, speed)

sim = WaveSimulation(
    wave,
    ic = (x, y) -> exp(- 0.5 * ((x - 2.5)^2 + (y - 0.0)^2)),
    C = wavespeed,
    n = 30,
    p = [speed => 2.0, x_pos => -0.25, y_pos => 2.5, r => 0.1, cyl_speed => 0.0])

sim.prob = remake(sim.prob, tspan = (0.0, 10.0), p = [1.0, -0.5, 0.0, 2.0, 0.1])
@time sol = ∫ₜ(sim, dt = 0.05)
# animate!(sol, "vid_large.mp4")