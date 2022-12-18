using Plots
ENV["GKSwstype"] = "nul"
pyplot()

using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using IfElse

using Waves
using Waves: boundary_conditions, wave_equation, WaveSolution2D

wave = Wave2D(
    x_min = -10.0, x_max = 10.0,
    y_min = -10.0, y_max = 10.0,
    t_max = 1.0)

# struct Cylinder
#     x_pos::Num
#     y_pos::Num
#     r::Num
#     wavespeed::Num
# end

# function Cylinder(wave, x_pos, y_pos)
#     @parameters x_pos, y_pos, r, wavespeed

#     IfElse((wave.x - x_pos) ^ 2 + (wave.y - y_pos) ^ 2 < r, wavespeed, )
#     r, wavespeed
# end

# function wave_speed(cyl::Cylinder)
# end

@parameters speed
@parameters x_pos, y_pos, r, cyl_speed

wavespeed(x, y, t) = IfElse.ifelse((x - x_pos) ^ 2 + (y - y_pos) ^ 2 < r, cyl_speed, speed)

sim = WaveSimulation(
    wave,
    ic = (x, y) -> exp(- 1 * ((x - 0.25)^2 + (y - 2.5)^2)),
    # C = (x, y, t) -> speed,
    C = wavespeed,
    n = 30,
    params = [
        speed => 2.0, 
        x_pos => -0.25, 
        y_pos => 2.5, 
        r => 0.1, 
        cyl_speed => 0.0]
    )

sim.prob = remake(sim.prob, tspan = (0.0, 5.0), p = [1.0])

@time sol = ∫ₜ(sim, dt = 0.05)
animate!(sol, "vid_large.mp4")