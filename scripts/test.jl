using GLMakie
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using ReinforcementLearning

using Waves
using Waves: AbstractDesign

include("../src/env.jl")

gs = 10.0
dx = 0.05
dt = 0.03
ambient_speed = 2.0
pml_width = 4.0
pml_scale = 20.0
tspan = (0.0, 20.0)

dim = TwoDim(gs, dx)
g = grid(dim)
u = pulse(dim)
u0 = cat(u, zeros(size(u)..., 2), dims = 3)

grad = gradient(dim.x)
design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.2), Cylinder(0.0, 0.0, 0.0, 0.0), tspan...)
C_inc = SpeedField(dim, ambient_speed)
C = SpeedField(dim, ambient_speed, design)

pml = build_pml(dim, pml_width, pml_scale)

prob_inc = ODEProblem(wave!, u0, tspan, [grad, C_inc, pml])
prob = ODEProblem(wave!, u0, tspan, [grad, C, pml])

@time sol_inc = solve(prob_inc, DGLDDRK73_C(), dt = dt)
reward_signal = ScatteredFlux(sol_inc, WaveFlux(dim, circle_mask(dim, 6.0)))

iter = init(prob, DGLDDRK73_C(),  advance_to_tstop = true,  dt = dt)

env = WaveEnv(iter, C, 0.5, reward_signal)
designs = Vector{DesignInterpolator}()

@time while !is_terminated(env)
    action = Cylinder(rand(action_space(env))..., 0.0, 0.0)
    env(action)
    println("Time: $(env.iter.t)")
    push!(designs, env.C.design)
end

# sol_inc_interp = Waves.interpolate(sol_inc, dim, 0.1)
# sol_tot_interp = Waves.interpolate(env.iter.sol, dim, 0.1)

# sol_sc = sol_tot_interp - sol_inc_interp

# flux_inc = reward_signal.flux(sol_inc_interp)
# flux_sc = reward_signal.flux(sol_sc)

# fig = Figure(resolution = (1920, 1080), fontsize = 40)
# ax = Axis(fig[1, 1], title = "Acoustic Flux", xlabel = "Time (s)", ylabel = "Flux")
# lines!(ax, sol_inc_interp.t, flux_inc, label = "Incident", color = :blue, linewidth = 3)
# lines!(ax, sol_sc.t, flux_sc, label = "Scattered", color = :red, linewidth = 3)
# axislegend(ax)
# save("flux.png", fig)

# design_interp = Waves.interpolate(designs, 0.1)
# Waves.render!(sol_tot_interp, design_interp, path = "sol_tot.mp4")
# Waves.render!(sol_inc_interp, path = "sol_inc.mp4")
# Waves.render!(sol_sc, design_interp, path = "sol_sc.mp4")