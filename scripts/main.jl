import GLMakie
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using DifferentialEquations: init
using Distributions: Uniform
using IntervalSets
using ReinforcementLearning

using Waves
using Waves: AbstractDesign

include("../src/env.jl")

gs = 10.0
Δ = 0.05
C0 = 2.0
pml_width = 4.0
pml_scale = 10.0
tspan = (0.0, 20.0)
dim = TwoDim(gs, Δ)
# u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)
grad = gradient(dim.x)
u = pulse(dim)
u0 = cat(u, zeros(size(u)..., 2), dims = 3)

design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.2), Cylinder(0.0, 0.0, 0.0, 0.0), tspan...)
C = WaveSpeed(dim, C0, design)
pml = build_pml(dim, pml_width) * pml_scale

prob_mat = ODEProblem(wave!, u0, (0.0, 10.0), [grad, C, pml])
# prob_tot = ODEProblem(split_wave!, u0, tspan, [Δ, C, pml])
# prob_inc = ODEProblem(split_wave!, u0, tspan, [Δ, WaveSpeed(dim, C0), pml])

dt = 0.05
sol_inc = solve(prob_inc, ORK256(), dt = dt)
iter_tot = init(
    # prob_tot, 
    prob_mat,
    ORK256(), advance_to_tstop = true, dt = dt)

reward_signal = ScatteredFlux(sol_inc, WaveFlux(dim, circle_mask(dim, 6.0)))

env = WaveEnv(iter_tot, C, 0.5, reward_signal)
designs = Vector{DesignInterpolator}()

@time while !is_terminated(env)
    action = Cylinder(rand(action_space(env))..., 0.0, 0.0)
    env(action)
    # println("Reward: $(reward(env))")

    println("Time: $(env.iter.t)")
    push!(designs, env.C.design)
end

dt_plot = 0.05
sol_tot = Waves.interpolate(env.iter.sol, dim, dt_plot)
# sol_inc = Waves.interpolate(sol_inc, dim, dt_plot)

Waves.render!(sol_tot, Waves.interpolate(designs, dt_plot), path = "env_dx=$(Δ)_dt=$(dt_plot).mp4")

# sol_sc = sol_tot - sol_inc

# f_tot = reward_signal.flux(sol_tot)
# f_inc = reward_signal.flux(sol_inc)
# f_sc = reward_signal.flux(sol_sc)

# fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 40)
# ax = GLMakie.Axis(fig[1, 1], title = "Flux With Single Random Scatterer", xlabel = "Time (s)", ylabel = "Flux")
# GLMakie.lines!(ax, sol_inc.t, f_inc, linewidth = 3, color = :blue, label = "Incident")
# GLMakie.lines!(ax, sol_sc.t, f_sc, linewidth = 3, color = :red, label = "Scattered")
# GLMakie.axislegend(ax)
# Waves.save(fig, "flux_dx=$(Δ)_dt=$(dt_plot).png")
