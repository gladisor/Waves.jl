import GLMakie
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using DifferentialEquations: init
using Distributions: Uniform
using ReinforcementLearning
using IntervalSets
using Flux

using Waves
using Waves: AbstractDesign, AbstractDim

include("../src/env.jl")

"""
Renders an animation of a wave solution.
"""
function render!(sol::WaveSol, design_trajectory::Vector{<:AbstractDesign} = nothing; path::String)
    fig = plot(sol.dim)

    GLMakie.record(fig, path, 1:length(sol)) do i
        GLMakie.empty!(fig.content[1].scene)
        if !isnothing(design_trajectory)
            plot!(fig, design_trajectory[i])
        end
        wave = Wave(sol.dim, sol.u[i])
        plot!(fig, wave)
    end
end

"""
"""
function interpolate(sol::ODESolution, dim::AbstractDim, dt::Float64)
    u = typeof(sol.prob.u0)[]
    t = collect(sol.prob.tspan[1]:dt:sol.prob.tspan[end])

    for i ∈ axes(t, 1)
        push!(u, sol(t[i]))
    end

    return WaveSol(dim, t, u)
end

function interpolate(designs::Vector{DesignInterpolator}, dt::Float64)
    design_trajectory = typeof(first(designs).initial)[]

    for i ∈ axes(designs, 1)

        design = designs[i]
        t = collect(design.ti:dt:design.tf)

        for j ∈ axes(t, 1)
            push!(design_trajectory, designs[i](t[j]))
        end

        pop!(design_trajectory)
    end

    last_design = last(designs)

    push!(design_trajectory, last_design(last_design.tf))

    return design_trajectory
end

function Waves.plot(x::Vector, y::Vector; title = "", xlabel = "", ylabel = "")
    fig = GLMakie.Figure(resolution = Waves.RESOLUTION, fontsize = Waves.FONT_SIZE)
    ax = GLMakie.Axis(fig[1, 1], title = title, xlabel = xlabel, ylabel = ylabel)
    GLMakie.lines!(ax, x, y, color = :blue)
    return fig
end

gs = 10.0
Δ = 0.1
C0 = 2.0
pml_width = 4.0
pml_scale = 10.0
tspan = (0.0, 20.0)
dim = TwoDim(gs, Δ)
u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)

design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.2), Cylinder(0.0, 0.0, 0.0, 0.0), tspan...)
C = WaveSpeed(dim, C0, design)
pml = build_pml(dim, pml_width) * pml_scale

prob_tot = ODEProblem(split_wave!, u0, tspan, [Δ, C, pml])
prob_inc = ODEProblem(split_wave!, u0, tspan, [Δ, WaveSpeed(dim, C0), pml])

sol_inc = solve(prob_inc, Midpoint())
iter_tot = init(prob_tot, Midpoint(), advance_to_tstop = true)
reward_signal = ScatteredFlux(sol_inc, WaveFlux(dim, circle_mask(dim, 6.0)))

env = WaveEnv(iter_tot, C, 0.5, reward_signal)
designs = Vector{DesignInterpolator}()

actor = Chain(
    Conv((2, 2), 3 => 32, relu),
    MaxPool((2, 2)),
    Conv((2, 2), 32 => 1, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256, 2, tanh))

# traj = CircularArraySARTTrajectory(
#     capacity = 1000,
#     state = typeof(state(env)) => size(state(env)),
#     action = typeof(actor(Flux.batch([state(env)]))) => size(actor(Flux.batch([state(env)]))))

@time while !is_terminated(env)
    action = Cylinder(rand(action_space(env))..., 0.0, 0.0)
    # action = Cylinder(actor(Flux.batch([state(env)]))..., 0.0, 0.0) * 10.0
    env(action)
    println("Reward: $(reward(env))")

    println("Time: $(env.iter.t)")
    push!(designs, env.C.design)
end

dt_plot = 0.05
sol_tot = interpolate(env.iter.sol, dim, dt_plot)
sol_inc = interpolate(sol_inc, dim, dt_plot)

render!(sol_tot, interpolate(designs, dt_plot), path = "env_dx=$(Δ)_dt=$(dt_plot).mp4")
render!(sol_inc, interpolate(designs, dt_plot), path = "env_inc_dx=$(Δ)_dt=$(dt_plot).mp4")

sol_sc = sol_tot - sol_inc

f_tot = reward_signal.flux(sol_tot)
f_inc = reward_signal.flux(sol_inc)
f_sc = reward_signal.flux(sol_sc)

fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 40)
ax = GLMakie.Axis(fig[1, 1], title = "Flux With Single Random Scatterer", xlabel = "Time (s)", ylabel = "Flux")
GLMakie.lines!(ax, sol_inc.t, f_inc, linewidth = 3, color = :blue, label = "Incident")
GLMakie.lines!(ax, sol_sc.t, f_sc, linewidth = 3, color = :red, label = "Scattered")
GLMakie.axislegend(ax)
save(fig, "flux_dx=$(Δ)_dt=$(dt_plot).png")
