import GLMakie
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using DifferentialEquations: init
using Distributions: Uniform
using IntervalSets
using ReinforcementLearning

using SparseArrays
using LinearAlgebra

using Waves
using Waves: AbstractDesign

using CUDA

include("../src/env.jl")

function Waves.OneDim(grid_size::Float64, n::Int)
    return OneDim(collect(range(-grid_size, grid_size, n)))
end

function Waves.TwoDim(grid_size::Float64, n::Int)
    return TwoDim(
        collect(range(-grid_size, grid_size, n)),
        collect(range(-grid_size, grid_size, n)))
end

"""
Function for constructing a gradient operator for a one dimensional scalar field.
"""
function gradient(x::Vector)
    grad = zeros(size(x, 1), size(x, 1))
    Δ = (x[end] - x[1]) / (length(x) - 1)

    grad[[1, 2, 3], 1] .= [-3.0, 4.0, -1.0] ## left boundary edge case
    grad[[end-2, end-1, end], end] .= [1.0, -4.0, 3.0] ## right boundary edge case

    for i ∈ 2:(size(grad, 2) - 1)
        grad[[i - 1, i + 1], i] .= [-1.0, 1.0]
    end

    return sparse((grad / (2 * Δ))')
end

function pulse(dim::OneDim, x = 0.0, intensity = 1.0)
    return exp.(- intensity * (dim.x .- x) .^ 2)
end

function pulse(dim::TwoDim, x = 0.0, y = 0.0, intensity = 1.0)
    u = zeros(length(dim.x), length(dim.y))

    for i ∈ axes(u, 1)
        for j ∈ axes(u, 2)
            u[i, j] = exp(- intensity * ((dim.x[i] - x) .^ 2 + (dim.y[j] - y) .^ 2))
        end
    end

    return u
end

function wave!(du::Array{Float64, 3}, u::Array{Float64, 3}, p, t::Float64)
    grad, C, pml = p

    U = u[:, :, 1]
    Vx = u[:, :, 2]
    Vy = u[:, :, 3]

    du[:, :, 1] .= C(t) .* (grad * Vx .+ (grad * Vy')') .- U .* pml
    du[:, :, 2] .= (grad * U) .- Vx .* pml
    du[:, :, 3] .= (grad * U')' .- Vy .* pml
end

gs = 10.0
Δ = 0.05
C0 = 2.0
pml_width = 4.0
pml_scale = 10.0
tspan = (0.0, 20.0)
dim = TwoDim(gs, Δ)
# u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)
grad = gradient(dim.x) |> cu
u = pulse(dim)
u0 = cat(u, zeros(size(u)..., 2), dims = 3) |> cu

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
    ORK256(), 
    advance_to_tstop = true, 
    dt = dt)

reward_signal = ScatteredFlux(sol_inc, WaveFlux(dim, circle_mask(dim, 6.0)))

env = WaveEnv(iter_tot, C, 0.5, reward_signal)
designs = Vector{DesignInterpolator}()

@time while !is_terminated(env)
    action = Cylinder(rand(action_space(env))..., 0.0, 0.0)
    env(action)
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
