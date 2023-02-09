import GLMakie
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using DifferentialEquations: init
using Distributions: Uniform

using Waves
using Waves: AbstractDim, ∇, ∇x, ∇y

# struct WaveSim{D <: AbstractDim} 
#     dim::D
#     iter::ODEIntegrator
# end

struct WaveSol{D <: AbstractDim}
    dim::D
    t::Vector{Float64}
    u::Vector{<: AbstractArray{Float64}}
end

function Base.length(sol::WaveSol)
    return length(sol.t)
end

# """
# Extracting a solution from the simulator requires specification of a uniform timestep.
# The way the integration process works is that the timesteps are determined based on error.
# Interpolation is required to get uniform timesteps.
# """
# function WaveSol(sim::WaveSim; dt)
#     ti, tf = sim.iter.sol.prob.tspan
#     t = collect(ti:dt:tf)

#     u = Vector{typeof(sim.iter.u)}()

#     for i ∈ axes(t, 1)
#         push!(u, sim.iter.sol(t[i]))
#     end

#     return WaveSol(dim, t, u)
# end

function render!(sol::WaveSol; path::String)
    fig = plot(sol.dim)

    GLMakie.record(fig, path, 1:length(sol)) do i
        GLMakie.empty!(fig.content[1].scene)
        wave = Wave(sol.dim, sol.u[i])
        plot!(fig, wave)
    end
end

abstract type WaveMetric end

struct WaveFlux <: WaveMetric
    Δ::Float64
    mask::AbstractArray{Float64}
end

function WaveFlux(dim::AbstractDim, mask = nothing)
    Δ = (dim.x[end] - dim.x[1]) / (length(dim.x) - 1)
    mask = isnothing(mask) ? ones(size(dim)) : mask
    return WaveFlux(Δ, mask)
end

"""
Compute flux on the state of a one dimensional wave simulation.
"""
function (metric::WaveFlux)(u::Vector{Float64})
    return sum(∇(∇(u, metric.Δ), metric.Δ) .* metric.mask)
end

function (metric::WaveFlux)(u::Matrix{Float64})
    d = ∇x(∇x(u, metric.Δ), metric.Δ) .+ ∇y(∇y(u, metric.Δ), metric.Δ)
    return sum(d .* metric.mask)
end

function (metric::WaveFlux)(sol::WaveSol{OneDim})
    return [metric(u[:, 1]) for u ∈ sol.u]
end

function (metric::WaveFlux)(sol::WaveSol{TwoDim})
    return [metric(u[:, :, 1]) for u ∈ sol.u]
end

function square_mask(dim::OneDim, radius::Float64)
    mask = zeros(size(dim))
    mask[(-radius .< dim.x .< radius)] .= 1.0
    return mask
end

function square_mask(dim::TwoDim, radius::Float64)
    mask = zeros(size(dim))
    mask[(-radius .< dim.x .< radius), (-radius .< dim.y .< radius)] .= 1.0
    return mask
end

function circle_mask(dim::TwoDim, radius::Float64)
    mask = zeros(size(dim))
    g = grid(dim)

    for i ∈ axes(mask, 1)
        for j ∈ axes(mask, 2)
            x, y = g[i, j]
            mask[i, j] = (x ^ 2 + y ^ 2) <= radius ^ 2
        end
    end

    return mask
end

mutable struct WaveEnv
    iter::ODEIntegrator
    C::WaveSpeed
end

function update!(env::WaveEnv, action, dt)
    design = DesignInterpolator(env.C.design(env.iter.t), action, env.iter.t, env.iter.t + dt)
    C = WaveSpeed(env.C.dim, env.C.C0, design)
    env.C = C
    env.iter.p[2] = env.C
end

gs = 15.0
Δ = 0.3
C0 = 2.0
pml_width = 4.0
pml_scale = 10.0
tspan = (0.0, 20.0)
dim = TwoDim(gs, Δ)
u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)
policy = Uniform.([-2.0, -2.0], [2.0, 2.0])
design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.2), Cylinder(0.0, 0.0, 0.0, 0.0), tspan...)
C = WaveSpeed(dim, C0, design)
pml = build_pml(dim, pml_width) * pml_scale

prob = ODEProblem(split_wave!, u0, tspan, [Δ, C, pml])
iter = init(prob, Midpoint(), advance_to_tstop = true)

env = WaveEnv(iter, C)
dt = 1.0

designs = Vector{DesignInterpolator}()

@time while env.iter.t < tspan[end]
    action = Cylinder(rand.(policy)..., 0.0, 0.0)
    update!(env, action, dt)
    add_tstop!(env.iter, env.iter.t + dt)
    step!(env.iter)
    println("Time: $(env.iter.t)")
    push!(designs, env.C.design)
end

u = Array{Float64, 3}[]
times = Float64[]
design_steps = Cylinder[]

for i ∈ axes(designs, 1)
    design = designs[i]
    t = collect(design.ti:0.1:design.tf)
    for j ∈ axes(t, 1)
        push!(u, env.iter.sol(t[j]))
        push!(times, t[j])
        push!(design_steps, designs[i](t[j]))
    end
end

sol = WaveSol(dim, times, u)

fig = plot(sol.dim)

GLMakie.record(fig, "design.mp4", 1:length(sol)) do i
    GLMakie.empty!(fig.content[1].scene)
    wave = Wave(sol.dim, sol.u[i])
    plot!(fig, wave)
    plot!(fig, design_steps[i])
end

metric = WaveFlux(dim, circle_mask(dim, 8.0))
f = metric(sol)
save(plot(sol.t, f), "flux_design.png")