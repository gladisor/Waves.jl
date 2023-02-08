using DifferentialEquations
using DifferentialEquations: init
using Waves
using Waves: AbstractDesign, ∇x, ∇y, AbstractDim
import GLMakie

function displacement(wave::Wave{TwoDim})
    return wave.u[:, :, 1]
end

function flux(wave::Wave{TwoDim})
    u = displacement(wave)
    Δ = (dim.x[end] - dim.x[1]) / (length(dim.x) - 1)
    return (∇x(∇x(u, Δ), Δ) .+ ∇y(∇y(u, Δ), Δ))
end

function square_mask(dim::TwoDim, r::Float64)
    g = grid(dim)
    mask = zeros(size(g))

    for i ∈ axes(g, 1)
        for j ∈ axes(g, 2)
            x, y = g[i, j]
            mask[i, j] = max(x^2, y^2) <= r^2
        end
    end

    return mask
end

function plot_flux!(sol, dim::TwoDim, mask; path, dt = 0.1)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis(fig[1, 1], title = "Flux", xlabel = "Time (s)", ylabel = "Flux")

    n = Int(round((sol.prob.tspan[end] - sol.prob.tspan[1]) / dt))

    flux_over_time = Float64[]
    tspan = Float64[]

    for i ∈ 1:n
        t = dt * i
        wave = Wave(dim, sol(t))

        push!(tspan, t)
        push!(flux_over_time, sum(flux(wave) .* mask))
    end

    GLMakie.lines!(ax, tspan, flux_over_time, color = :blue)
    save(fig, path)
end

mutable struct WaveSim{D <: AbstractDim}
    dim::D
    C::WaveSpeed
    pml::AbstractArray
    iter::OrdinaryDiffEq.ODEIntegrator
end

include("configuration.jl")

# M = 8
# r = 0.5
# c = 0.2
# config = Configuration(dim, M = M, r = r, c = c, offset = pml_width)
# final = Configuration(dim, M = M, r = r, c = c, offset = pml_width)
# action = (final - config) / 2.0
# design = DesignInterpolator(config, action, tspan...)

gs = 10.0
Δ = 0.1
C0 = 1.0
design = nothing
pml_width = 2.0
pml_scale = 10.0
tspan = (0.0, 20.0)

dim = TwoDim(gs, Δ)
C = WaveSpeed(dim, C0, design)
pml = build_pml(dim, pml_width) * pml_scale
u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)
prob = ODEProblem(split_wave!, u0, tspan, [Δ, C, pml])
iter = init(prob, Midpoint(), advance_to_tstop = true)
sim = WaveSim(dim, C, pml, iter)
reinit!(sim.iter)
step!(sim.iter)

mask = square_mask(dim, 5.0)

function flux(sim::WaveSim, mask, tspan, dt = 0.05)
    f = Float64[]

    for t ∈ tspan[1]:dt:tspan[end]
        wave = Wave(dim, sim.iter.sol(t))
        push!(f, sum(mask .* flux(wave)))
    end

    return f
end

f = flux(sim, mask, tspan)

# render!(sim.iter.sol, dim, path = "no_obstical.mp4")
# mask = square_mask(dim, gs - pml_width - 1.0)
# plot_flux!(sim.iter.sol, dim, mask, path = "flux.png")

