using CairoMakie
using Flux
using Flux: batch, unbatch
using Waves

include("plot.jl")
include("env.jl")

grid_size = 10.f0
elements = 512
ti = 0.0f0
dt = 0.00002f0
steps = 100

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

dim = TwoDim(grid_size, elements)
g = grid(dim)
grad = build_gradient(dim)
pml = build_pml(dim, 2.0f0, 50.0f0 * ambient_speed)

initial = Scatterers([0.0f0 0.0f0; 0.0f0 -2.0f0], [1.0f0, 1.0f0], [2100.0f0, 2100.0f0])
design = DesignInterpolator(initial)

pulse = Pulse(dim, -4.0f0, 0.0f0, pulse_intensity)
ui = pulse(build_wave(dim, fields = 6))

env = ScatteredWaveEnv(
    ui, ui,
    SplitWavePMLDynamics(design, dim, g, ambient_speed, grad, pml),
    SplitWavePMLDynamics(nothing, dim, g, ambient_speed, grad, pml),
    zeros(Float32, steps + 1), 0, dt, steps)

# iter = Integrator(runge_kutta, env.total, time(env), dt, steps)

# sigma = []
# for i in 1:10
#     # action = Scatterers([0.0f0 0.0f0], [-1.0f0^i * 0.2f0], [0.0f0])
#     action = Scatterers([0.0f0 0.0f0; 0.0f0 0.0f0], [-1.0f0^i * 0.2f0, -1.0f0^i * 0.2f0], [0.0f0, 0.0f0])
#     u = @time batch(env(action))
#     display(size(u))
#     plot_solution!(2, 2, dim, u, path = "results/u$i.png")
#     push!(sigma, env.σ)
# end

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, vcat(sigma...))
# save("results/sigma.png", fig)

action = Scatterers([0.0f0 0.0f0; 0.0f0 0.0f0], [0.2f0, -0.2f0], [0.0f0, 0.0f0])
@time u = env(action)

ui = u[1]
sigma = env.σ

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, sigma)
save("sigma.png", fig)

struct LatentDynamics <: AbstractDynamics
    ambient_speed::Float32
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
    pml_scale::Float32
    pml::Vector{Float32}
end

Flux.@functor LatentDynamics (pml,)

function (dyn::LatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]
    c = wave[:, 3]

    σ = dyn.pml * dyn.pml_scale

    b = (dyn.ambient_speed * c) .^ 2
    du = b .* (dyn.grad * v) .* dyn.bc .- σ .* u
    dv = dyn.grad * u .- σ .* v
    return hcat(du, dv, c * 0.0f0)
end

latent_dim = OneDim(10.0f0, 1024)
latent_dynamics = LatentDynamics(
    ambient_speed,
    build_gradient(latent_dim), 
    dirichlet(latent_dim), 
    70000.0f0,
    build_pml(latent_dim, 1.0f0, 1.0f0))

latent_iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)

d = design.initial
a = design.action

wave_enc = Chain(WaveEncoder(6, 1, 2, relu), Dense(4096, 1024, tanh))
design_enc = DesignEncoder(2 * length(vec(d)), 128, 1024, relu)
zi = hcat(wave_enc(ui), design_enc(d, a))
z = latent_iter(zi)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "vid.mp4", axes(z, 3)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, i], color = :blue, linewidth = 3)
end