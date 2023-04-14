using CairoMakie
using Flux
using Flux: batch, unbatch, mse
using Optimisers
using Optimisers: Restructure
using Waves

include("plot.jl")
include("env.jl")

struct LatentDynamics <: AbstractDynamics
    ambient_speed::Float32
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
    grid::AbstractArray{Float32}
    pml_scale::Float32

    pml_model_ps::Vector{Float32}
    pml_model_re::Restructure
end

Flux.@functor LatentDynamics (pml_model_ps,)

function (dyn::LatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]
    c = wave[:, 3]

    pml_model = dyn.pml_model_re(dyn.pml_model_ps)
    σ = pml_model(dyn.grid') * dyn.pml_scale

    b = (dyn.ambient_speed * c) .^ 2
    du = b .* (dyn.grad * v) .* dyn.bc .- σ .* u
    dv = dyn.grad * u .- σ .* v
    return hcat(du, dv, c * 0.0f0)
end

struct WaveControlModel
    wave_encoder::Chain
    design_encoder::DesignEncoder
    latent_iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function (model::WaveControlModel)(wave, design, action)
    zi = hcat(model.wave_encoder(wave), model.design_encoder(design, action))
    z = model.latent_iter(zi)
    mlp(z)
end

grid_size = 10.f0
elements = 512
ti = 0.0f0
dt = 0.00002f0
steps = 100

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

dim = TwoDim(grid_size, elements)
g = build_grid(dim)
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

action = Scatterers([0.0f0 0.0f0; 0.0f0 0.0f0], [0.2f0, -0.2f0], [0.0f0, 0.0f0])
@time u = env(action)

ui = u[1]
sigma = env.σ

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, sigma)
save("sigma.png", fig)

latent_elements = 1024
latent_dim = OneDim(10.0f0, latent_elements)
pml_model = Chain(Dense(1, 64, relu), Dense(64, 64, relu), Dense(64, 1), vec, x -> x .^ 2)
pml_model_ps, pml_model_re = Flux.destructure(pml_model)

latent_dynamics = LatentDynamics(
    ambient_speed,
    build_gradient(latent_dim), 
    dirichlet(latent_dim), 
    build_grid(latent_dim),
    70000.0f0,
    pml_model_ps,
    pml_model_re
    )

latent_iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)

d = design.initial
a = design.action

wave_enc = Chain(WaveEncoder(6, 8, 2, relu), Dense(4096, 1024, tanh))
design_enc = DesignEncoder(2 * length(vec(d)), 128, 1024, relu)
mlp = Chain(Flux.flatten, Dense(3 * 1024, 512, relu), Dense(512, 1), vec)
model = WaveControlModel(wave_enc, design_enc, latent_iter, mlp)

opt_state = Optimisers.setup(Optimisers.Adam(1e-5), model)

for i in 1:20

    loss, gs = Flux.withgradient(model) do _model
        mse(_model(ui, d, a), sigma)
    end

    println(loss)
    opt_state, model = Optimisers.update(opt_state, model, gs[1])
end

zi = hcat(model.wave_encoder(ui), model.design_encoder(d, a))
z = model.latent_iter(zi)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "vid.mp4", axes(z, 3)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, i], color = :blue, linewidth = 3)
end