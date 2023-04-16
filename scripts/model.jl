using CairoMakie
using Flux
using Flux: unbatch, mse
using Optimisers

using Flux: pullback
using ChainRulesCore


using Interpolations
using Interpolations: Extrapolation
using Waves

include("plot.jl")
using Flux: Params, Recur
using Waves: speed
include("../src/dynamics.jl")
include("env.jl")

struct LatentPMLWaveDynamics <: AbstractDynamics
    ambient_speed::Float32
    pml_scale::Float32

    pml::AbstractArray
    grad::AbstractMatrix
    bc::AbstractArray
end

Flux.@functor LatentPMLWaveDynamics
Flux.trainable(dyn::LatentPMLWaveDynamics) = (;dyn.pml)

function LatentPMLWaveDynamics(dim::AbstractDim; ambient_speed::Float32, pml_scale::Float32)
    pml = zeros(Float32, size(dim)...)
    grad = build_gradient(dim)
    bc = dirichlet(dim)
    return LatentPMLWaveDynamics(ambient_speed, pml_scale, pml, grad, bc)
end

function (dyn::LatentPMLWaveDynamics)(u::AbstractMatrix{Float32}, t::Float32)
    U = u[:, 1]
    V = u[:, 2]
    C = u[:, 3] * dyn.ambient_speed

    ∇ = dyn.grad
    σ = dyn.pml * dyn.pml_scale

    du = C .^ 2 .* ∂x(∇, V) .- σ .* U
    dv = ∂x(∇, U) .- σ .* V
    dc = C * 0.0f0

    return hcat(dyn.bc .* du, dv, dc)
end

struct WaveControlModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function encode(model::WaveControlModel, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    return hcat(model.wave_encoder(wave), model.design_encoder(design, action))
end

function (model::WaveControlModel)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    zi = encode(model, wave, design, action)
    z = model.iter(zi)
    return model.mlp(z)
end

grid_size = 8.0f0
elements = 256
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100
tspan = build_tspan(ti, dt, steps)
tf = tspan[end]

dim = TwoDim(grid_size, elements)
grid = build_grid(dim)
grad = build_gradient(dim)
bc = dirichlet(dim)
pml = build_pml(dim, 2.0f0, 20000.0f0)
wave = build_wave(dim, fields = 6)

pulse = Pulse(dim, -5.0f0, 0.0f0, 1.0f0)
wave = pulse(wave)

initial = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])
# design = linear_interpolation([ti, tf], [initial, initial])
design = DesignInterpolator(initial)

dynamics = SplitWavePMLDynamics(design, dim, grid, ambient_speed, grad, bc, pml)
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

env = ScatteredWaveEnv(
    wave, wave,
    SplitWavePMLDynamics(design, dim, grid, ambient_speed, grad, bc, pml),
    SplitWavePMLDynamics(nothing, dim, grid, ambient_speed, grad, bc, pml),
    zeros(Float32, steps + 1),
    0,
    dt,
    steps) |> gpu

sigma = []

for i in 1:10
    action = Scatterers([0.75f0 * randn(Float32) 0.75f0 * randn(Float32)], [0.0f0], [0.0f0])
    @time env(gpu(action))
    push!(sigma, cpu(env.σ))
end

all_sigma = [sigma[1][1], vcat([s[2:end] for s in sigma]...)...]

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, all_sigma, color = :blue)
save("sigma.png", fig)

# ui = sol.u[1]

# latent_dim = OneDim(grid_size, 1024)
# latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 10000.0f0)

# model = WaveControlModel(
#     WaveEncoder(6, 32, 2, relu),
#     DesignEncoder(2 * length(vec(initial)), 128, 1024, relu),
#     Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
#     Chain(Flux.flatten, Dense(3072, 512, relu), Dense(512, 512, relu), Dense(512, 1), vec))

# opt_state = Optimisers.setup(Optimisers.Adam(1e-5), model)

# for i in 1:10
#     loss, back = Flux.pullback(_model -> mse(sigma, _model(ui, initial, action)), model)
#     gs = back(one(loss))[1]

#     opt_state, model = Optimisers.update(opt_state, model, gs)

#     println(loss)
# end


# # render!(latent_dim, z, path = "vid.mp4")

# # designs = []
# # sigma = []
# # ts = []
# # us = []
# # sols = []

# # for i in 1:10
# #     action = Scatterers([(-1) ^ (i + 1) * 1.0f0 (-1) ^ i * 1.0f0], [0.0f0], [0.0f0])
# #     @time sol = env(action)

# #     push!(sols, sol)
# #     push!(designs, env.total.design)
# #     push!(sigma, env.σ)
# # end

# # design_tspan = vcat(sols[1].t[1], [sol.t[end] for sol in sols]...)
# # ds = [designs[1](0.0f0), [d(sol.t[end]) for (d, sol) in zip(designs, sols)]...]
# # design_interp = linear_interpolation(design_tspan, ds)

# # sol = WaveSol(sols...)
# # u_interp = linear_interpolation(sol.t, sol.u)

# # render!(dim, sol.t, u_interp, design_interp, seconds = 5.0f0)

# # sigma = [sigma[1][1], [s[2:end] for s in sigma]...]

# # fig = Figure()
# # ax = Axis(fig[1, 1])
# # lines!(ax, sol.t, vcat(sigma...), color = :blue)
# # save("sigma2.png", fig)