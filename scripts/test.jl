using Waves

using Interpolations: Extrapolation
using ReinforcementLearning
using CairoMakie

include("plot.jl")

using ProgressMeter: @showprogress
using Flux
using Flux: DataLoader, flatten, mse, pullback
using Optimisers
include("wave_control_model.jl")

episode = EpisodeData(path = "data/double_ring_cloak/episode1/episode.bson")
# states, actions, tspans, sigmas = prepare_data(episode, 1)

# idx = 10
# s = gpu(states[idx])
# a = gpu(actions[idx])
# tspan = gpu(tspans[idx])
# sigma = gpu(sigmas[idx])

# model = gpu(build_wave_control_model(
#     in_channels = 1,
#     h_channels = 16,
#     design_size = length(vec(s.design)),
#     action_size = length(vec(a[1])),
#     h_size = 256,
#     latent_grid_size = 15.0f0,
#     latent_elements = 512,
#     latent_pml_width = 1.0f0,
#     latent_pml_scale = 20000.0f0,
#     ambient_speed = AIR,
#     dt = 5e-5,
#     steps = 100,
#     n_mlp_layers = 3))

function (pulse::Pulse)()
    return build_pulse(pulse.grid, pulse.pos..., pulse.intensity)
end

struct LatentDynamics <: AbstractDynamics
    ambient_speed::Float32
    source::Source
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor LatentDynamics
Flux.trainable(dyn::LatentDynamics) = (;)

function (dyn::LatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    U = wave[:, 1]
    V = wave[:, 2]

    dU = dyn.grad * V
    dV = dyn.grad * U

    return hcat(dU .* dyn.bc, dV)
end

dim = OneDim(15.0f0, 100000)
# pulse = Pulse(dim, intensity = 5.0f0)
# source = Source(pulse(), freq = 200.0f0)
grad = build_gradient(dim)# |> gpu
# bc = dirichlet(dim)

# dyn = LatentDynamics(AIR, source, grad, bc)
# iter = Integrator(runge_kutta, dyn, 0.0f0, 5e-5, 1000) |> gpu
wave = build_wave(dim, fields = 2)# |> gpu

@time grad * wave[:, 1]
@time grad * wave[:, 1]
@time grad * wave[:, 1]


# @time u = iter(wave)
# @time u = iter(wave)
# @time u = iter(wave)
;

