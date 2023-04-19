include("dependencies.jl")
using Waves

struct PercentageWaveControlModel <: AbstractWaveControlModel
    wave_encoder::WaveEncoder
    wave_encoder_mlp::Chain

    design_encoder::DesignEncoder
    design_encoder_mlp::Chain

    iter::Integrator
    mlp::Chain
end

Flux.@functor PercentageWaveControlModel

function (model::PercentageWaveControlModel)(s::ScatteredWaveEnvState, a::AbstractDesign)
    u = vec(model.wave_encoder_mlp(model.wave_encoder(s.wave_total)))
    v = u * 0.0f0
    c = model.design_encoder_mlp(model.design_encoder(s.design, a))
    
    zi = hcat(u, v, c)
    z = model.iter(zi)
    return model.mlp(z)
end

env = BSON.load("results/WaveControlModel/env.bson")[:env] |> gpu
policy = RandomDesignPolicy(action_space(env))

data = generate_episode_data(policy, env, 100)
states = vcat([d.states for d in data]...)
actions = vcat([d.actions for d in data]...)
sigmas = vcat([d.sigmas for d in data]...)
train_loader = Flux.DataLoader((states, actions, sigmas), shuffle = true)

s = state(env)
a = policy(env) |> gpu
design_size = 2 * length(vec(a))

latent_elements = 1024
latent_dim = OneDim(grid_size, latent_elements)
latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 5000.0f0)

model = PercentageWaveControlModel(
    WaveEncoder(6, 64, 1, tanh),
    Chain(Dense(latent_elements, latent_elements, tanh), vec),
    DesignEncoder(design_size, 512, latent_elements, relu),
    Chain(c -> 1.0f0 .+ (c .- 0.5f0) * 0.5f0),
    Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
    Chain(Flux.flatten, Dense(3 * latent_elements, latent_elements, relu), Dense(latent_elements, latent_elements, relu), Dense(latent_elements, 1), vec)) |> gpu

model = train(model, train_loader, 50)

u = vec(model.wave_encoder_mlp(model.wave_encoder(s.wave_total)))
v = u * 0.0f0
c = model.design_encoder_mlp(model.design_encoder(s.design, a))

zi = hcat(u, v, c)
z = model.iter(zi)
render!(latent_dim, cpu(z), path = "results/PercentageWaveControlModel/vid.mp4")

for (i, ep) in enumerate(data)
    plot_sigma!(model, ep, path = "results/PercentageWaveControlModel/train_ep$i.png")
end

model = cpu(model)
@save "results/PercentageWaveControlModel/model.bson" model