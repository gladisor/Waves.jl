using CairoMakie
using Flux
using Flux: unbatch, mse, huber_loss
Flux.CUDA.allowscalar(false)
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

using ReinforcementLearning
using IntervalSets
include("env.jl")

using BSON: @save

struct WaveControlModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function encode(model::WaveControlModel, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    u = model.wave_encoder(wave)
    v = u * 0.0f0
    return hcat(u, v, model.design_encoder(design, action))
end

function (model::WaveControlModel)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    zi = encode(model, wave, design, action)
    z = model.iter(zi)
    return model.mlp(z)
end

function build_control_sequence(action::AbstractDesign, steps::Int)
    return [zero(action) for i in 1:steps]
end

function build_mpc_cost(model::WaveControlModel, s::ScatteredWaveEnvState, control_sequence::Vector{ <: AbstractDesign})
    cost = 0.0f0

    d1 = s.design
    c1 = model.design_encoder(d1, control_sequence[1])
    z1 = hcat(model.wave_encoder(s.wave_total), c1)
    z = model.iter(z1)
    cost = cost + sum(model.mlp(z))

    d2 = d1 + control_sequence[1]
    c2 = model.design_encoder(d2, control_sequence[2])
    z2 = hcat(z[:, 1:2, end], c2)
    z = model.iter(z2)
    cost = cost + sum(model.mlp(z))

    d3 = d2 + control_sequence[2]
    c3 = model.design_encoder(d3, control_sequence[3])
    z3 = hcat(z[:, 1:2, end], c3)
    z = model.iter(z3)
    cost = cost + sum(model.mlp(z))

    return cost
end

struct EpisodeData
    states::Vector{ScatteredWaveEnvState}
    actions::Vector{<: AbstractDesign}
    tspans::Vector{Vector{Float32}}
    sigmas::Vector{Vector{Float32}}
end

Flux.@functor EpisodeData

function generate_episode_data(policy::AbstractPolicy, env::ScatteredWaveEnv)
    states = ScatteredWaveEnvState[]
    actions = AbstractDesign[]
    tspans = []
    sigmas = []

    reset!(env)

    while !is_terminated(env)
        action = gpu(policy(env))
        tspan = build_tspan(time(env), env.dt, env.integration_steps)

        push!(states, cpu(state(env)))
        push!(actions, cpu(action))
        push!(tspans, tspan)

        @time env(action)

        push!(sigmas, cpu(env.σ))
    end

    return EpisodeData(states, actions, tspans, sigmas)
end

function generate_episode_data(policy::AbstractPolicy, env::ScatteredWaveEnv, episodes::Int)
    data = EpisodeData[]

    for episode in 1:episodes
        episode_data = generate_episode_data(policy, env)
        push!(data, episode_data)
    end

    return data
end

function plot_sigma!(episode_data::EpisodeData; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, vcat(episode_data.tspans...), vcat(episode_data.sigmas...))
    save(path, fig)
    return nothing
end

function plot_episode_data!(episode_data::EpisodeData; cols::Int, path::String)

    fig = Figure(resolution = (1920, 1080))

    for i in axes(episode_data.states, 1)
        dim = episode_data.states[i].dim
        wave = episode_data.states[i].wave_total
        design = episode_data.states[i].design

        row = (i - 1) ÷ cols
        col = (i - 1) % cols + 1

        ax = Axis(fig[row, col], aspect = 1.0f0)
        heatmap!(ax, dim.x, dim.y, wave[:, :, 1], colormap = :ice)
        mesh!(ax, design)
    end

    save(path, fig)
    return nothing
end

function plot_sigma!(model::WaveControlModel, episode::EpisodeData; path::String)
    pred_sigmas = cpu([model(gpu(s.wave_total), gpu(s.design), gpu(a)) for (s, a) in zip(episode.states, episode.actions)])

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, vcat(episode.tspans...), vcat(episode.sigmas...), color = :blue)
    lines!(ax, vcat(episode.tspans...), vcat(pred_sigmas...), color = :orange)
    save(path, fig)
    return nothing
end

Flux.trainable(config::Scatterers) = (;config.r,)

grid_size = 8.0f0
elements = 256
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100
tf = ti + steps * dt

dim = TwoDim(grid_size, elements)
pulse = Pulse(dim, -5.0f0, 0.0f0, 1.0f0)
initial_design = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])

env = ScatteredWaveEnv(
    dim,
    initial_condition = gpu(pulse),
    design = initial_design,
    pml_width = 2.0f0,
    pml_scale = 20000.0f0,
    reset_design = d -> random_pos(d, 3.0f0),
    action_space = Waves.design_space(initial_design, 1.0f0),
    dt = dt,
    integration_steps = steps,
    max_steps = 1000
) |> gpu

policy = RandomDesignPolicy(action_space(env))
data = generate_episode_data(policy, env, 3)

latent_dim = OneDim(grid_size, 1024)
latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 5000.0f0)

activation = relu
model = WaveControlModel(
    WaveEncoder(6, 64, 1, tanh),
    DesignEncoder(2 * length(vec(initial_design)), 256, 1024, activation),
    Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
    Chain(Flux.flatten, Dense(3072, 1024, activation), Dense(1024, 1024, activation), Dense(1024, 1), vec)
    ) |> gpu

episode = data[1]
s, d, a, sigma = episode.states[5].wave_total, episode.states[5].design, episode.actions[5], episode.sigmas[5]
s, d, a, sigma = gpu(s), gpu(d), gpu(a), gpu(sigma)

states = vcat([d.states for d in data]...)
actions = vcat([d.actions for d in data]...)
sigmas = vcat([d.sigmas for d in data]...)

train_loader = Flux.DataLoader((states, actions, sigmas), shuffle = true)
println("Train Loader Length: $(length(train_loader))")
# opt_state = Optimisers.setup(Optimisers.Adam(8e-5), model)
opt_state = Optimisers.setup(Optimisers.Adam(1e-4), model)


zi = encode(model, s, d, a)
z = model.iter(zi)
render!(latent_dim, cpu(z), path = "latent_wave_original.mp4")

for i in 1:50
    train_loss = 0.0f0

    for batch in train_loader
        s, a, sigma = gpu.(batch)

        loss, back = pullback(_model -> sqrt(mse(_model(s[1].wave_total, s[1].design, a[1]), sigma[1])), model)
        gs = back(one(loss))[1]
        opt_state, model = Optimisers.update(opt_state, model, gs)

        train_loss += loss
    end

    print("Epoch: $i, Loss: ")
    println(train_loss / length(data))
end

plot_sigma!(model, episode, path = "sigma_opt.png")

episode = data[1]
s, d, a, sigma = episode.states[5].wave_total, episode.states[5].design, episode.actions[5], episode.sigmas[5]
s, d, a, sigma = gpu(s), gpu(d), gpu(a), gpu(sigma)

zi = encode(model, s, d, a)
z = model.iter(zi)
render!(latent_dim, cpu(z), path = "latent_wave_opt.mp4")

@save "model.bson" model

validation_episode = generate_episode_data(policy, env, 1)
plot_sigma!(model, validation_episode[1], path = "validation_ep.png")


