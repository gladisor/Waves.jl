include("dependencies.jl")

struct WaveMPC <: AbstractWaveControlModel
    wave_encoder::Chain
    design_encoder::Chain
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveMPC

function encode(model::WaveMPC, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    z_wave = model.wave_encoder(wave)
    z_u = z_wave[:, 1]
    z_v = z_wave[:, 2] * 0.0f0

    z_design = model.design_encoder(vcat(vec(design), vec(action)))
    z_design = reshape(z_design, size(z_design, 1) ÷ 2, 2)
    z_f = tanh.(z_design[:, 1])
    z_c = sigmoid.(z_design[:, 2])
    zi = hcat(z_u, z_v, z_f, z_c)
    return zi
end

function (model::WaveMPC)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    z = model.iter(encode(model, wave, design, action))
    return model.mlp(z)
end

function (model::WaveMPC)(s::ScatteredWaveEnvState, action::AbstractDesign)
    return model(s.wave_total, s.design, action)
end

function plot_action_distribution!(
    model::WaveMPC,
    s::ScatteredWaveEnvState,
    policy::RandomDesignPolicy, 
    env::ScatteredWaveEnv; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1])

    for _ in 1:10
        lines!(ax, cpu(model(s, gpu(policy(env)))))
    end

    save(path, fig)
end

function render_latent_wave!(
        dim::OneDim, 
        model::WaveMPC, 
        s::ScatteredWaveEnvState,
        action::AbstractDesign; path::String)

    z = cpu(model.iter(encode(model, s.wave_total, s.design, action)))
    render!(dim, z, path = path)
end

path = "results/radii/PercentageWaveControlModel"

initial_design = random_radii_scatterer_formation(;random_design_kwargs...)

dim = TwoDim(grid_size, elements)
env = gpu(ScatteredWaveEnv(
    dim,
    initial_condition = Pulse(dim, -5.0f0, 0.0f0, 1.0f0),
    design = initial_design,
    ambient_speed = ambient_speed,
    pml_width = 2.0f0,
    pml_scale = 20000.0f0,
    reset_design = d -> gpu(random_radii_scatterer_formation(;random_design_kwargs...)),
    action_space = radii_design_space(initial_design, 1.0f0),
    dt = dt,
    integration_steps = steps,
    actions = 10
))

reset!(env)

policy = RandomDesignPolicy(action_space(env))
;

data = generate_episode_data(policy, env, 50)
episode = first(data)
# plot_episode_data!(episode, cols = 5, path = "data.png")
# plot_sigma!(episode, path = "episode_sigma.png")

idx = 6
s = gpu(episode.states[idx])
a = gpu(episode.actions[idx])
sigma = gpu(episode.sigmas[idx])

wave = s.wave_total
design = s.design
design_size = length(vec(design))

h_size = 128
latent_elements = 512
latent_dim = OneDim(grid_size, latent_elements)

latent_dynamics = ForceLatentDynamics(ambient_speed, build_gradient(latent_dim), dirichlet(latent_dim))
model = gpu(WaveMPC(
    Chain(WaveEncoder(6, 8, 2, tanh), Dense(1024, latent_elements, tanh)),
    Chain(Dense(2 * design_size, h_size, relu), Dense(h_size, 2 * latent_elements)),
    Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
    Chain(flatten, Dense(latent_elements * 4, h_size, relu), Dense(h_size, 1), vec)
))

## package data into train_loader
states = vcat([d.states for d in data]...)
actions = vcat([d.actions for d in data]...)
sigmas = vcat([d.sigmas for d in data]...)
train_loader = Flux.DataLoader((states, actions, sigmas), shuffle = true)
println("Train Loader Length: $(length(train_loader))")

# plot the latent wave before training
plot_action_distribution!(model, s, policy, env, path = "action_distribution_original.png")
render_latent_wave!(latent_dim, model, s, a, path = "latent_wave_original.mp4")

# ## train the model
model = train(model, train_loader, 20)
## plot latent wave after training
plot_action_distribution!(model, s, policy, env, path = "action_distribution_opt.png")
render_latent_wave!(latent_dim, model, s, a, path = "latent_wave_opt.mp4")

## generate and plot prediction performance after training
validation_episode = generate_episode_data(policy, env, 2)
for (i, ep) in enumerate(validation_episode)
    plot_sigma!(model, ep, path = "validation_ep$i.png")
end

## saving model and env
model = cpu(model)
@save "model.bson" model
env = cpu(env)
@save "env.bson" env