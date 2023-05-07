include("dependencies.jl")

## Declaring some important hyperparameters
ambient_speed = AIR
h_channels = 32
h_size = 1024
latent_elements = 512
latent_pml_width = 2.0f0
latent_pml_scale = 0.0f0
n_mlp_layers = 3
horizon = 3
lr = 1e-4
# lr = 8e-5 ## for finetuning
epochs = 10

## Establising the data pathway and loading in the env
data_path = "data/M=6_as=1.0_normalized"
println("Loading Env")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])

println("Resetting Env")
reset!(env)
policy = RandomDesignPolicy(action_space(env))

## Advancing the env several steps so that the latent wave is more representative of steady state behavior
[env(policy(env)) for i in 1:5]
s = gpu(state(env))
a = gpu(policy(env))

## Building or loading in model
latent_grid_size = cpu(env.dim.x)[end]
println("Building WaveControlModel")
model = gpu(build_wave_control_model(
    in_channels = 1,
    h_channels = h_channels,
    design_size = length(vec(s.design)),
    action_size = length(vec(a)),
    h_size = h_size,
    latent_grid_size = latent_grid_size,
    latent_elements = latent_elements,
    latent_pml_width = latent_pml_width,
    latent_pml_scale = latent_pml_scale,
    ambient_speed = ambient_speed,
    dt = env.dt,
    steps = env.integration_steps,
    n_mlp_layers = n_mlp_layers))

## Loading data into memory
println("Load Train Data")
train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:46])
println("Load Val Data")
val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 47:50])

## Establishing model path
model_path = mkpath(joinpath(data_path, "models/h_channels=$(h_channels)_h_size=$(h_size)_latent_elements=$(latent_elements)_n_mlp_layers=$(n_mlp_layers)_lr=$(lr)_horizon=$(horizon)_epochs=$(epochs)_latent_pml_scale=$(latent_pml_scale)"))

## Rendering latent wave
latent_dim = OneDim(latent_grid_size, latent_elements)
render_latent_wave!(latent_dim, model, s, a, path = joinpath(model_path, "latent_wave_original.mp4"))

## Preprocessing data into samples with a particular time horizon
println("Preparing Data")
train_states, train_actions, _, train_sigmas = prepare_data(train_data, horizon)
val_states, val_actions, _, val_sigmas = prepare_data(val_data, horizon)
train_loader = Flux.DataLoader((train_states, train_actions, train_sigmas), shuffle = true)
val_loader = Flux.DataLoader((val_states, val_actions, val_sigmas), shuffle = true)

## Executing training command
println("Training Model")
model = train(model, train_loader, val_loader, epochs, lr, path = model_path)

## Rendering the latent wave after training
render_latent_wave!(latent_dim, model, s, a, path = joinpath(model_path, "latent_wave_opt.mp4"))

## Evaluating the models single step prediction capabilities after training
println("Evaluating Sigma Prediction")
for (i, episode) in enumerate(val_data)
    plot_sigma!(model, episode, path = joinpath(model_path, "val_episode$i.png"))
end