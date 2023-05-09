include("dependencies.jl")

Flux.device!(0)

## Declaring some important hyperparameters
ambient_speed = AIR
h_channels = 32
h_size = 512
latent_elements = 256
latent_pml_width = 3.0f0
latent_pml_scale = 0.0f0
horizon = 3
lr = 5e-5
epochs = 10
act = relu

## Establising the data pathway and loading in the env
data_path = "data/M=6_as=1.0_normalized"
println("Loading Env")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
println("Resetting Env")
reset!(env)
policy = RandomDesignPolicy(action_space(env))
## Advancing the env several steps so that the latent wave is more representative of steady state behavior
@time [env(policy(env)) for i in 1:5]
s = gpu(state(env))
a = gpu(policy(env))

## Building or loading in model
latent_grid_size = cpu(env.dim.x)[end]
println("Building WaveControlModel")
model = build_hypernet_model(
    grid_size = latent_grid_size,
    elements = latent_elements,
    pml_width = latent_pml_width,
    pml_scale = latent_pml_scale,
    ambient_speed = ambient_speed,
    dt = env.dt,
    steps = env.integration_steps,
    h_channels = h_channels,
    design_input_size = length(vcat(vec(s.design), vec(policy(env)))),
    h_size = h_size,
    act = act) |> gpu

## Loading data into memory
println("Load Train Data")
train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:45])
println("Load Val Data")
val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 46:50])

## Establishing model path
model_path = mkpath(joinpath(data_path, "models/hypernet_testing/h_channels=$(h_channels)_h_size=$(h_size)_latent_elements=$(latent_elements)_latent_pml_width=$(latent_pml_width)_latent_pml_scale=$(latent_pml_scale)_horizon=$(horizon)_lr=$(lr)_epochs=$(epochs)_act=$(act)"))

## Rendering latent wave
latent_dim = OneDim(latent_grid_size, latent_elements)
# render_latent_wave!(latent_dim, model, s, a, path = joinpath(model_path, "latent_wave_original.mp4"))

## Preprocessing data into samples with a particular time horizon
println("Preparing Data")
train_loader = Flux.DataLoader(prepare_data(train_data, horizon), shuffle = true)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon), shuffle = true)

## Executing training command
println("Training Model")

model = train(
    model, 
    train_loader, 
    val_loader, 
    epochs, 
    lr, 
    path = model_path, 
    checkpoint_every = 1,
    evaluation_samples = 5)