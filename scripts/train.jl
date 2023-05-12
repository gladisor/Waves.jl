include("dependencies.jl")

Flux.device!(0)

## Declaring some important hyperparameters
ambient_speed = AIR
h_size = 512
latent_elements = 512
latent_pml_width = 5.0f0
latent_pml_scale = 0.0f0
horizon = 3
lr = 1e-5
epochs = 20
act = gelu
num_train_episodes = 25
num_val_episodes = 5
decay = 0.95f0

## Establising the data pathway and loading in the env
data_path = "data/hexagon_large_grid"
println("Loading Env")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
println("Resetting Env")
reset!(env)
policy = RandomDesignPolicy(action_space(env))
s = state(env)
a = policy(env)

## Building model
design_action = vcat(vec(s.design), vec(a))
latent_dim = OneDim(cpu(s.dim.x)[end], latent_elements)

model = build_hypernet_speed_model(
    h_size = h_size,
    act = act,
    latent_dim = latent_dim,
    ambient_speed = ambient_speed,
    freq = env.total_dynamics.source.freq,
    design_action_size = length(design_action),
    dt = env.dt,
    steps = env.integration_steps,
    pml_width = latent_pml_width,
    pml_scale = latent_pml_scale) |> gpu

## Loading data into memory
println("Load Train Data")
train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:num_train_episodes])
println("Load Val Data")
val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in (num_train_episodes+1):(num_train_episodes + 1 + num_val_episodes)])

## Establishing model path
model_path = mkpath(joinpath(data_path, "models/speed_force_hypernet/bounded_speed_sigmoid=5.0/h_size=$(h_size)_elements=$(latent_elements)_pml_width=$(latent_pml_width)_pml_scale=$(latent_pml_scale)_horizon=$(horizon)_lr=$(lr)_epochs=$(epochs)_act=$(act)_num_train_episodes=$(num_train_episodes)_decay=$(decay)"))

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
    latent_dim = latent_dim,
    path = model_path, 
    checkpoint_every = 1,
    evaluation_samples = 10,
    decay = decay)