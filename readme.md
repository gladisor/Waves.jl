# Waves.jl

## Usage

First install all dependancies using the instantiate comand in the REPL. To run a simulation execute the following code.

```
instantiate
```

To build the environment run the following code:
```
function build_simple_radii_design_space()
    pos = [0.0f0 0.0f0]

    r_low = fill(0.2f0, size(pos, 1))
    r_high = fill(1.0f0, size(pos, 1))
    c = fill(Waves.AIR, size(pos, 1))

    core = Cylinders([5.0f0, 0.0f0]', [2.0f0], [AIR])

    design_low = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_low, c)), core)
    design_high = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_high, c)), core)

    return DesignSpace(design_low, design_high)
end

function build_radii_design_space(pos::AbstractMatrix{Float32})

    DESIGN_SPEED = 3 * AIR

    r_low = fill(0.2f0, size(pos, 1))
    r_high = fill(1.0f0, size(pos, 1))
    c = fill(DESIGN_SPEED, size(pos, 1))

    core = Cylinders([5.0f0, 0.0f0]', [2.0f0], [DESIGN_SPEED])

    design_low = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_low, c)), core)
    design_high = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_high, c)), core)

    return DesignSpace(design_low, design_high)
end

function build_triple_ring_design_space()

    rot = Float32.(Waves.build_2d_rotation_matrix(30))

    cloak_rings = vcat(
        Waves.hexagon_ring(3.5f0),
        Waves.hexagon_ring(4.75f0) * rot,
        Waves.hexagon_ring(6.0f0)
    )

    pos = cloak_rings .+ [5.0f0, 0.0f0]'
    return build_radii_design_space(pos)
end

## selecting gpu
## setting discretization in space and time
grid_size = 15.0f0
elements = 512
dt = 1e-5
## various environment parameters
action_speed = 500.0f0
# freq = 2000.0f0
freq = 1000.0f0
pml_width = 5.0f0
pml_scale = 10000.0f0
actions = 10 #200 #100
integration_steps = 100
## point source settings
pulse_x = -10.0f0
pulse_y = 0.0f0
pulse_intensity = 10.0f0
## number of episodes to generate
episodes = 1000
## declaring name of dataset
design_space_func = build_triple_ring_design_space
name = "actions=$(actions)_design_space=$(design_space_func)_freq=$(freq)"

## building FEM grid
dim = TwoDim(grid_size, elements)
grid = build_grid(dim)
pulse = build_pulse(grid, pulse_x, pulse_y, pulse_intensity)

## initializing environment with settings
println("Building WaveEnv")

env = WaveEnv(
    dim,
    reset_wave = Silence(),
    design_space = design_space_func(),
    action_speed = action_speed,
    source = Source(pulse, freq = freq),
    sensor = WaveImage(),
    ambient_speed = WATER,
    pml_width = pml_width,
    pml_scale = pml_scale,
    dt = Float32(dt),
    integration_steps = integration_steps,
    actions = actions) |> gpu

policy = RandomDesignPolicy(action_space(env))
```

To render an animation:
```
## saving environment
data_path = mkpath(joinpath(STORAGE_PATH, "$name/episodes"))
BSON.bson(joinpath(data_path, "env.bson"), env = cpu(env))

## rendering a sample animation
@time render!(policy, env, path = "test_vid.mp4", seconds = env.actions * 0.5f0, minimum_value = -0.5f0, maximum_value = 0.5f0)
```

To generate and save data:
```
# # starting data generation loop
println("Generating Data")
for i in 1:episodes
    path = mkpath(joinpath(data_path, "episode$i"))
    @time episode = generate_episode_data(policy, env)
    save(episode, joinpath(path, "episode.bson"))
    plot_sigma!(episode, path = joinpath(path, "sigma.png"))
end
```

Building and training the model:
```
main_path = "..."
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Declaring Hyperparameters")
nfreq = 200
h_size = 256
activation = leakyrelu
latent_grid_size = 30.0f0 #15.0f0
latent_elements = 700 #512
horizon = 20
batchsize = 32

pml_width = 10.0f0
pml_scale = 10000.0f0
lr = 1e-4
decay_rate = 1.0f0
k_size = 2
steps = 20
epochs = 500
loss_func = Flux.mse

MODEL_PATH = mkpath(joinpath(main_path, "models/RERUN/latent_gs=$(latent_grid_size)_latent_elements=$(latent_elements)_horizon=$(horizon)_nfreq=$(nfreq)_pml=$(pml_scale)_lr=$(lr)_batchsize=$(batchsize)"))
println(MODEL_PATH)

println("Initializing Model Components")
latent_dim = OneDim(latent_grid_size, latent_elements)
wave_encoder = build_split_hypernet_wave_encoder(latent_dim = latent_dim, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)
mlp = build_scattered_wave_decoder(latent_elements, h_size, k_size, activation)
println("Constructing Model")
model = gpu(ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp))

println("Initializing DataLoaders")
@time begin
    train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:100])
    val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 101:120])
    train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize, partial = false)
    val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize, partial = false)
end

opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(), Optimisers.Adam(lr))
states, actions, tspans, sigmas = gpu(first(train_loader))

println("Training")
train_loop(
    model,
    loss_func = loss_func,
    train_steps = steps,
    val_steps = steps,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = epochs,
    lr = lr,
    decay_rate = decay_rate,
    evaluation_samples = 15,
    checkpoint_every = 10,
    path = MODEL_PATH,
    opt = opt
    )
```

Details on running MPC can be found in the mpc.jl file in the scripts file. 

Relevant plots can be generated by running the following files:
```
latent_energy_plot.jl
mpc_performance_plot.jl
planning_horizon_plot.jl
```