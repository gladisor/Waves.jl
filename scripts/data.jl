using ReinforcementLearning
using Flux
using BSON
using FileIO
using Waves

function plot_sigma(episode::EpisodeData; path::String)
    _, _, t, y = prepare_data(episode, length(episode))
    tspan = flatten_repeated_last_dim(t[1])

    fig = Figure()
    ax = Axis(fig[1, 1], 
        title = "Sigma During Episode",
        xlabel = "Time (s)",
        ylabel = "Sigma")

    lines!(ax, tspan, y[1], color = :blue)
    save(path, fig)

    return nothing
end

## selecting gpu
# Flux.device!(0)
## setting discretization in space and time
grid_size = 15.0f0
elements = 128
dt = 1e-5
## various environment parameters
action_speed = 500.0f0
freq = 1000.0f0
pml_width = 5.0f0
pml_scale = 10000.0f0
actions = 20
integration_steps = 100
## point source settings
pulse_x = -10.0f0
pulse_y = 0.0f0
pulse_intensity = 10.0f0
## number of episodes to generate
episodes = 3
## declaring name of dataset
name = "actions=$(actions)_freq=$(freq)"

## building FEM grid
dim = TwoDim(grid_size, elements)
grid = build_grid(dim)
pulse = build_pulse(grid, pulse_x, pulse_y, pulse_intensity)

## initializing environment with settings
println("Building WaveEnv")

env = WaveEnv(
    dim,
    reset_wave = Silence(),
    design_space = Waves.build_triple_ring_design_space(),
    resolution = (64, 64),
    action_speed = action_speed,
    source = Source(pulse, freq = freq),
    ambient_speed = WATER,
    pml_width = pml_width,
    pml_scale = pml_scale,
    dt = Float32(dt),
    integration_steps = integration_steps,
    actions = actions) |> gpu

policy = RandomDesignPolicy(action_space(env))

# saving environment
STORAGE_PATH = ""
data_path = mkpath(joinpath(STORAGE_PATH, "$name/episodes"))
BSON.bson(joinpath(data_path, "env.bson"), env = cpu(env))

# rendering a sample animation
println("Rendering Example")
@time render!(policy, env, path = "test_vid.mp4", seconds = env.actions * 0.5f0, minimum_value = -0.5f0, maximum_value = 0.5f0)

@time episode = generate_episode_data(policy, env)
plot_sigma(episode, path = "sigma.png")

# starting data generation loop
# println("Generating Data")
# for i in 1:episodes
#     path = mkpath(joinpath(data_path, "episode$i"))
#     @time episode = generate_episode_data(policy, env)
#     save(episode, joinpath(path, "episode.bson"))
#     plot_sigma(episode, path = joinpath(path, "sigma.png"))
# end