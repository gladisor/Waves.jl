include("dependencies.jl")

function build_rotation_matrix(theta::Float32)
    return [
        cos(pi * theta / 180.0f0) -sin(pi * theta / 180.0f0);
        sin(pi * theta / 180.0f0) cos(pi * theta / 180.0f0)
    ]
end

function single_cylinder_cloak(x_offset::Float32)
    offset = [x_offset, 0.0f0]
    pos = [-3.3f0 0.0f0] .+ offset'
    config = Radii(pos, AIR)
    core = Scatterers(offset', [2.0f0], [AIR])
    return RandomCloak(config, core)
end

function single_ring_cloak(x_offset::Float32)
    offset = [x_offset, 0.0f0]
    pos = hexagon(3.3f0) .+ offset'
    config = Radii(pos, AIR)
    core = Scatterers(offset', [2.0f0], [AIR])
    return RandomCloak(config, core)
end

function double_ring_cloak(x_offset::Float32)
    offset = [x_offset, 0.0f0]
    ring1 = hexagon(3.3f0)

    rot = build_rotation_matrix(30.0f0)
    ring2_radius = 4.4f0
    ring2 = (rot * hexagon(ring2_radius)')'

    pos = vcat(ring1, ring2) .+ offset'

    config = Radii(pos, AIR)
    core = Scatterers(offset', [2.0f0], [AIR])
    return RandomCloak(config, core)
end

Flux.device!(0)

grid_size = 15.0f0
elements = 512
dt = 1e-5

dim = TwoDim(grid_size, elements)
grid = build_grid(dim)

design_func = single_cylinder_cloak
# design_func = single_ring_cloak
# design_func = double_ring_cloak
cloak = design_func(4.0f0)
pulse = build_pulse(grid, -10.0f0, 0.0f0, 10.0f0)

action_scale = 1.0f0
pml_width = 5.0f0
pml_scale = 10000.0f0
actions = 100
integration_steps = 100

println("Building WaveEnv")
env = gpu(WaveEnv(
    dim, 
    reset_wave = Silence(),
    reset_design = cloak,
    action_space = design_space(cloak, action_scale),
    source = Source(pulse, freq = 2000.0f0),
    sensor = DisplacementImage(),
    ambient_speed = WATER,
    pml_width = pml_width,
    pml_scale = pml_scale,
    actions = actions,
    integration_steps = integration_steps,
    dt = Float32(dt))
    )

policy = RandomDesignPolicy(action_space(env))

data_path = mkpath("data/testing/dt=$dt/$design_func/episodes")

println("Rendering Example")
@time render!(policy, env, path = joinpath(data_path, "vid.mp4"), seconds = env.actions * 0.5f0)

# println("Generating Data")
# for i in 1:100
#     path = mkpath(joinpath(data_path, "episode$i"))
#     @time episode = generate_episode_data(policy, env)
#     save(episode, joinpath(path, "episode.bson"))
#     plot_sigma!(episode, path = joinpath(path, "sigma.png"))
# end

env = cpu(env)
BSON.bson(joinpath(data_path, "env.bson"), env = env)
