using ReinforcementLearning
using Flux
using BSON
using FileIO
using Waves
include("plot.jl")

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
Flux.device!(0)
## setting discretization in space and time
grid_size = 15.0f0
elements = 512
dt = 1e-5
## various environment parameters
action_speed = 500.0f0
freq = 2000.0f0
pml_width = 5.0f0
pml_scale = 10000.0f0
actions = 100
integration_steps = 100
## point source settings
pulse_x = -10.0f0
pulse_y = 0.0f0
pulse_intensity = 10.0f0
## number of episodes to generate
episodes = 200
## declaring name of dataset
# name = "single_cylinder_dataset"
# name = "full_state_single_adjustable_radii"
name = "full_state_triple_ring_dataset"

## building FEM grid
dim = TwoDim(grid_size, elements)
grid = build_grid(dim)
pulse = build_pulse(grid, pulse_x, pulse_y, pulse_intensity)

## initializing environment with settings
println("Building WaveEnv")
env = WaveEnv(
    dim,
    reset_wave = Silence(),
    design_space = build_triple_ring_design_space(),
    # design_space = build_simple_radii_design_space(),
    action_speed = action_speed,
    source = Source(pulse, freq = freq),
    # sensor = DisplacementImage(),
    sensor = WaveImage(),
    ambient_speed = WATER,
    pml_width = pml_width,
    pml_scale = pml_scale,
    dt = Float32(dt),
    integration_steps = integration_steps,
    actions = actions) |> gpu

policy = RandomDesignPolicy(action_space(env))

## saving environment
data_path = mkpath("/scratch/cmpe299-fa22/tristan/data/$name/episodes")
BSON.bson(joinpath(data_path, "env.bson"), env = cpu(env))
## rendering a sample animation
println("Rendering Example")
include("plot.jl")

@time render!(policy, env, path = joinpath(data_path, "vid.mp4"), seconds = env.actions * 0.1f0, minimum_value = -0.5f0, maximum_value = 0.5f0)
# starting data generation loop

println("Generating Data")
for i in 1:episodes
    path = mkpath(joinpath(data_path, "episode$i"))
    @time episode = generate_episode_data(policy, env)
    save(episode, joinpath(path, "episode.bson"))
    plot_sigma!(episode, path = joinpath(path, "sigma.png"))
end