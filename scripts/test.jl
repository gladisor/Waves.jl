using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using IntervalSets

using Waves

function radii_design_space(config::Scatterers, scale::Float32)

    pos = zeros(Float32, size(config.pos))

    radii_low = - scale * ones(Float32, size(config.r))
    radii_high =  scale * ones(Float32, size(config.r))
    c = zeros(Float32, size(config.c))

    return Scatterers(pos, radii_low, c)..Scatterers(pos, radii_high, c)
end

function circular_points(radius::Float32, spacing::Float32, num::Int)
    θ = acos(spacing ^ 2 / (2 * radius ^ 2) - 1)

    points = []
    
    push!(points, [radius * cos(π) radius * sin(π)])

    for i ∈ 1:num
        push!(points, [radius * cos(π + i*θ) radius * sin(π + i*θ)])
        push!(points, [radius * cos(π - i*θ) radius * sin(π - i*θ)])
    end

    return vcat(points...)
end

function square_formation()
    points = [
         0.0f0   0.0f0;
        -1.0f0   0.0f0;
        -1.0f0   1.0f0;
         0.0f0   1.0f0;
         1.0f0   1.0f0;
         1.0f0   0.0f0;
         1.0f0  -1.0f0;
         0.0f0  -1.0f0;
        -1.0f0  -1.0f0;  
        ]

    return points * 2
end

grid_size = 5.0f0
elements = 256
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

# pos = circular_points(2.0f0, 2.5f0, 2)
pos = square_formation()

r = ones(Float32, size(pos, 1)) * 0.5f0
c = ones(Float32, size(pos, 1)) * 0.5f0

translation = [-1.0f0, 0.0f0]
config = Scatterers(pos .- translation', r, c)

env = gpu(WaveEnv(
    initial_condition = Pulse(dim, -4.0f0, 0.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    space = radii_design_space(config, 0.2f0),
    design_steps = 50,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))
traj = episode_trajectory(env)
agent = Agent(policy, traj)

@time run(agent, env, StopWhenDone())
render!(traj, path = "vid.mp4")