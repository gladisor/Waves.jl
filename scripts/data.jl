using Waves
using Flux
using ReinforcementLearning
using CairoMakie
using BSON
using Images: imresize

include("random_pos_gaussian_source.jl")

function build_rectangular_grid(nx::Int, ny::Int, r::Float32)

    x = []

    for i in 1:nx
        for j in 1:ny
            push!(x, [(i-1) * 2 * r, (j-1) * 2 * r])
        end
    end

    x = hcat(x...)

    return x .- Flux.mean(x, dims = 2)
end

function build_rectangular_grid_design_space()

    pos = Matrix(build_rectangular_grid(5, 5, 1.0f0 + 0.1f0)')
    M = size(pos, 1)

    low = AdjustableRadiiScatterers(Cylinders(pos, fill(0.2f0, M), fill(3 * AIR, M)))
    high = AdjustableRadiiScatterers(Cylinders(pos, fill(1.0f0, M), fill(3 * AIR, M)))
    return DesignSpace(low, high)
end

Flux.device!(2)
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/"

dim = TwoDim(15.0f0, 700)
μ_low = [-10.0f0 -10.0f0]
μ_high = [-10.0f0 10.0f0]
σ = [0.3f0]
a = [1.0f0]

# ## single pulse
# μ = zeros(Float32, 1, 2)
# μ[1, :] .= [-10.0f0, 0.0f0]
# σ = [0.3f0]
# a = [1.0f0]
# pulse = build_normal(build_grid(dim), μ, σ, a)

M = 4

r = fill(1.0f0, M)
c = fill(AIR * 3, M)

low_pos = fill(-8.0f0, M, 2)
high_pos = fill(8.0f0, M, 2)

low = AdjustablePositionScatterers(Cylinders(low_pos, r, c))
high = AdjustablePositionScatterers(Cylinders(high_pos, r, c))

design_space = DesignSpace(low, high)

env = gpu(WaveEnv(dim; 
    # design_space = build_rectangular_grid_design_space(),
    design_space=design_space,
    # source = Source(pulse, 1000.0f0),
    source = RandomPosGaussianSource(build_grid(dim), μ_low, μ_high, σ, a, 1000.0f0),
    integration_steps = 100,
    actions = 10
    ))

policy = RandomDesignPolicy(action_space(env))
render!(policy, env, path = "vid.mp4")

# name =  "AdditionalDataset" *
#         "$(typeof(env.iter.dynamics))_" *
#         "$(typeof(env.design))_" *
#         "Pulse_" * 
#         "dt=$(env.dt)_" *
#         "steps=$(env.integration_steps)_" *
#         "actions=$(env.actions)_" *
#         "actionspeed=$(env.action_speed)_" *
#         "resolution=$(env.resolution)"

# name = "part2_variable_source_yaxis_x=-10.0"

# path = mkpath(joinpath(DATA_PATH, name))
# BSON.bson(joinpath(path, "env.bson"), env = cpu(env))

# for i in 1:500
#     ep = generate_episode!(policy, env)
#     save(ep, joinpath(path, "episode$i.bson"))
# end