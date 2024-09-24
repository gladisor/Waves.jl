using Waves
using Flux
using ReinforcementLearning
using CairoMakie
using BSON
using Images: imresize
include("../src/masks.jl")

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

Flux.device!(1)
DATA_PATH = "./scratch/"

dim = TwoDim(15.0f0, 700)
μ_low = [-10.0f0 0.0f0]
μ_high = [-10.0f0 0.0f0]
σ = [0.3f0]
a = [1.0f0]

function Base.rand(space::DesignSpace{NoDesign})
    return NoDesign()
end

M = 1
r = fill(1.0f0, M)
# r_low = fill(0.25f0, M)
# r_high = fill(1.5f0, M)
c = fill(AIR * 3, M)

low_pos = hcat(fill(-8.0f0, M), fill(-8.0f0, M))
high_pos = hcat(fill(8.0f0, M), fill(8.0f0, M))

# low = FullyAdjustableScatterers(Cylinders(low_pos, r_low, c))
# high = FullyAdjustableScatterers(Cylinders(high_pos, r_high, c))
low = AdjustablePositionScatterers(Cylinders(low_pos, r, c))
high = AdjustablePositionScatterers(Cylinders(high_pos, r, c))

design_space = DesignSpace(low, high)

masks = cat(create_patches(700, 350)..., dims = 3)

env = gpu(WaveEnv(dim; 
    design_space=design_space,
    source = RandomPosGaussianSource(build_grid(dim), μ_low, μ_high, σ, a, 1000.0f0),
    integration_steps = 100,
    actions = 200
    ))

policy = RandomDesignPolicy(action_space(env))
# render!(policy, env, path = "vid_x2.mp4")
# ep = generate_episode!(policy, env, position_mask=masks)
# save(ep, "scratch/dataset_pos_adjustment_masked/episodes/episode501.bson")

# name =  "AdditionalDataset" *
#         "$(typeof(env.iter.dynamics))_" *
#         "$(typeof(env.design))_" *
#         "Pulse_" * 
#         "dt=$(env.dt)_" *
#         "steps=$(env.integration_steps)_" *
#         "actions=$(env.actions)_" *
#         "actionspeed=$(env.action_speed)_" *
#         "resolution=$(env.resolution)"

name = "pos_adjustment_masked_signals_M=1"
path = mkpath(joinpath(DATA_PATH, name))
mkpath(joinpath(path, "episodes/"))
BSON.bson(joinpath(path, "env.bson"), env = cpu(env))

for i in 1:500
    ep = generate_episode!(policy, env, position_mask=masks)
    save(ep, joinpath(path, "episodes/episode$i.bson"))
end


function custom_env()

    dim = TwoDim(15.0f0, 700)
    μ_low = [-10.0f0 0.0f0]
    μ_high = [-10.0f0 0.0f0]
    σ = [0.3f0]
    a = [1.0f0]

    M = 15
    r = fill(1.0f0, M)
    c = fill(AIR * 3, M)

    low_pos = hcat(fill(-8.0f0, M), fill(-8.0f0, M))
    high_pos = hcat(fill(8.0f0, M), fill(8.0f0, M))

    low = AdjustablePositionScatterers(Cylinders(low_pos, r, c))
    high = AdjustablePositionScatterers(Cylinders(high_pos, r, c))

    design_space = DesignSpace(low, high)

    env = WaveEnv(dim; 
        design_space=design_space,
        source = RandomPosGaussianSource(build_grid(dim), μ_low, μ_high, σ, a, 1000.0f0),
        integration_steps = 100,
        actions = 200
        )

    DATA_PATH = "./scratch/"
    name = "pos_adjustment_masked_M=2"
    path = mkpath(joinpath(DATA_PATH, name))
    BSON.bson(joinpath(path, "env_1.bson"), env = cpu(env))
end