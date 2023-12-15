using Waves
using Flux
using ReinforcementLearning
using CairoMakie
using BSON
using Distributions: Uniform

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

struct RandomPosGaussianSource <: AbstractSource
    dim::TwoDim

    low::AbstractVector{Float32}
    high::AbstractVector{Float32}

    μ::AbstractVector{Float32}
    σ::Float32
    a::Float32

    freq::Float32
end

Flux.@functor RandomPosGaussianSource

function reset!(source::RandomPosGaussianSource)
    x = rand(Uniform(source.x_low, source.x_high))
    y = rand(Uniform(source.y_low, source.y_high))
    source.μ = [x, y]
end

function (source::RandomPosGaussianSource)(t::AbstractVector{Float32})
    shape = build_normal(build_grid(source.dim), source.μ, [source.σ], [source.a])
    return shape .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
end


Flux.device!(1)
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/"

dim = TwoDim(15.0f0, 700)

## beam
# n = 10
# μ = zeros(Float32, n, 2)
# μ[:, 1] .= -10.0f0
# μ[:, 2] .= range(-2.0f0, 2.0f0, n)
# σ = ones(Float32, n) * 0.3f0
# a = ones(Float32, n) * 0.3f0

## single pulse
μ = zeros(Float32, 1, 2)
μ[1, :] .= [-10.0f0, 0.0f0]
σ = [0.3f0]
a = [1.0f0]
pulse = build_normal(build_grid(dim), μ, σ, a)

env = gpu(WaveEnv(dim; 
    # design_space = build_rectangular_grid_design_space(),
    design_space = Waves.build_triple_ring_design_space(),
    source = Source(pulse, 1000.0f0),
    integration_steps = 100,
    actions = 200
    ))

policy = RandomDesignPolicy(action_space(env))
# render!(policy, env, path = "vid.mp4")

# name =  "AdditionalDataset" *
#         "$(typeof(env.iter.dynamics))_" *
#         "$(typeof(env.design))_" *
#         "Pulse_" * 
#         "dt=$(env.dt)_" *
#         "steps=$(env.integration_steps)_" *
#         "actions=$(env.actions)_" *
#         "actionspeed=$(env.action_speed)_" *
#         "resolution=$(env.resolution)"

# path = mkpath(joinpath(DATA_PATH, name))
# BSON.bson(joinpath(path, "env.bson"), env = cpu(env))

# for i in 1:500
#     ep = generate_episode!(policy, env)
#     save(ep, joinpath(path, "episode$i.bson"))
# end