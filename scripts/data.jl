using Waves
using Flux
using ReinforcementLearning
using CairoMakie
using BSON
using Images: imresize

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

mutable struct RandomPosGaussianSource <: AbstractSource
    grid::AbstractArray{Float32, 3}

    μ_low::AbstractMatrix{Float32}
    μ_high::AbstractMatrix{Float32}

    σ::AbstractVector{Float32}
    a::AbstractVector{Float32}

    shape::AbstractMatrix{Float32}
    freq::Float32
end

Flux.@functor RandomPosGaussianSource
Flux.trainable(::RandomPosGaussianSource) = (;)

function Waves.reset!(source::RandomPosGaussianSource)
    ϵ = rand(Float32, size(source.μ_low)...) ## generate some noise

    if Flux.device(source.grid) != Val{:cpu}()
        ϵ = gpu(ϵ)
    end

    μ = (source.μ_high .- source.μ_low) .* ϵ .+ source.μ_low ## scale noise to high and low bounds
    source.shape = build_normal(source.grid, μ, source.σ, source.a) ## build normal distribution shape
    return nothing
end

function RandomPosGaussianSource(
        grid::AbstractArray{Float32, 3},
        μ_low::AbstractMatrix, 
        μ_high::AbstractMatrix, 
        σ::AbstractVector, 
        a::AbstractVector,
        freq::Float32)

    shape = build_normal(grid, μ_high, σ, a)
    source = RandomPosGaussianSource(grid, μ_low, μ_high, σ, a, shape, freq)
    Waves.reset!(source)
    return source
end

function (source::RandomPosGaussianSource)(t::AbstractVector{Float32})
    return source.shape .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
end

function RLBase.state(env::WaveEnv)
    ## only the total wave is observable
    w = cpu(cat(env.wave[:, :, 1, :], env.source.shape, dims = 3))
    x = imresize(w, env.resolution)
    return WaveEnvState(cpu(env.dim), build_tspan(env), x, cpu(env.design))
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

env = gpu(WaveEnv(dim; 
    # design_space = build_rectangular_grid_design_space(),
    design_space = Waves.build_triple_ring_design_space(),
    # source = Source(pulse, 1000.0f0),
    source = RandomPosGaussianSource(build_grid(dim), μ_low, μ_high, σ, a, 1000.0f0),
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

name = "variable_source_yaxis_x=-10.0"

path = mkpath(joinpath(DATA_PATH, name))
BSON.bson(joinpath(path, "env.bson"), env = cpu(env))

for i in 1:500
    ep = generate_episode!(policy, env)
    save(ep, joinpath(path, "episode$i.bson"))
end