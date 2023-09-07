using Waves
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie
using ReinforcementLearning

function build_normal(x::AbstractVector{Float32}, μ::AbstractVector{Float32}, σ::AbstractVector, a::AbstractVector)

    μ = permutedims(μ)
    σ = permutedims(σ)
    a = permutedims(a)

    f = (1.0f0 ./ (σ * sqrt(2.0f0 * π))) .* a .* exp.(- ((x .- μ) .^ 2) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 2), dims = 2)
end

function build_normal(x::AbstractArray{Float32, 3}, μ::AbstractMatrix, σ::AbstractVector, a::AbstractVector)
    μ = permutedims(μ[:, :, :, :], (3, 4, 2, 1))
    σ = permutedims(σ[:, :, :], (2, 3, 1))
    a = permutedims(a[:, :, :], (2, 3, 1))
    
    f = (1.0f0 ./ (2.0f0 * π * σ .^ 2)) .* a .* exp.(-dropdims(sum((x .- μ) .^ 2, dims = 3), dims = 3) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 3), dims = 3)
end

dim = TwoDim(15.0f0, 512)
n = 5
μ = zeros(Float32, n, 2)
μ[:, 1] .= -10.0f0
μ[:, 2] .= [-2.0f0, -1.0f0, 0.0f0, 1.0f0, 2.0f0]
σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 0.5f0
pulse = build_normal(build_grid(dim), μ, σ, a)
source = Source(pulse, 1000.0f0)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = source,
    integration_steps = 100,
    actions = 20))

policy = RandomDesignPolicy(action_space(env))
# render!(policy, env)
signal = []

for i in 1:100
    @time env(policy(env))
    push!(signal, cpu(env.signal))
end

signal = flatten_repeated_last_dim(permutedims(cat(signal..., dims = 3), (2, 1, 3)))

fig = Figure()
ax = Axis(fig[1 ,1])
lines!(ax, signal[1, :])
lines!(ax, signal[2, :])
lines!(ax, signal[3, :])
save("signal.png", fig)