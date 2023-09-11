using Waves
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie
using ReinforcementLearning

dim = TwoDim(15.0f0, 256)
n = 10
μ = zeros(Float32, n, 2)
μ[:, 1] .= -10.0f0
μ[:, 2] .= range(-2.0f0, 2.0f0, n)

σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 0.3f0
pulse = build_normal(build_grid(dim), μ, σ, a)
source = Source(pulse, 1000.0f0)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = source,
    integration_steps = 20,
    actions = 10))

policy = RandomDesignPolicy(action_space(env))
render!(policy, env)

# signal = []
# for i in 1:100
#     @time env(policy(env))
#     push!(signal, cpu(env.signal))
# end

# signal = flatten_repeated_last_dim(permutedims(cat(signal..., dims = 3), (2, 1, 3)))
# fig = Figure()
# ax = Axis(fig[1 ,1])
# lines!(ax, signal[1, :])
# lines!(ax, signal[2, :])
# lines!(ax, signal[3, :])
# save("signal.png", fig)