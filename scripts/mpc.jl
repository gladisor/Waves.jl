include("dependencies.jl")

Flux.trainable(config::Scatterers) = (;config.pos)

model = BSON.load("results/PercentageWaveControlModel/model.bson")[:model] |> gpu
env = BSON.load("results/WaveControlModel/env.bson")[:env] |> gpu
reset!(env)
policy = RandomDesignPolicy(action_space(env))

@time data = generate_episode_data(policy, env)
plot_episode_data!(data, cols = 5, path = "data.png")
plot_sigma!(model, data, path = "sigma.png")

idx = 5
s = gpu(data.states[idx])
a = gpu(data.actions[idx])
sigma = data.sigmas[idx]
tspan = data.tspans[idx]
println("True σ = ", sum(sigma))

opt_state = Optimisers.setup(Optimisers.Adam(0.01), a)

# display(a)
# a = optimize_action(opt_state, model, s, a, 10)
# display(a)

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0, title = "Variation of predicted sigma", xlabel = "Time (s)", ylabel = "Total Scattered Energy")
for i in 1:10
    lines!(ax, tspan, cpu(model(s, policy(env))))
end
save("sigma_variation.png", fig)