include("dependencies.jl")

Flux.trainable(config::Scatterers) = (;config.pos)

function optimize_action(opt_state::NamedTuple, model::WaveControlModel, s::ScatteredWaveEnvState, a::AbstractDesign, steps::Int)
    println("optimize_action")

    for i in 1:steps
        cost, back = pullback(_a -> sum(model(s, _a)), a)
        gs = back(one(cost))[1]
        opt_state, a = Optimisers.update(opt_state, a, gs)
        println(cost)
    end

    return a
end

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