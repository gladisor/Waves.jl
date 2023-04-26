include("dependencies.jl")

function optimise_actions(opt_state, model::WaveMPC, s::ScatteredWaveEnvState, a::Vector{<: AbstractDesign})
    cost, back = pullback(_a -> sum(model(s, _a)[2:end, :]) + 500.0f0 * sum(norm.(vec.(_a))), a)
    gs = back(one(cost))[1]
    opt_state, a = Optimisers.update(opt_state, a, gs)
    return opt_state, a
end

function optimise_actions(model::WaveMPC, s::ScatteredWaveEnvState, a::Vector{<:AbstractDesign}, steps::Int)
    opt_state = Optimisers.setup(Optimisers.Descent(0.01), a)

    for i in 1:steps
        opt_state, a = optimise_actions(opt_state, model, s, a)
    end
    
    return a
end

model = BSON.load("results/radii/force/model.bson")[:model] |> gpu
env = BSON.load("results/radii/force/env.bson")[:env] |> gpu
reset!(env)
policy = RandomDesignPolicy(action_space(env))
episode = generate_episode_data(policy, env)

plot_sigma!(episode, path = "episode.png")
plot_episode_data!(episode, cols = 5, path = "data.png")

states = episode.states
actions = episode.actions
sigmas = episode.sigmas
tspans = episode.tspans

idx = 3

s = gpu(states[idx])
a = gpu(actions[idx:idx+1])

println("Initial Action")
display(a)

fig = Figure()
ax = Axis(fig[1, 1])

tspan = vec(hcat(tspans[idx:idx+1]...)[2:end, :])
sigma = vec(model(s, a)[2:end, :])
lines!(ax, tspan, cpu(sigma))

a = optimise_actions(model, s, a, 3)
sigma = vec(model(s, a)[2:end, :])
lines!(ax, tspan, cpu(sigma))

save("sigma.png", fig)

println("Optimized Action")
display(a)


# sigma = model(s, a)
# sigma = vec(sigma[2:end, :])
# tspan = vec(hcat(tspans[1:2]...)[2:end, :])

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, tspan, cpu(sigma), label = "Predicted")
# lines!(ax, tspan, vec(hcat(sigmas[idx:idx+1]...)[2:end, :]), label = "True")
# axislegend(ax)
# save("sigma.png", fig)