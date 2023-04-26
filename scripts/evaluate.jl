include("dependencies.jl")

function optimise_actions(opt_state, model::WaveMPC, s::ScatteredWaveEnvState, a::Vector{<: AbstractDesign})
    cost, back = pullback(_a -> sum(model(s, _a)[2:end, :]) + 500.0f0 * sum(norm.(vec.(_a))), a)
    gs = back(one(cost))[1]
    opt_state, a = Optimisers.update(opt_state, a, gs)
    return opt_state, a
end

function optimise_actions(opt::AbstractRule, model::WaveMPC, s::ScatteredWaveEnvState, a::Vector{<: AbstractDesign}, steps::Int)
    opt_state = Optimisers.setup(opt, a)

    for i in 1:steps
        opt_state, a = optimise_actions(opt_state, model, s, a)
    end

    return a
end

struct MPC <: AbstractPolicy
    policy::AbstractPolicy
    model::WaveMPC
    opt::AbstractRule
    horizon::Int
    opt_steps::Int
end

function (policy::MPC)(env::ScatteredWaveEnv)
    a = gpu([policy.policy(env) for _ in 1:policy.horizon])
    s = gpu(state(env))
    a = optimise_actions(policy.opt, policy.model, s, a, policy.opt_steps)
    return a[1]
end

model = BSON.load("results/radii/long/model.bson")[:model] |> gpu
env = BSON.load("results/radii/long/env.bson")[:env] |> gpu
reset!(env)
policy = RandomDesignPolicy(action_space(env))

opt = Optimisers.Descent(0.001)
mpc = MPC(policy, model, opt, 2, 3)
a = mpc(env)
run(mpc, env, StopWhenDone(), TotalRewardPerEpisode())

# episode = generate_episode_data(policy, env)

# plot_sigma!(episode, path = "episode.png")
# plot_sigma!(model, episode, path = "episode_full.png")
# plot_episode_data!(episode, cols = 5, path = "data.png")

# states = episode.states
# actions = episode.actions
# sigmas = episode.sigmas
# tspans = episode.tspans

# idx = 8

# s = gpu(states[idx])
# a = gpu(actions[idx:idx+1])

# println("Initial Action")
# display(a)

# fig = Figure()
# ax = Axis(fig[1, 1])

# tspan = vec(hcat(tspans[idx:idx+1]...)[2:end, :])
# sigma = vec(model(s, a)[2:end, :])
# lines!(ax, tspan, cpu(sigma))

# opt = Optimisers.Descent(0.001)
# a = optimise_actions(opt, model, s, a, 3)

# sigma = vec(model(s, a)[2:end, :])
# lines!(ax, tspan, cpu(sigma))

# save("sigma.png", fig)

# println("Optimized Action")
# display(a)