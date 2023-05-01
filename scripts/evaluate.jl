include("dependencies.jl")

function optimise_actions(opt_state, model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign})
    cost, back = pullback(_a -> sum(model(s, _a)[2:end, :]) + 500.0f0 * sum(norm.(vec.(_a))), a)
    gs = back(one(cost))[1]
    opt_state, a = Optimisers.update(opt_state, a, gs)
    return opt_state, a
end

function optimise_actions(opt::AbstractRule, model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, steps::Int)
    opt_state = Optimisers.setup(opt, a)

    for i in 1:steps
        opt_state, a = optimise_actions(opt_state, model, s, a)
    end

    return a
end

struct MPC <: AbstractPolicy
    policy::AbstractPolicy
    model::WaveControlModel
    opt::AbstractRule
    horizon::Int
    opt_steps::Int
end

function (policy::MPC)(env::WaveEnv)
    a = gpu([policy.policy(env) for _ in 1:policy.horizon])
    s = gpu(state(env))
    a = optimise_actions(policy.opt, policy.model, s, a, policy.opt_steps)
    return a[1]
end

path = "results/cloak_h_channels=16_h_size=512_latent_elements=1024_n_mlp_layers=2_horizon=3_lr=0.0001"
model_path = joinpath(path, "model.bson")
env_path = "data/env.bson"

model = BSON.load(model_path)[:model] |> gpu
env = BSON.load(env_path)[:env] |> gpu
reset!(env)
policy = RandomDesignPolicy(action_space(env))

horizon = 3
opt_steps = 3
episodes = 10
lr = 1e-3

opt = Optimisers.Descent(lr)
mpc = MPC(policy, model, opt, horizon, opt_steps)
a = mpc(env)

mpc_hook = TotalRewardPerEpisode()
run(mpc, env, StopAfterEpisode(episodes), mpc_hook)

random_hook = TotalRewardPerEpisode()
run(policy, env, StopAfterEpisode(episodes), random_hook)

avg_mpc = mean(mpc_hook.rewards)
avg_random = mean(random_hook.rewards)

mpc_episode = generate_episode_data(mpc, env)
plot_episode_data!(mpc_episode, cols = 5, path = "mpc_episode.png")

random_episode = generate_episode_data(policy, env)
plot_episode_data!(random_episode, cols = 5, path = "random_episode.png")

println("MPC: $avg_mpc")
println("RANDOM: $avg_random")