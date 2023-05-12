include("dependencies.jl")

function compute_cost(
        sigma::AbstractMatrix{Float32}, 
        action_sequence::Vector{<:AbstractDesign},
        alpha::Float32)

    return sum(sigma[2:end, :]) + alpha * sum(norm.(vec.(action_sequence)))
end

function optimise_actions(opt_state, model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign})
    cost, back = pullback(_a -> compute_cost(model(s, _a), _a, 1.0f0), a)
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

Flux.@functor MPC

function (mpc::MPC)(env::WaveEnv)
    s = gpu(state(env))
    a = gpu([mpc.policy(env) for _ in 1:mpc.horizon])
    a = optimise_actions(mpc.opt, mpc.model, s, a, mpc.opt_steps)
    return a[1]
end

Flux.device!(1)

data_path = "data/hexagon_large_grid"
model_path = joinpath(
    data_path, 
    "models/speed_force_hypernet/h_size=512_elements=512_pml_width=5.0_pml_scale=5000.0_horizon=3_lr=1.0e-5_epochs=20_act=leakyrelu_num_train_episodes=50_decay=0.95/model9"
    )

# model = WaveControlModel(;path = model_path) |> gpu
# env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
policy = RandomDesignPolicy(action_space(env))
reset!(env)

horizon = 3
opt_steps = 2
episodes = 5
lr = 1e-3
opt = Optimisers.Descent(lr)
mpc = gpu(MPC(policy, model, opt, horizon, opt_steps))
# @time episode = generate_episode_data(policy, env)
# plot_sigma!(model, episode, path = "sigma.png")
# states, actions, tspans, sigmas = prepare_data(episode, 3)

# idx = 20
# s = gpu(states[idx])
# a = gpu(actions[idx])
# t = tspans[idx]
# sigma = gpu(sigmas[idx])
# pred_sigma_original = model(s, a)

# opt_state = Optimisers.setup(opt, a)

# alpha = 0.0f0

# cost, back = pullback(_a -> compute_cost(model(s, _a), _a, alpha), a)
# @time gs = back(one(cost))[1]
# opt_state, a = Optimisers.update(opt_state, a, gs)
# pred_sigma_opt_1 = model(s, a)

# cost, back = pullback(_a -> compute_cost(model(s, _a), _a, alpha), a)
# @time gs = back(one(cost))[1]
# opt_state, a = Optimisers.update(opt_state, a, gs)
# pred_sigma_opt_2 = model(s, a)

# cost, back = pullback(_a -> compute_cost(model(s, _a), _a, alpha), a)
# @time gs = back(one(cost))[1]
# opt_state, a = Optimisers.update(opt_state, a, gs)
# pred_sigma_opt_3 = model(s, a)

# fig = Figure()
# ax = Axis(fig[1, 1], aspect = 1.0f0)

# for i in axes(t, 2)
#     lines!(ax, t[:, i], cpu(pred_sigma_original[:, i]), color = :blue)
# end

# for i in axes(t, 2)
#     lines!(ax, t[:, i], cpu(pred_sigma_opt_1[:, i]), color = :red)
# end

# for i in axes(t, 2)
#     lines!(ax, t[:, i], cpu(pred_sigma_opt_2[:, i]), color = :orange)
# end

# for i in axes(t, 2)
#     lines!(ax, t[:, i], cpu(pred_sigma_opt_3[:, i]), color = :yellow)
# end

# save("sigma.png", fig)

println("Rendering")
@time render!(mpc, env, path = joinpath(model_path, "mpc.mp4"), seconds = 50.0f0)
@time render!(policy, env, path = joinpath(model_path, "random.mp4"), seconds = 50.0f0)

println("Running MPC")
mpc_hook = TotalRewardPerEpisode()
run(mpc, env, StopAfterEpisode(episodes), mpc_hook)

println("Running Random")
random_hook = TotalRewardPerEpisode()
run(policy, env, StopAfterEpisode(episodes), random_hook)

avg_mpc = mean(mpc_hook.rewards)
avg_random = mean(random_hook.rewards)

println("MPC: $avg_mpc")
println("RANDOM: $avg_random")