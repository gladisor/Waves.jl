using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves
include("improved_model.jl")
include("plot.jl")

# function compute_cost(
#         sigma::AbstractMatrix{Float32}, 
#         action_sequence::Vector{<:AbstractDesign},
#         alpha::Float32)

#     return sum(sigma[2:end, :]) + alpha * sum(norm.(vec.(action_sequence)))
# end

# function optimise_actions(opt_state, model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign})
#     cost, back = pullback(_a -> compute_cost(model(s, _a), _a, 1.0f0), a)
#     gs = back(one(cost))[1]
#     opt_state, a = Optimisers.update(opt_state, a, gs)
#     return opt_state, a
# end

# function optimise_actions(opt::AbstractRule, model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, steps::Int)
#     opt_state = Optimisers.setup(opt, a)

#     for i in 1:steps
#         opt_state, a = optimise_actions(opt_state, model, s, a)
#     end

#     return a
# end

# struct MPC <: AbstractPolicy
#     policy::AbstractPolicy
#     model::WaveControlModel
#     opt::AbstractRule
#     horizon::Int
#     opt_steps::Int
# end

# Flux.@functor MPC

# function (mpc::MPC)(env::WaveEnv)
#     s = gpu(state(env))
#     a = gpu([mpc.policy(env) for _ in 1:mpc.horizon])
#     a = optimise_actions(mpc.opt, mpc.model, s, a, mpc.opt_steps)
#     return a[1]
# end

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, n::Int)
    return [policy(env) for i in 1:n]
end

function predict(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{ <: AbstractDesign})
    z_wave = model.wave_encoder(s)
    d = s.design

    z, d = propagate(model, z_wave, d, a[1])
    sigma1 = model.mlp(z)
    z_wave = z[:, [1, 2, 3], end]

    z, d = propagate(model, z_wave, d, a[2])
    sigma2 = model.mlp(z)
    z_wave = z[:, [1, 2, 3], end]

    z, d = propagate(model, z_wave, d, a[3])
    sigma3 = model.mlp(z)
    z_wave = z[:, [1, 2, 3], end]

    return hcat(sigma1, sigma2, sigma3)

    # z, d = propagate(model, z_wave, d, a[1])
    # sigma = model.mlp(z)
    # z_wave = z[:, [1, 2, 3], end]

    # for action in a[2:end]
    #     z, d = propagate(model, z_wave, d, action)
    #     sigma = hcat(sigma, model.mlp(z))
    #     z_wave = z[:, [1, 2, 3], end]
    # end

    # return sigma
end

function compute_cost(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{ <:AbstractDesign})
    pred_sigma = model(s, a)
    return sum(pred_sigma[2:end, :])
end

Flux.device!(2)
main_path = "data/single_cylinder_dataset"
data_path = joinpath(main_path, "episodes")
model_path = joinpath(main_path, "models/PML_HORIZON/nfreq=6_hsize=256_act=leakyrelu_gs=15.0_ele=512_hor=3_in=TotalWaveInput()_bs=10_pml=10000.0_lr=5.0e-6_decay=1.0/epoch_380/model.bson")

model = gpu(BSON.load(model_path)[:model])
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
policy = RandomDesignPolicy(action_space(env))
reset!(env)
time(env)

s = gpu(state(env))
a = gpu(build_action_sequence(policy, env, 3))

cost, back = Flux.pullback(_a -> compute_cost(model, s, _a), a)
gs = back(one(cost))[1]


# @time [env(policy(env)) for i in 1:20]
# fig = Figure()
# ax = Axis(fig[1, 1])

# for i in 1:3
#     a = gpu(build_action_sequence(policy, env, 10))
#     pred_sigma = cpu(model(s, a))
#     pred_sigma = vcat(pred_sigma[1], vec(pred_sigma[2:end, :]))
#     lines!(ax, pred_sigma)
# end

# save("sigma.png", fig)

# horizon = 3
# opt_steps = 2
# episodes = 5
# lr = 1e-3
# opt = Optimisers.Descent(lr)
# mpc = gpu(MPC(policy, model, opt, horizon, opt_steps))
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

# println("Rendering")
# @time render!(mpc, env, path = joinpath(model_path, "mpc.mp4"), seconds = 50.0f0)
# @time render!(policy, env, path = joinpath(model_path, "random.mp4"), seconds = 50.0f0)

# println("Running MPC")
# mpc_hook = TotalRewardPerEpisode()
# run(mpc, env, StopAfterEpisode(episodes), mpc_hook)

# println("Running Random")
# random_hook = TotalRewardPerEpisode()
# run(policy, env, StopAfterEpisode(episodes), random_hook)

# avg_mpc = mean(mpc_hook.rewards)
# avg_random = mean(random_hook.rewards)

# println("MPC: $avg_mpc")
# println("RANDOM: $avg_random")