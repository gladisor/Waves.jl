using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves
using DataFrames
using CSV
include("improved_model.jl")
include("plot.jl")

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, n::Int)
    return [policy(env) for i in 1:n]
end

total_energy(sigma::AbstractMatrix{Float32}) = sum(sigma[2:end, :])

function compute_gradient(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, f::Function)
    cost, back = Flux.pullback(_a -> f(model(s, _a)), a)
    gs = back(one(cost))[1]
    return cost, gs
end

function optimise_actions(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, f::Function, opt::AbstractRule, n::Int)
    a = deepcopy(a)
    opt_state = Optimisers.setup(opt, a)

    println()

    for i in 1:n
        cost, gs = compute_gradient(model, s, a, f)
        opt_state, a = Optimisers.update(opt_state, a, gs)
        println(cost)
    end

    return a
end

struct MPC <: AbstractPolicy
    policy::AbstractPolicy
    model::ScatteredEnergyModel
    opt::AbstractRule
    horizon::Int
    opt_steps::Int
end

function (mpc::MPC)(env::WaveEnv)
    s = gpu(state(env))
    a = gpu([mpc.policy(env) for _ in 1:mpc.horizon])
    a = optimise_actions(mpc.model, s, a, total_energy, mpc.opt, mpc.opt_steps)
    return a[1]
end

function flatten_series(series::AbstractVector{<: AbstractVector{Float32}})
    return vcat(series[1], [series[i][2:end] for i in 2:length(series)]...)
end

function flatten_series(series::AbstractMatrix{Float32})
    return flatten_series([series[:, i] for i in axes(series, 2)])
end

function measure_error(model::ScatteredEnergyModel, data::Tuple)
    states, actions, tspans, sigmas = data

    error = 0.0f0

    for i in 1:length(states)
        s, a, t, sigma = states[i], actions[i], tspans[i], sigmas[i]
        s = gpu(s)
        a = gpu(a)
        sigma = gpu(sigma)
        error += Flux.mse(model(s, a), sigma)

        println(i)
    end

    return error / length(states)
end

function episode_cost(episode::EpisodeData)
    return sum(episode.sigmas[1]) + sum([sum(episode.sigmas[i][2:end]) for i in 2:length(episode)])
end

Flux.device!(1)

# model_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset/models/FullState/nfreq=6_hsize=256_act=leakyrelu_gs=15.0_ele=512_hor=3_in=TotalWaveInput()_bs=10_pml=10000.0_lr=5.0e-6_decay=1.0/epoch_840/model.bson"
# model_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset/models/FullState/nfreq=6_hsize=256_act=leakyrelu_gs=15.0_ele=512_hor=3_in=TotalWaveInput()_bs=10_pml=0.0_lr=5.0e-6_decay=1.0/epoch_840/model.bson"
model_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset/models/TESTINGCNN/nfreq=6_hsize=256_act=leakyrelu_gs=15.0_ele=512_hor=3_in=TotalWaveInput()_bs=10_pml=0.0_lr=5.0e-6_decay=1.0/epoch_730/model.bson"
env_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset/env.bson"
model = gpu(BSON.load(model_path)[:model])
env = gpu(BSON.load(env_path)[:env])
policy = RandomDesignPolicy(action_space(env))
reset!(env)

horizon = 3
opt_steps = 3
lr = 0.01
mpc = MPC(policy, model, Optimisers.Descent(lr), horizon, opt_steps)
# states, actions, tspans, sigmas = prepare_data(random_episode1, horizon)

# idx = 30
# s, a, t, sigma = gpu((states[idx], actions[idx], tspans[idx], sigmas[idx]))

# model(s, a)
# a = gpu([mpc.policy(env) for _ in 1:mpc.horizon])
# a_star = optimise_actions(mpc.model, s, a, total_energy, mpc.opt, mpc.opt_steps)

# fig = Figure()
# ax = Axis(fig[1, 1])
# # lines!(ax, cpu(flatten_series(sigma)), color = :blue)
# lines!(ax, cpu(flatten_series(model(s, a))), color = :blue)
# lines!(ax, cpu(flatten_series(model(s, a_star))), color = :orange)
# save("sigma.png", fig)

## evaluating random baseline agent
# @time random_episode1 = generate_episode_data(policy, env)
# @time random_episode2 = generate_episode_data(policy, env)
# @time random_episode3 = generate_episode_data(policy, env)
# @time data = prepare_data([random_episode1, random_episode2, random_episode3], horizon)
# @time error = measure_error(model, data)
# random_cost1 = episode_cost(random_episode1)
# random_cost2 = episode_cost(random_episode2)
# random_cost3 = episode_cost(random_episode3)

# idx = 70
# s = gpu(states[idx])
# a = gpu(actions[idx])
# tspan = tspans[idx]
# sigma = sigmas[idx]

# episode1 = EpisodeData(path = "episode1.bson")
# episode2 = EpisodeData(path = "episode2.bson")
# episode3 = EpisodeData(path = "episode3.bson")

## running and evaluating mpc in the environment
@time episode1 = generate_episode_data(mpc, env)
cost1 = episode_cost(episode1)
println(cost1)

@time episode2 = generate_episode_data(mpc, env)
cost2 = episode_cost(episode2)
println(cost2)

@time episode3 = generate_episode_data(mpc, env)
cost3 = episode_cost(episode3)
println(cost3)

tspan = flatten_series(episode1.tspans)
sigma1 = flatten_series(episode1.sigmas)
sigma2 = flatten_series(episode2.sigmas)
sigma3 = flatten_series(episode3.sigmas)

data = DataFrame(tspan = tspan, sigma1 = sigma1, sigma2 = sigma2, sigma3 = sigma3)
CSV.write("pml=off,latent=15m,horizon=3/data.csv", data)