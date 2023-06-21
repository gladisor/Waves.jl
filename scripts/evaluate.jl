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

total_energy(sigma::AbstractMatrix{Float32}) = sum(sigma[2:end])

function compute_actions_gradient(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, f::Function)
    cost, back = Flux.pullback(_a -> f(model(s, _a)), a)
    gs = back(one(cost))[1]
    return cost, gs
end

function optimise_actions(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, f::Function, opt::AbstractRule, n::Int)
    a = deepcopy(a)
    opt_state = Optimisers.setup(opt, a)

    println()

    for i in 1:n
        cost, gs = compute_actions_gradient(model, s, a, f)
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

Flux.device!(3)

main_path = "data/triple_ring_dataset"
model_path = joinpath(main_path, "models/full_cnn/pml_width=10.0_pml_scale=10000.0_k_size=2_latent_elements=1024/epoch_400/model.bson")
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = BSON.load(joinpath(data_path, "env.bson"))[:env]
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

random_episode1 = EpisodeData(path = joinpath(data_path, "episode1/episode.bson"))
random_episode2 = EpisodeData(path = joinpath(data_path, "episode2/episode.bson"))
random_episode3 = EpisodeData(path = joinpath(data_path, "episode3/episode.bson"))

tspan = flatten_series(random_episode1.tspans)
sigma1 = flatten_series(random_episode1.sigmas)
sigma2 = flatten_series(random_episode2.sigmas)
sigma3 = flatten_series(random_episode3.sigmas)
data = DataFrame(tspan = tspan, sigma1 = sigma1, sigma2 = sigma2, sigma3 = sigma3)
# CSV.write("random.csv", data)
random_avg = (data[!, :sigma1] .+ data[!, :sigma2] .+ data[!, :sigma3]) ./ 3.0f0
model = gpu(BSON.load(model_path)[:model])

# mpc_episode1 = EpisodeData(path = "mpc_episode1.bson")
# mpc_episode2 = EpisodeData(path = "mpc_episode2.bson")
# mpc_episode3 = EpisodeData(path = "mpc_episode3.bson")

# tspan = flatten_series(mpc_episode1.tspans)
# sigma1 = flatten_series(mpc_episode1.sigmas)
# sigma2 = flatten_series(mpc_episode2.sigmas)
# sigma3 = flatten_series(mpc_episode3.sigmas)

# mpc_avg = (sigma1 .+ sigma2 .+ sigma3) ./ 3.0f0

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, data[!, :tspan], random_avg)
# lines!(ax, data[!, :tspan], mpc_avg)
# save("compare.png", fig)

horizon = 5
opt_steps = 30
lr = 0.01
idx = 30

# states, actions, tspans, sigmas = prepare_data(random_episode1, horizon)
# s, a, t, sigma = gpu((states[idx], actions[idx], tspans[idx], sigmas[idx]))
# model = gpu(BSON.load(model_path)[:model])
mpc = MPC(policy, model, Optimisers.Adam(lr), horizon, opt_steps)

# a = gpu([mpc.policy(env) for _ in 1:mpc.horizon])
# a_star = optimise_actions(mpc.model, s, a, total_energy, mpc.opt, mpc.opt_steps)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(vec(model(s, a))), color = :blue)
# lines!(ax, cpu(vec(model(s, a_star))), color = :orange)
# save("sigma.png", fig)

@time episode1 = generate_episode_data(mpc, env)
save(episode1, "mpc_episode1.bson")
@time episode2 = generate_episode_data(mpc, env)
save(episode2, "mpc_episode2.bson")
@time episode3 = generate_episode_data(mpc, env)
save(episode3, "mpc_episode3.bson")

# cost1 = episode_cost(episode1)
# cost2 = episode_cost(episode2)
# cost3 = episode_cost(episode3)

# tspan = flatten_series(episode1.tspans)
# sigma1 = flatten_series(episode1.sigmas)
# sigma2 = flatten_series(episode2.sigmas)
# sigma3 = flatten_series(episode3.sigmas)
# data = DataFrame(tspan = tspan, sigma1 = sigma1)
#, sigma2 = sigma2, sigma3 = sigma3)
# CSV.write("mpc.csv", data)
