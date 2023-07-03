println("Importing Packages")
using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves

Flux.CUDA.allowscalar(false)
include("improved_model.jl")
include("plot.jl")

function plot_latent_energy!(model::ScatteredEnergyModel, s::WaveEnvState, a::DesignSequence; path::String)
    z = generate_latent_solution(model, s, a)

    inc = z[:, 1, :, 1]
    tot = z[:, 2, :, 1]

    inc_energy = vec(cpu(sum(inc .^ 2, dims = 1)))
    tot_energy = vec(cpu(sum(tot .^ 2, dims = 1)))
    sc_energy = vec(cpu(sum((tot .- inc) .^ 2, dims = 1)))

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, inc_energy, label = "Incident")
    lines!(ax, tot_energy, label = "Total")
    lines!(ax, sc_energy, label = "Scattered")
    axislegend(ax)
    save(path, fig)
end

function optimise_actions(model::ScatteredEnergyModel, s::WaveEnvState, a::DesignSequence; opt, n::Int, beta::Float32)

    testmode!(model)

    a_star = deepcopy(a)
    opt_state = Optimisers.setup(opt, a_star)

    costs = []

    for i in 1:n

        cost, back = Flux.pullback(a_star) do _a
            sum(pml_model(s, _a)) + beta * sum(Flux.norm.(vec.(_a)))
        end
    
        gs = back(one(cost))[1]
        opt_state, a_star = Optimisers.update(opt_state, a_star, gs)
        println("Cost: $(cost)")
        push!(costs, cost)
    end

    # return costs, a_star
    return a_star
end

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, n::Int)
    return [policy(env) for i in 1:n]
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
    a = gpu(build_action_sequence(mpc.policy, env, mpc.horizon))
    a = optimise_actions(mpc.model, s, a, opt = mpc.opt, n = mpc.opt_steps, beta = 1.0f0)
    return a[1]
end

Flux.device!(0)

main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_design_space=build_triple_ring_design_space_freq=1000.0"
pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=10000.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
no_pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=0.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
data_path = joinpath(main_path, "episodes")

horizon = 20
batchsize = 32

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))


train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 998:1000])
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize, partial = false)
states, actions, tspans, sigmas = gpu(first(train_loader))
# s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]

pml_model = gpu(BSON.load(pml_model_path)[:model])
no_pml_model = gpu(BSON.load(no_pml_model_path)[:model])
testmode!(pml_model)
testmode!(no_pml_model)

pml_sigmas = pml_model(states, actions, tspans)
no_pml_sigmas = no_pml_model(states, actions, tspans)

y = flatten_repeated_last_dim(sigmas)
pml_loss = Flux.mse(pml_sigmas, y)
no_pml_loss = Flux.mse(no_pml_sigmas, y)

println("PML Loss: $(pml_loss)")
println("NO PML Loss: $(no_pml_loss)")

ts = flatten_repeated_last_dim(tspans)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, cpu(ts[:, 1]), cpu(pml_sigmas[:, 1]), label = "PML", color = :orange)
lines!(ax, cpu(ts[:, 1]), cpu(no_pml_sigmas[:, 1]), label = "NO PML", color = :green)
lines!(ax, cpu(ts[:, 1]), cpu(y[:, 1]), label = "True", color = :blue)
axislegend(ax)
save("sigma.png", fig)



# opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(), Optimisers.Adam(1e-3))
# mpc = MPC(policy, pml_model, opt, 20, 50)

# @time episode1 = generate_episode_data(mpc, env)
# save(episode1, "pml_mpc_episode1.bson")
# @time episode2 = generate_episode_data(mpc, env)
# save(episode2, "pml_mpc_episode2.bson")
# @time episode3 = generate_episode_data(mpc, env)
# save(episode3, "pml_mpc_episode3.bson")
