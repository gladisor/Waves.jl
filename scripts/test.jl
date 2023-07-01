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

Flux.device!(1)

main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_design_space=build_triple_ring_design_space_freq=1000.0"
pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV9/horizon=20_nfreq=200_pml=10000.0_lr=0.0001_batchsize=32/epoch_60/model.bson")
no_pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV9/horizon=20_nfreq=200_pml=0.0_lr=0.0001_batchsize=32/epoch_40/model.bson")
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

pml_model = gpu(BSON.load(pml_model_path)[:model])
# no_pml_model = gpu(BSON.load(no_pml_model_path)[:model])

# opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(), Optimisers.Adam(1e-3))
# mpc = MPC(policy, pml_model, opt, 20, 50)

# @time episode1 = generate_episode_data(mpc, env)
# save(episode1, "pml_mpc_episode1.bson")
# @time episode2 = generate_episode_data(mpc, env)
# save(episode2, "pml_mpc_episode2.bson")
# @time episode3 = generate_episode_data(mpc, env)
# save(episode3, "pml_mpc_episode3.bson")
