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

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int)
    return [policy(env) for i in 1:horizon]
end

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int, shots::Int)
    return [build_action_sequence(policy, env, horizon) for i in 1:shots]
end

function Waves.build_tspan(ti::Float32, dt::Float32, steps::Int, horizon::Int)

    tspans = []

    for i in 1:horizon
        tspan = build_tspan(ti, dt, steps)
        push!(tspans, tspan)
        ti = tspan[end]
    end

    return hcat(tspans...)
end

function Waves.build_tspan(ti::Float32, dt::Float32, steps::Int, horizon::Int, shots::Int)
    tspan = build_tspan(time(env), dt, steps, horizon)
    return fill(tspan, shots)
end

function random_shooting(model::ScatteredEnergyModel, policy::AbstractPolicy, env::AbstractEnv, horizon::Int, shots::Int)
    s = state(env)
    states = gpu(fill(s, shots))
    actions = gpu(build_action_sequence(policy, env, horizon, shots))
    tspan = build_tspan(time(env), model.iter.dt, model.iter.steps, horizon)
    tspans = gpu(fill(tspan, shots))

    sigma_pred = model(states, actions, tspans)
end

struct RandomShooting <: AbstractPolicy
    policy::AbstractPolicy
    model::ScatteredEnergyModel
    horizon::Int
    shots::Int
    beta::Float32
end

function compute_action_penalty(a::DesignSequence)
    sum(sqrt.(sum(hcat(vec.(a)...) .^ 2, dims = 1)))
end

function compute_action_penalty(a::Vector{<: DesignSequence})
    f = _a -> hcat(vec.(_a)...)
    control_penalty = cat(map(f, a)..., dims = 3)
    return vec(sum(sqrt.(sum(control_penalty .^ 2, dims = 1)), dims = 2))
end

function (mpc::RandomShooting)(env::WaveEnv)

    s = gpu(fill(state(env), mpc.shots))
    a = gpu(build_action_sequence(mpc.policy, env, mpc.horizon, mpc.shots))
    t = gpu(build_tspan(time(env), env.dt, env.integration_steps, mpc.horizon, mpc.shots))
    
    pred_sigma = mpc.model(s, a, t)
    cost = vec(sum(pred_sigma, dims = 1)) .+ mpc.beta * compute_action_penalty(a)
    idx = argmin(cost)
    return a[idx][1]
end

Flux.device!(0)
main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_design_space=build_triple_ring_design_space_freq=1000.0"
pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=10000.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
no_pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=0.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Loading Models")
# pml_model = gpu(BSON.load(pml_model_path)[:model])
# no_pml_model = gpu(BSON.load(no_pml_model_path)[:model])
# testmode!(pml_model)
# testmode!(no_pml_model)

horizon = 20
shots = 256
beta = 1.0f0

mpc = RandomShooting(policy, pml_model, horizon, shots, beta)

s = gpu(state(env))
a = gpu(build_action_sequence(mpc.policy, env, mpc.horizon))
t = gpu(build_tspan(time(env), env.dt, env.integration_steps, mpc.horizon))

@time begin
        
    cost, back = Flux.pullback(a) do _a
        pred_sigma = mpc.model(s, _a, t)
        return sum(pred_sigma) .+ mpc.beta * compute_action_penalty(_a)
    end

    gs = back(one(cost))[1]
end

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(vec(pred_sigma)))
# save("pred_sigma.png", fig)

# gs = back(one(cost))

# for i in 1:6
#     episode = generate_episode_data(mpc, env)
#     save(episode, "mpc_results/mpc_episode_horizon=$(horizon)_$(i).bson")
# end
