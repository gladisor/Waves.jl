using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
using ReinforcementLearning

println("Loaded Packages")
Flux.device!(2)
display(Flux.device())

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")
include("dataset.jl")
include("../src/model/node.jl")
include("../src/model/mpc.jl")

function compute_energy_cost(model::AcousticEnergyModel, s, a, t)
    y_hat = model(s, a, t)
    return vec(sum(y_hat[:, 3, :], dims = 1))
end

function compute_energy_cost(model::NODEEnergyModel, s, a, t)
    y_hat = model(s, a, t)
    return vec(sum(y_hat, dims = 1))
end

function Waves.build_tspan(mpc::RandomShooting, env::WaveEnv)
    return hcat(fill(
        build_tspan(time(env), env.dt, env.integration_steps * mpc.horizon),
        mpc.shots)...)
end

function (mpc::RandomShooting)(env::WaveEnv)
    s = gpu(fill(state(env), mpc.shots))
    a = build_action_sequence(mpc.policy, env, mpc.horizon, mpc.shots)
    t = build_tspan(mpc, env) |> gpu

    energy = compute_energy_cost(mpc.model, s, a, t)
    penalty = compute_action_cost(a)
    cost = energy .+ mpc.alpha * penalty
    idx = argmin(cost)
    return a[1, idx]
end

env = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/env.bson")[:env])

pml_checkpoint = 2920
pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal/checkpoint_step=$pml_checkpoint/checkpoint.bson")[:model])

policy = RandomDesignPolicy(action_space(env))

horizon = 20
shots = 256
alpha = 1.0

mpc = RandomShooting(policy, pml_model, horizon, shots, alpha)
@time a = mpc(env)