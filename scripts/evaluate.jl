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

# path = "results/M=6_core=1.0/h_channels=32_h_size=1024_latent_elements=512_n_mlp_layers=3_lr=0.0001_horizon=3_epochs=10/transfer_lr=8.0e-5_horizon=5/"
# model_path = joinpath(path, "model10.bson")
# env_path = "data/M=6_as=1.0/env.bson"

data_path = "data/M=6_as=1.0_normalized"
model_path = joinpath(data_path, "models/hypernet_testing/h_channels=32_h_size=512_latent_elements=256_latent_pml_width=3.0_latent_pml_scale=5000.0_horizon=3_lr=5.0e-5_epochs=10_act=gelu/model6")
model = WaveControlModel(;path = model_path) |> gpu
env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu

policy = RandomDesignPolicy(action_space(env))
reset!(env)

horizon = 3
opt_steps = 2
episodes = 5
lr = 1e-4

opt = Optimisers.Descent(lr)
mpc = gpu(MPC(policy, model, opt, horizon, opt_steps))

@time render!(mpc, env, path = joinpath(model_path, "mpc.mp4"), seconds = 50.0f0)
@time render!(policy, env, path = joinpath(model_path, "random.mp4"), seconds = 50.0f0)

mpc_hook = TotalRewardPerEpisode()
run(mpc, env, StopAfterEpisode(episodes), mpc_hook)

random_hook = TotalRewardPerEpisode()
run(policy, env, StopAfterEpisode(episodes), random_hook)

avg_mpc = mean(mpc_hook.rewards)
avg_random = mean(random_hook.rewards)

println("MPC: $avg_mpc")
println("RANDOM: $avg_random")