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

Flux.@functor MPC

function (policy::MPC)(env::WaveEnv)
    a = gpu([policy.policy(env) for _ in 1:policy.horizon])
    s = gpu(state(env))
    a = optimise_actions(policy.opt, policy.model, s, a, policy.opt_steps)
    return a[1]
end

path = "results/M=2_scaling_velocity_long/double_ring_cloak_h_channels=32_h_size=1024_latent_elements=512_n_mlp_layers=3_lr=0.0001_horizon=3_epochs=10"
model_path = joinpath(path, "model10.bson")
env_path = "data/M=2/env.bson"

model = BSON.load(model_path)[:model] |> gpu
env = BSON.load(env_path)[:env] |> gpu
policy = RandomDesignPolicy(action_space(env))
reset!(env)

[env(policy(env)) for i in 1:5]

s = cpu(state(env))

dim = cpu(env.dim)
fig = Figure(backgroundcolor = :transparent)
ax = Axis(fig[1, 1], aspect = 1.0f0, xlabel = "x (m)", ylabel = "y (m)", backgroundcolor = :transparent)
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, dim.y[1], dim.y[end])
heatmap!(ax, dim.x, dim.y, s.wave_total[:, :, 1], colormap = :ice)
mesh!(ax, s.design)
save("u.png", fig)

# horizon = 3
# opt_steps = 3
# episodes = 5
# lr = 1e-3


# # s = state(env)
# # a = [policy(env)] |> gpu

# # model.wave_encoder(s.wave_total)

# opt = Optimisers.Descent(lr)
# mpc = gpu(MPC(policy, model, opt, horizon, opt_steps))
# # @time a = mpc(env)
# # @time a = mpc(env)
# # @time a = mpc(env)
# # @time a = mpc(env)


# # @time for i in 1:20
# #     a = gpu([policy(env) for i in 1:3])
# #     model(s, a)
# # end

# # as = gpu([[policy(env) for i in 1:3] for i in 1:20])


# mpc_hook = TotalRewardPerEpisode()
# run(mpc, env, StopAfterEpisode(episodes), mpc_hook)

# random_hook = TotalRewardPerEpisode()
# run(policy, env, StopAfterEpisode(episodes), random_hook)

# avg_mpc = mean(mpc_hook.rewards)
# avg_random = mean(random_hook.rewards)

# println("MPC: $avg_mpc")
# println("RANDOM: $avg_random")

# @time render!(mpc, env, path = "mpc_M=3.mp4", seconds = 50.0f0)
# @time render!(policy, env, path = "random_M=3.mp4", seconds = 50.0f0)