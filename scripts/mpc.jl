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

# pml_model_path = joinpath(main_path, "models/RERUN/horizon=20_nfreq=200_pml=10000.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=10000.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
no_pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=0.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Loading Models")
pml_model = gpu(BSON.load(pml_model_path)[:model])
no_pml_model = gpu(BSON.load(no_pml_model_path)[:model])
testmode!(pml_model)
testmode!(no_pml_model)

episode = EpisodeData(path = joinpath(data_path, "episode900/episode.bson"))
s, a, t, sigma = prepare_data(episode, 2)#length(episode))

# idx = 20
idx = 22
s = gpu(s[idx])
a = gpu(a[idx])
t = gpu(t[idx])

# fig = Figure(resolution = (1920, 1080), fontsize = 45)
# ax = Axis(fig[1, 1], aspect = 1.0, xlabel = "Space (m)", ylabel = "Space (m)", title = "Wave State at Time = $(cpu(t)[1]) (s)")
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, dim.y[1], dim.y[end])

# heatmap!(ax, dim.x, dim.y, cpu(s.wave_total[:, :, end]), colormap = :ice)
# mesh!(ax, cpu(s.design))
# save("design.png", fig)

# latent_dim = cpu(pml_model.latent_dim)
# tspan = cpu(flatten_repeated_last_dim(t))

# z = cpu(generate_latent_solution(pml_model, s, a, t))
# # pml_z = permutedims(z[:, 2, :, 1] .- z[:, 1, :, 1])

# u = permutedims(z[:, 2, :, 1] .- z[:, 1, :, 1])
# c = permutedims(z[:, 6, :, 1] * pml_model.iter.dynamics.C0)

# fig = Figure(resolution = (1920, 1080), fontsize = 45)
# ax1 = Axis(fig[1, 1], ylabel = "Space (m)", title = L"u_{sc}^z(x, t)")
# hm1 = heatmap!(ax1, tspan, latent_dim.x, u, colormap = :ice)
# ax1.xticklabelsvisible = false
# ax1.xticksvisible = false
# ax1.titlesize =  60

# Colorbar(fig[1, 2], hm1, label = L"m")
# ax2 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Space (m)", title = L"c^z(x, t)")
# ax2.titlesize =  60
# hm2 = heatmap!(ax2, tspan, latent_dim.x, c, colormap = :turbid)
# Colorbar(fig[2, 2], hm2, label = L"m/s")
# save("c.png", fig)


fig = Figure(resolution = (1920, 1920))
ax = Axis(fig[1, 1], aspect = 1.0)

ax.xticklabelsvisible = false
ax.xticksvisible = false
ax.xgridvisible = false

ax.yticklabelsvisible = false
ax.yticksvisible = false
ax.ygridvisible = false

ax.topspinevisible = false
ax.rightspinevisible = false
ax.leftspinevisible = false
ax.bottomspinevisible = false

mesh!(ax, cpu(s.design))
save("design.png", fig)

# fig = Figure(resolution = (1920, 1080))

# ax1 = Axis(
#     fig[1, 1], #, aspect = (1, 5, 1),
#     title = "Displacement"
# )
# heatmap!(ax1, tspan, latent_dim.x, u, colormap = :ice)

# ax2 = Axis(
#     fig[2, 1], #aspect = (1, 5, 1),
#     title = "Wave Speed"
#     )
# # zlims!(ax2, 0.0f0, pml_model.iter.dynamics.C0)
# heatmap!(ax2, tspan, latent_dim.x, c, colormap = :delta)
# save("c.png", fig)

# z = cpu(generate_latent_solution(no_pml_model, s, a, t))
# no_pml_z = permutedims(z[:, 2, :, 1] .- z[:, 1, :, 1])

# fig = Figure(fontsize = 30)
# ax1 = Axis(fig[1, 1], ylabel = "Space (m)", title = "PML")
# ax1.xticklabelsvisible = false
# heatmap!(ax1, tspan, latent_dim.x, pml_z, colorrange = )
# ax2 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Space (m)", title = "No PML")
# heatmap!(ax2, tspan, latent_dim.x, no_pml_z)
# save("latent.png", fig)

# horizon = 200
# s = gpu(state(env))
# a = gpu(build_action_sequence(policy, env, horizon))
# t = gpu(build_tspan(time(env), env.dt, env.integration_steps, horizon))

######################### Wave Image Plots
# @time env(policy(env))
# s1 = state(env)
# @time env(policy(env))
# s2 = state(env)
# @time env(policy(env))
# s3 = state(env)
# @time env(policy(env))
# s4 = state(env)
# @time env(policy(env))
# s5 = state(env)
# @time env(policy(env))
# s6 = state(env)
# @time env(policy(env))
# s7 = state(env)
# @time env(policy(env))
# s8 = state(env)
# @time env(policy(env))
# s9 = state(env)
# @time env(policy(env))
# s10 = state(env)

# wave_image = cpu(s10.wave_total)
# dim = cpu(env.dim)

# colorrange = (-0.5, 0.5)

# fig = Figure(resolution = (1920, 1080))
# ax = Axis(fig[1, 1], aspect = 1.0f0)
# heatmap!(ax, dim.x, dim.y, wave_image[:, :, 1], colormap = "ice", colorrange = colorrange)
# mesh!(ax, cpu(s8.design))
# ax.xticklabelsvisible = false
# ax.xticksvisible = false
# ax.yticklabelsvisible = false
# ax.yticksvisible = false
# save("wave_image1.png", fig)

# fig = Figure(resolution = (1920, 1080))
# ax = Axis(fig[1, 1], aspect = 1.0f0)
# heatmap!(ax, dim.x, dim.y, wave_image[:, :, 2], colormap = "ice", colorrange = colorrange)
# mesh!(ax, cpu(s9.design))
# ax.xticklabelsvisible = false
# ax.xticksvisible = false
# ax.yticklabelsvisible = false
# ax.yticksvisible = false
# save("wave_image2.png", fig)

# fig = Figure(resolution = (1920, 1080))
# ax = Axis(fig[1, 1], aspect = 1.0f0)
# heatmap!(ax, dim.x, dim.y, wave_image[:, :, 3], colormap = "ice", colorrange = colorrange)
# mesh!(ax, cpu(s10.design))
# ax.xticklabelsvisible = false
# ax.xticksvisible = false
# ax.yticklabelsvisible = false
# ax.yticksvisible = false
# save("wave_image3.png", fig)

# tspan = cpu(flatten_repeated_last_dim(t))
# sigma = flatten_repeated_last_dim(sigma[1])
# pml_pred_sigma = cpu(vec(pml_model(s, a, t)))
# no_pml_pred_sigma = cpu(vec(no_pml_model(s, a, t)))
############# END WAVE IMAGE PLOTS


############# Pred Energy Plot
# fig = Figure(fontsize = 30)
# ax = Axis(fig[1, 1], 
#     # title = "Scattered Energy With Random Control",
#     xlabel = "Time (s)",
#     ylabel = "Scattered Energy", 
#     )

# lines!(ax, tspan, sigma, label = "True Sigma", color = (:blue, 1.0))
# lines!(ax, tspan, pml_pred_sigma, label = "PML", color = (:orange, 0.5))
# lines!(ax, tspan, no_pml_pred_sigma, label = "No PML", color = (:green, 0.5))
# axislegend(ax, position = :rb)
# save("sigma.png", fig)
############ END ENERGY PLOT

################## RUN MPC EPISODES
# horizon = 50
# shots = 128
# beta = 1.0f0
# mpc = RandomShooting(policy, pml_model, horizon, shots, beta)
# env.actions = 100
# render!(mpc, env, path = "vid.mp4", seconds = env.actions * 0.5f0)

# s = gpu(state(env))
# a = gpu(build_action_sequence(mpc.policy, env, mpc.horizon))
# t = gpu(build_tspan(time(env), env.dt, env.integration_steps, mpc.horizon))

# @time begin
#     cost, back = Flux.pullback(a) do _a
#         pred_sigma = mpc.model(s, _a, t)
#         return sum(pred_sigma) .+ mpc.beta * compute_action_penalty(_a)
#     end
#     gs = back(one(cost))[1]
# end

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(vec(pred_sigma)))
# save("pred_sigma.png", fig)
# gs = back(one(cost))

# for i in 1:6
#     episode = generate_episode_data(mpc, env)
#     save(episode, "mpc_results/pml_mpc_episode_horizon=$(horizon)_$(i).bson")
# end