using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
using ReinforcementLearning
using Interpolations: linear_interpolation
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(3)
display(Flux.device())
include("../src/masks.jl")

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int)
    return [policy(env) for i in 1:horizon]
end

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int, shots::Int)
    return hcat([build_action_sequence(policy, env, horizon) for i in 1:shots]...)
end

struct RandomShooting <: AbstractPolicy
    policy::AbstractPolicy
    model
    horizon::Int
    shots::Int
    alpha::Float32
    focusing::Bool
end

function compute_action_cost(a::Matrix{<: AbstractDesign})
    x = cat([hcat(vec.(a)[:, i]...) for i in axes(a, 2)]..., dims = 3)
    return vec(sum(sqrt.(sum(x .^ 2, dims = 1)), dims = 2))
end

function compute_energy_cost(model, s, a, t)
    y_hat = model(s, a, t)
    return vec(sum(y_hat[:, 3, :], dims = 1))
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
    if focusing
        cost = energy .- mpc.alpha * penalty
        idx = argmax(cost)
    else
        cost = energy .+ mpc.alpha * penalty
        idx = argmin(cost)
    end
    return a[1, idx]
end

function compute_energy_cost(model::WaveControlPINN, s, a, t)
    @time y_hat_1 = model(s[1:64], a[:, 1:64], t[:, 1:64])
    @time y_hat_2 = model(s[65:128], a[:, 65:128], t[:, 65:128])
    @time y_hat_3 = model(s[129:192], a[:, 129:192], t[:, 129:192])
    @time y_hat_4 = model(s[193:end], a[:, 193:end], t[:, 193:end])
    y_hat = vcat(y_hat_1, y_hat_2, y_hat_3, y_hat_4)
    return vec(sum(y_hat[:, 3, :], dims = 1))
end


function build_interpolator(
        policy::AbstractPolicy,
        env::WaveEnv;
        reset::Bool = true, 
        field::Symbol = :tot)

    @assert field ∈ [:tot, :inc, :sc]

    tspans = []
    interps = DesignInterpolator[]

    x = []
    σ = []

    if reset
        RLBase.reset!(env)
    end

    masks = cat(create_patches(700, 350)..., dims = 3)

    while !is_terminated(env)
        tspan, interp, u_tot, u_inc = cpu(env(policy(env), masks))

        push!(tspans, tspan)
        push!(interps, interp)

        if field == :tot
            push!(x, u_tot)
        elseif field == :inc
            push!(x, u_inc)
        elseif field == :sc
            push!(x, u_tot .- u_inc)
        end

        push!(σ, cpu(env.signal))

        println("time step: $(env.time_step)  ||  free memory [MB]: $(Sys.free_memory()/2^20)")
    end

    tspan = flatten_repeated_last_dim(hcat(tspans...))
    println("tspan flattened.  ||  free memory [MB]: $(Sys.free_memory()/2^20)")
    x = flatten_repeated_last_dim(cat(x..., dims = 4))
    println("x flattened.  ||  free memory [MB]: $(Sys.free_memory()/2^20)")
    x = linear_interpolation(tspan, Flux.unbatch(x))
    println("linear interpolation done.  ||  free memory [MB]: $(Sys.free_memory()/2^20)")
    return x, interps, σ
end


function create_data(env, mpc, frames, tspan, output_path, title)
    x, interps, σ = build_interpolator(mpc, env, reset = false, field = :sc)
    signal = flatten_repeated_last_dim(cat(transpose.(σ)..., dims = 3))

    BSON.bson(output_path, 
        x=[x(tspan[i]) .^ 2 for i in 1:frames], 
        interps=interps, 
        signal=signal,
        title=title)

    return output_path
end

function log_message(logpath, message)
    logfile = open(joinpath(logpath, "log.txt"), "a")
    println(logfile, "$message")
    close(logfile)
end

# dataset_name = "dataset_pos_adjustment_masked"
# dataset_name = "pos_adjustment_masked_M=2"
# dataset_name = "full_adjustment_masked_signals_M=2"
dataset_name = "pos_adjustment_masked_signals_M=4"
DATA_PATH = "scratch/$dataset_name"
@time env = gpu(BSON.load(joinpath(DATA_PATH, "env.bson"))[:env])
dim = cpu(env.dim)
focusing = true

jobid = 42880
model_name = "AEM_focusing_jobID=$jobid"
checkpoint_step = 10000

MODEL_PATH = "scratch/$dataset_name/models/$model_name/checkpoint_step=$checkpoint_step/checkpoint.bson"
model = gpu(BSON.load(MODEL_PATH)[:model])
policy = RandomDesignPolicy(action_space(env))

node_jobid = 42883
node_model_name = "NODE_focusing_jobID=$node_jobid"
NODE_MODEL_PATH = "scratch/$dataset_name/models/$node_model_name/checkpoint_step=$checkpoint_step/checkpoint.bson"
node_model = gpu(BSON.load(NODE_MODEL_PATH)[:model])

# output_folder = mkpath("$(jobid)AEM_$(node_jobid)NODE_$(checkpoint_step)_M=2_focus_20.mpc")
output_folder = mkpath("M=4_focus_shots=512.mpc")
log_message(output_folder, "dataset_name: $(dataset_name)\nAEM: $model_name\nNODE: $node_model_name\ncheckpoint_step = $checkpoint_step")

runs = 20
if isfile(joinpath(output_folder, "positions.bson"))
    initial_positions = BSON.load(joinpath(output_folder, "positions.bson"))[:pos]
else
    initial_positions = []
    for run in 1:runs
        reset!(env)
        # design_ = env.design
        design_ = AdjustablePositionScatterers(Cylinders(-8 .* rand(4, 2), env.design.cylinders.r, env.design.cylinders.c))
        # design_ = AdjustablePositionScatterers(Cylinders(-8 .* rand(2, 2), env.design.cylinders.r, env.design.cylinders.c))
        # design_ = FullyAdjustableScatterers(Cylinders(-8 .* rand(2, 2), env.design.cylinders.r, env.design.cylinders.c))
        push!(initial_positions, design_)
    end
    BSON.bson(joinpath(output_folder, "positions.bson"), pos=cpu(initial_positions))
end

for horizon in [1]
    shots = 512
    alpha = 1.0
    mpc = RandomShooting(policy, model, horizon, shots, alpha, focusing)
    node_mpc = RandomShooting(policy, node_model, horizon, shots, alpha, focusing)

    env.actions = 200
    t = build_tspan(0.0f0, env.dt, env.actions * env.integration_steps)
    seconds = 40.0
    frames = Int(round(Waves.FRAMES_PER_SECOND * seconds))
    tspan = collect(range(t[1], t[end], frames))
    log_message(output_folder, "shots = $shots\nalpha = $alpha\nenv.actions = $(env.actions)\nseconds = $seconds")

    for run_idx in 7:12
        try
            reset!(env)
            env.design = gpu(initial_positions[run_idx])
            design_1 = deepcopy(env.design)
            design_2 = deepcopy(env.design)
            @time create_data(env, mpc, frames, tspan, joinpath(output_folder, "mpc_$run_idx.bson"), "MPC (AEM)")
            
            reset!(env)
            env.design = design_1
            @time create_data(env, policy, frames, tspan, joinpath(output_folder, "random_$run_idx.bson"), "Random")

            reset!(env)
            env.design = design_2
            @time create_data(env, node_mpc, frames, tspan, joinpath(output_folder, "node_$run_idx.bson"), "MPC (NODE)")
            println("\n\n *** Finished run number $run_idx ***\n\n")
        catch e
            println("Caught error for run number $run_idx : $(e)")
            GC.gc()
        end
    end
end


# for run_idx in 1:10
#     try
#         @time mpc_path = create_random_data(env, mpc, frames, tspan, joinpath(output_folder, "random_$run_idx.bson"))
#         println("\n\n\n *** Finished run number $run_idx ***\n\n\n")
#     catch e
#         println("Caught error for run number $run_idx : $e")
#         GC.gc()
#     end
# end

# println("recording \"mpc.mp4\"")
# fig = Figure()
# ax1 = Axis(fig[1, 1], aspect = 1.0, title = "Random Control (Red)", xlabel = "Space (m)", ylabel = "Space (m)")
# ax2 = Axis(fig[2, 1], aspect = 1.0, title = "MPC (Green)", xlabel = "Space (m)", ylabel = "Space (m)")
# ax3 = Axis(fig[1:2, 2], title = "Scattered Energy in Environment", xlabel = "Time (s)", ylabel = "Energy")
# xlims!(ax3, t[1], t[end])
# ylims!(ax3, 0.0, max(maximum(mpc_signal[3, :]), maximum(random_signal[3, :])) * 1.20)

# record(fig, joinpath(output_folder, "mpc.mp4"), axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
#     println(i)
#     empty!(ax1)
#     heatmap!(ax1, dim.x, dim.y, x_random(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
#     mesh!(ax1, Waves.multi_design_interpolation(interps_random, tspan[i]))
#     empty!(ax2)
#     heatmap!(ax2, dim.x, dim.y, x_mpc(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
#     mesh!(ax2, Waves.multi_design_interpolation(interps_mpc, tspan[i]))

#     idx = findfirst(tspan[i] .<= t)[1]
#     empty!(ax3)
#     lines!(ax3, t[1:idx], mpc_signal[3, 1:idx], color = :green)
#     lines!(ax3, t[1:idx], random_signal[3, 1:idx], color = :red)
# end

# println("recording \"actions=100_random_control.mp4\"")
# fig = Figure()
# ax1 = Axis(fig[1, 1], aspect = 1.0, title = "Random Control", xlabel = "Space (m)", ylabel = "Space (m)")
# record(fig, joinpath(output_folder, "actions=100_random_control.mp4"), axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
#     println(i)
#     empty!(ax1)
#     heatmap!(ax1, dim.x, dim.y, x_random(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
#     mesh!(ax1, Waves.multi_design_interpolation(interps_random, tspan[i]))
# end

# println("recording \"actions=100_mpc.mp4\"")
# fig = Figure()
# ax1 = Axis(fig[1, 1], aspect = 1.0, title = "MPC", xlabel = "Space (m)", ylabel = "Space (m)")
# record(fig, joinpath(output_folder, "actions=100_mpc.mp4"), axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
#     println(i)
#     empty!(ax1)
#     heatmap!(ax1, dim.x, dim.y, x_mpc(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
#     mesh!(ax1, Waves.multi_design_interpolation(interps_mpc, tspan[i]))
# end

# println("recording \"actions=100_scattered_energy.mp4\"")
# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "Scattered Energy in Environment", xlabel = "Time (s)", ylabel = "Energy")
# xlims!(ax1, t[1], t[end])
# ylims!(ax1, 0.0, max(maximum(mpc_signal[3, :]), maximum(random_signal[3, :])) * 1.20)
# record(fig, joinpath(output_folder, "actions=100_scattered_energy.mp4"), axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
#     idx = findfirst(tspan[i] .<= t)[1]
#     empty!(ax1)
#     lines!(ax1, t[1:idx], mpc_signal[3, 1:idx], color = :green)
#     lines!(ax1, t[1:idx], random_signal[3, 1:idx], color = :red)
# end

# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "Scattered Energy in Environment", xlabel = "Time (s)", ylabel = "Energy")
# xlims!(ax1, t[1], t[end])
# ylims!(ax1, 0.0, max(maximum(mpc_signal[3, :]), maximum(random_signal[3, :])) * 1.20)
# empty!(ax1)
# lines!(ax1, vec(t), mpc_signal[3, :], color = :green, label = "MPC")
# lines!(ax1, vec(t), random_signal[3, :], color = :red, label = "Random Control")
# axislegend(ax1)
# save(joinpath(output_folder, "actions=100_scattered_energy.png"), fig)




# mpc_signal = render!(mpc, env, path = "mpc.mp4", energy = true, bound = 0.2f0, reset = false, field = :sc)
# mpc_signal = flatten_repeated_last_dim(cat(transpose.(mpc_signal)..., dims = 3))
# reset!(env)
# env.source.shape = shape
# random_signal = render!(policy, env, path = "random.mp4", energy = true, bound = 0.2f0, reset = false, field = :sc)
# random_signal = flatten_repeated_last_dim(cat(transpose.(random_signal)..., dims = 3))

# tspan = build_tspan(0.0f0, env.dt, size(mpc_signal, 2)-1)

# fig = Figure()
# ax = Axis(fig[1, 1], title = "Scattered Energy In Response to Actuation (50 actions)", xlabel = "Time (s)", ylabel = "Scattered Energy")
# lines!(ax, tspan, mpc_signal[3, :], label = "MPC", color = :green)
# lines!(ax, tspan, random_signal[3, :], label = "Random", color = :red)
# axislegend(ax, position = :rb)
# save("signals.png", fig)



# # delta_mu = (env.source.μ_high .- env.source.μ_low)
# # x = gpu(collect(range(0.0f0, 1.0f0, 5)))
# # mu = env.source.μ_low .+ delta_mu .* x

# # for location in axes(mu, 1)
# #     shape = build_normal(env.source.grid, mu[[location], :], env.source.σ, env.source.a)

# #     for episode in 1:4
# #         reset!(env)
# #         env.source.shape = shape
# #         mpc_ep = generate_episode!(mpc, env, reset = false)
# #         # save(mpc_ep, "control_results/cPILS_location=$location,episode=$episode.bson")
# #         save(mpc_ep, "control_results/PINC_location=$location,episode=$episode.bson")
# #         # reset!(env)
# #         # env.source.shape = shape
# #         # random_ep = generate_episode!(policy, env, reset = false)
# #         # save(random_ep, "control_results/random_location=$location,episode=$episode.bson")
# #     end
# # end
