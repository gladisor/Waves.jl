using Waves, Flux, BSON
Flux.CUDA.allowscalar(false)
using ReinforcementLearning
using Interpolations: linear_interpolation
using CairoMakie

function build_interpolator(
        env::WaveEnv, 
        a::Vector{<: AbstractDesign};
        bound::Float32 = 1.0f0,
        reset::Bool = true, 
        energy::Bool = false,
        field::Symbol = :tot)

    @assert field ∈ [:tot, :inc, :sc]

    tspans = []
    interps = DesignInterpolator[]

    x = []
    σ = []

    if reset
        RLBase.reset!(env)
    end
    
    for action in a
        tspan, interp, u_tot, u_inc = cpu(env(action))

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

        println(env.time_step)
    end

    tspan = flatten_repeated_last_dim(hcat(tspans...))

    x = flatten_repeated_last_dim(cat(x..., dims = 4))
    x = linear_interpolation(tspan, Flux.unbatch(x))

    return x, interps, σ
end

Flux.device!(0)

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
checkpoint = 10040
our_model_name = "ours_balanced_field_scale"

@time env = gpu(BSON.load(joinpath(DATA_PATH, "env.bson"))[:env])
OUR_MODEL_PATH = joinpath(DATA_PATH, "models/$our_model_name/checkpoint_step=$checkpoint/checkpoint.bson")
our_model = gpu(BSON.load(OUR_MODEL_PATH)[:model])

policy = RandomDesignPolicy(action_space(env))

horizon = 10
shots = 256
alpha = 1.0
mpc = RandomShooting(policy, model, horizon, shots, alpha)


T = 50
reset!(env)
s = gpu([state(env)])
a = gpu(reshape([policy(env) for i in 1:T], T, 1))
t = gpu(build_tspan(0.0f0, env.dt, T * env.integration_steps)[:, :])
z = cpu(generate_latent_solution(our_model, s, a, t))
z_tot = z[:, 1, 1, :]
z_inc = z[:, 3, 1, :]
z_sc = z_tot .- z_inc
y_hat = transpose(compute_latent_energy(z, our_model.dx)[:, :, 1])
x, interps, signal = build_interpolator(env, vec(a), reset = false)
y = flatten_repeated_last_dim(cat(transpose.(signal)..., dims = 3))
t = cpu(t)
dim = cpu(env.dim)
latent_dim = cpu(our_model.iter.dynamics.dim)
z_sc = linear_interpolation(vec(t), Flux.unbatch(z_sc))


fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0, title = "Acoustic Environment", xlabel = "Space (m)", ylabel = "Space (m)")
ax2 = Axis(fig[1:2, 2], title = "Real vs Latent Scattered Energy over Time", xlabel = "Time (s)", ylabel = "Energy")
xlims!(ax2, t[1], t[end])
ylims!(ax2, 0.0, max(maximum(y_hat[3, :]), maximum(y[3, :])) * 1.20)
ax3 = Axis(fig[2, 1], title = "Latent Scattered Field", xlabel = "Space (m)", ylabel = "Displacement (m)")
xlims!(ax3, latent_dim.x[1], latent_dim.x[end])
ylims!(ax3, -2.0, 2.0)

seconds = 40.0
frames = Int(round(Waves.FRAMES_PER_SECOND * seconds))
tspan = collect(range(t[1], t[end], frames))

record(fig, "dashboard.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    println(i)
    empty!(ax1)
    heatmap!(ax1, dim.x, dim.y, x(tspan[i]), colormap = :ice, colorrange = (-1.0, 1.0))
    mesh!(ax1, Waves.multi_design_interpolation(interps, tspan[i]))

    idx = findfirst(tspan[i] .<= t)[1]
    empty!(ax2)
    lines!(ax2, t[1:idx], y[3, 1:idx], color = :blue)
    lines!(ax2, t[1:idx], y_hat[3, 1:idx], color = :green, alpha = 0.2)

    empty!(ax3)
    lines!(ax3, latent_dim.x, z_sc(tspan[i]), color = :green)
end



fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0, title = "Acoustic Environment", xlabel = "Space (m)", ylabel = "Space (m)")
record(fig, "ax1_dashboard.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    println(i)
    empty!(ax1)
    heatmap!(ax1, dim.x, dim.y, x(tspan[i]), colormap = :ice, colorrange = (-1.0, 1.0))
    mesh!(ax1, Waves.multi_design_interpolation(interps, tspan[i]))
end

fig = Figure()
ax2 = Axis(fig[1, 1], title = "Real vs Latent Scattered Energy over Time", xlabel = "Time (s)", ylabel = "Energy")
xlims!(ax2, t[1], t[end])
ylims!(ax2, 0.0, max(maximum(y_hat[3, :]), maximum(y[3, :])) * 1.20)
record(fig, "ax2_dashboard.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    idx = findfirst(tspan[i] .<= t)[1]
    empty!(ax2)
    lines!(ax2, t[1:idx], y[3, 1:idx], color = :blue)
    lines!(ax2, t[1:idx], y_hat[3, 1:idx], color = :green, alpha = 0.2)
end


fig = Figure()
ax2 = Axis(fig[1, 1], title = "Real vs Latent Scattered Energy over Time", xlabel = "Time (s)", ylabel = "Energy")
xlims!(ax2, t[1], t[end])
ylims!(ax2, 0.0, max(maximum(y_hat[3, :]), maximum(y[3, :])) * 1.20)
lines!(ax2, vec(t), y[3, :], color = :blue)
lines!(ax2, vec(t), y_hat[3, :], color = :green, alpha = 0.2)
save("ax2_dashboard_final.png", fig)



fig = Figure()
ax3 = Axis(fig[1, 1], title = "Latent Scattered Field", xlabel = "Space (m)", ylabel = "Displacement (m)")
xlims!(ax3, latent_dim.x[1], latent_dim.x[end])
ylims!(ax3, -2.0, 2.0)
record(fig, "ax3_dashboard.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    empty!(ax3)
    lines!(ax3, latent_dim.x, z_sc(tspan[i]), color = :green)
end

# ep = Episode(path = joinpath(DATA_PATH, "episodes/episode500.bson"))
# horizon = 2
# s, a, t, _ = gpu(Flux.batch.(prepare_data(ep, horizon)))

# s = s[1, :]
# a = a[:, [1]]
# t = t[:, [1]]

# z = cpu(generate_latent_solution(our_model, s, a, t))

# u_tot = z[:, 1, 1, :]
# u_inc = z[:, 3, 1, :]
# u_sc = u_tot .- u_inc




# fig = Figure()
# ax = Axis(fig[1, 1], 
#     title = "Scattered Energy in Latent Space",
#     xlabel = "Distance (m)",
#     ylabel = "Energy")
# latent_dim = cpu(our_model.iter.dynamics.dim)
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, 0.0f0, 1.5f0)
# u_sc_energy = u_sc .^ 2
# record(fig, "no_pml_latent.mp4", axes(u_sc, 2), framerate = 2 * Waves.FRAMES_PER_SECOND) do i 
#     empty!(ax)
#     lines!(ax, latent_dim.x, u_sc_energy[:, i], color = "blue")
# end











# fig = Figure()
# ax = Axis(fig[1, 1])
# heatmap!(ax, u_sc .^ 2, colormap = :inferno)
# save("latent.png", fig)
