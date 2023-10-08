export render!

const FRAMES_PER_SECOND = 24

function render!(policy::AbstractPolicy, env::WaveEnv, seconds::Float32 = env.actions * 0.5f0; path::String)
    tspans = []
    interps = DesignInterpolator[]
    u_tots = []
    # u_incs = []

    reset!(env)
    while !is_terminated(env)
        tspan, interp, u_tot, u_inc = cpu(env(policy(env)))
        push!(tspans, tspan)
        push!(interps, interp)
        push!(u_tots, u_tot)
        # push!(u_incs, u_inc)
        println(env.time_step)
    end

    tspan = flatten_repeated_last_dim(hcat(tspans...))

    u_tot = flatten_repeated_last_dim(cat(u_tots..., dims = 4))
    u_tot = linear_interpolation(tspan, Flux.unbatch(u_tot))

    frames = Int(round(FRAMES_PER_SECOND * seconds))
    tspan = range(tspan[1], tspan[end], frames)

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0f0)

    dim = cpu(env.dim)
    @time record(fig, path, tspan, framerate = FRAMES_PER_SECOND) do t
        empty!(ax1)
        heatmap!(ax1, dim.x, dim.y, u_tot(t), colormap = :ice, colorrange = (-1.0f0, 1.0f0))
        mesh!(ax1, multi_design_interpolation(interps, t))
    end
    return nothing
end

function visualize(ep::Episode{WaveEnvState, Matrix{Float32}}; path::String, horizon::Int = length(ep), idx::Int = 1)
    _, _, t, y = prepare_data(ep, horizon)
    tspan = t[idx]
    sigma = y[idx]

    fig = Figure()
    ax = Axis(fig[1, 1], title = "Energy Signals in Real Dynamics")
    lines!(ax, tspan, sigma[:, 1], color = :blue, label = "Total")
    lines!(ax, tspan, sigma[:, 2], color = :orange, label = "Incident")
    lines!(ax, tspan, sigma[:, 3], color = :green, label = "Scattered")
    axislegend(ax, position = :rb)
    save(path, fig)
    return nothing
end

function plot_predicted_energy(tspan::Vector{Float32}, true_energy::Vector{Float32}, pred_energy::Vector{Float32}; title::String, path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], title = title, xlabel = "Time (s)", ylabel = "Energy")
    lines!(ax, tspan, true_energy, color = :blue, label = "True")
    lines!(ax, tspan, pred_energy, color = :orange, label = "Predicted")
    save(path, fig)
end

# function plot_latent_source(model::AcousticEnergyModel; path::String)
#     period = collect(0.0f0:model.iter.dt:(0.5f0/model.F.freq))

#     t = Flux.device!(Flux.device(model)) do
#         return gpu(period[:, :])
#     end

#     F = model.F
#     f = cpu(hcat([F(t[i, :]) for i in axes(t, 1)]...))
#     dim = cpu(model.iter.dynamics.dim)
#     tspan = cpu(t[:, 1])

#     fig = Figure()
#     ax1 = Axis(fig[1, 1], title = "One Period of Force Function", xlabel = "Time (s)", ylabel = "Space (m)")
#     hm = heatmap!(ax1, tspan, dim.x, permutedims(f), colormap = :ice)
#     Colorbar(fig[1, 2], hm)
    
#     ax2 = Axis(fig[2, 1], title = "Shape of Force Function", xlabel = "Space (m)")
#     lines!(ax2, dim.x, cpu(F(t[end÷2, :])[:, 1]))
#     save(path, fig)
# end