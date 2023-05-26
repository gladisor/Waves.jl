# export plot_episode_data!, plot_sigma!, render!
using Interpolations
using Interpolations: Extrapolation
using CairoMakie

const FRAMES_PER_SECOND = 24

# function plot_wave(dim::OneDim, wave::AbstractVector{Float32}; ylims::Tuple = (-1.0f0, 1.0f0))
#     fig = Figure()
#     ax = Axis(fig[1, 1], aspect = 1.0f0)
#     xlims!(ax, dim.x[1], dim.x[end])
#     ylims!(ax, ylims...)
#     lines!(ax, dim.x, wave)
#     return fig, ax
# end

# function plot_wave(dim::OneDim, wave::AbstractMatrix{Float32}; kwargs...)
#     return plot_wave(dim, wave[:, 1]; kwargs...)
# end

# function plot_wave(dim::TwoDim, wave::AbstractMatrix{Float32})
#     fig = Figure()
#     ax = Axis(fig[1, 1], aspect = 1.0)
#     heatmap!(ax, dim.x, dim.y, wave, colormap = :ice)
#     return fig, ax
# end

# function plot_wave(dim::TwoDim, wave::AbstractArray{Float32, 3})
#     return plot_wave(dim, wave[:, :, 1])
# end

# function plot_episode_data!(episode_data::EpisodeData; cols::Int, path::String)

#     fig = Figure(resolution = (1920, 1080))

#     for i in axes(episode_data.states, 1)
#         dim = episode_data.states[i].dim
#         wave = episode_data.states[i].wave_total
#         design = episode_data.states[i].design

#         row = (i - 1) ÷ cols
#         col = (i - 1) % cols + 1

#         ax = Axis(fig[row, col], aspect = 1.0f0)
#         heatmap!(ax, dim.x, dim.y, wave[:, :, 1], colormap = :ice)
#         mesh!(ax, design)
#     end

#     save(path, fig)
#     return nothing
# end

function plot_sigma!(episode::EpisodeData; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Total Scattered Energy During Episode", xlabel = "Time (s)", ylabel = "Total Scattered Energy")

    for i in 1:length(episode)
        lines!(ax, episode.tspans[i], episode.sigmas[i], color = :blue)
    end

    save(path, fig)
    return nothing
end

# function plot_sigma!(model::WaveControlModel, episode::EpisodeData; path::String)
#     pred_sigmas = cpu([model(gpu(s), gpu(a)) for (s, a) in zip(episode.states, episode.actions)])

#     fig = Figure()
#     ax = Axis(fig[1, 1])

#     for i in 1:length(episode)
#         lines!(ax, episode.tspans[i], episode.sigmas[i], color = :blue)
#         lines!(ax, episode.tspans[i], pred_sigmas[i], color = :orange)
#     end

#     # lines!(ax, vcat(episode.tspans...), vcat(episode.sigmas...), color = :blue)
#     # lines!(ax, vcat(episode.tspans...), vcat(pred_sigmas...), color = :orange)
#     save(path, fig)
#     return nothing
# end

# function plot_sigma!(
#         model::WaveControlModel, 
#         s::WaveEnvState, 
#         a::Vector{<: AbstractDesign}, 
#         tspan::AbstractMatrix{Float32},
#         sigma::AbstractMatrix{Float32};
#         path::String)

#     sigma_pred = model(s, a)

#     fig = Figure()
#     ax = Axis(fig[1, 1])

#     for i in axes(sigma, 2)
#         lines!(ax, cpu(tspan[:, i]), cpu(sigma[:, i]), color = :blue)
#         lines!(ax, cpu(tspan[:, i]), cpu(sigma_pred[:, i]), color = :orange)
#     end

#     save(path, fig)
# end

# function plot_action_distribution!(
#     model::WaveControlModel,
#     policy::RandomDesignPolicy, 
#     env::WaveEnv; path::String)

#     fig = Figure()
#     ax = Axis(fig[1, 1])

#     for _ in 1:10
#         lines!(ax, cpu(model(state(env), gpu(policy(env)))))
#     end

#     save(path, fig)
# end

function render!(;
        dim::TwoDim, 
        tspan::Vector{Float32}, 
        u_scattered::Extrapolation, 
        u_incident::Union{Extrapolation, Nothing} = nothing, 
        design::Union{Extrapolation, Nothing} = nothing, 
        seconds::Float32 = 1.0f0, 
        path::String)

    fig = Figure()

    ax1 = Axis(fig[1, 1], aspect = 1.0f0, title = "Scattered Field", xlabel = "Distance (m)", ylabel = "Distance (m)")

    if !isnothing(u_incident)
        ax2 = Axis(fig[1, 2], aspect = 1.0f0, title = "Incident Field", xlabel = "Distance (m)", ylabel = "Distance (m)")
    end

    frames = Int(round(FRAMES_PER_SECOND * seconds))
    tspan = range(tspan[1], tspan[end], frames)

    record(fig, path, tspan, framerate = FRAMES_PER_SECOND) do t
        empty!(ax1)
        heatmap!(ax1, dim.x, dim.y, u_scattered(t)[:, :, 1], colormap = :ice)

        if !isnothing(u_incident)
            empty!(ax2)
            heatmap!(ax2, dim.x, dim.y, u_incident(t)[:, :, 1], colormap = :ice)
        end
        
        if !isnothing(design)
            mesh!(ax1, design(t))
        end
    end

    return nothing
end

function render!(policy::AbstractPolicy, env::WaveEnv; kwargs...)

    reset!(env)

    design_times = [time(env)]
    designs = [cpu(env.total_dynamics.design(time(env)))]

    tspan0, u_incident_0, u_scattered_0 = cpu(env(policy(env)))

    sol = Dict(
        :tspan => [tspan0],
        :incident => [u_incident_0],
        :scattered => [u_scattered_0]
    )

    while !is_terminated(env)
        tspan, u_incident, u_scattered = cpu(env(policy(env)))

        push!(sol[:tspan], tspan[2:end])
        push!(sol[:incident], u_incident[2:end])
        push!(sol[:scattered], u_scattered[2:end])

        push!(design_times, time(env))
        push!(designs, cpu(env.total_dynamics.design(time(env))))
    end

    tspans = vcat(sol[:tspan]...)
    u_incident = vcat(sol[:incident]...)
    u_scattered = vcat(sol[:scattered]...)

    design_times = vcat(design_times...)
    designs = vcat(designs...)

    u_incident = linear_interpolation(tspans, u_incident)
    u_scattered = linear_interpolation(tspans, u_scattered)
    d = linear_interpolation(design_times, designs, extrapolation_bc = Flat())

    render!(
        dim = cpu(env.dim), 
        tspan = tspans, 
        u_scattered = u_scattered, 
        # u_incident = u_incident,
        design = d; 
        kwargs...)
end

function render!(dim::OneDim, u::AbstractArray{Float32, 3}; path::String)
    
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Latent Wave Displacement Animation", xlabel = "Space (m)", ylabel = "Displacement (m)")

    xlims!(ax, dim.x[1], dim.x[end])

    min_u = minimum(u[:, 1, :])
    max_u = maximum(u[:, 1, :])

    if max_u ≈ min_u
        ylims!(ax, -0.10f0, 0.10f0)
    else
        ylims!(ax, min_u, max_u)
    end

    record(fig, path, axes(u, 3), framerate = 60) do i
        empty!(ax)
        lines!(ax, dim.x, u[:, 1, i], color = :blue)
    end
end

# function render_latent_wave!(dim::OneDim, model::WaveControlModel, s::WaveEnvState, action::AbstractDesign; path::String)
#     z = cpu(model.iter(encode(model, s, action)))
#     render!(dim, z, path = path)
# end