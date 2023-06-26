# export plot_episode_data!, plot_sigma!, render!
using Interpolations
using Interpolations: Extrapolation
using CairoMakie

const FRAMES_PER_SECOND = 24

function plot_sigma!(episode::EpisodeData; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Total Scattered Energy During Episode", xlabel = "Time (s)", ylabel = "Total Scattered Energy")

    for i in 1:length(episode)
        lines!(ax, episode.tspans[i], episode.sigmas[i], color = :blue)
    end

    save(path, fig)
    return nothing
end

function render!(;
        dim::TwoDim, 
        tspan::Vector{Float32}, 
        u_total::Extrapolation, 
        design::Union{Extrapolation, Nothing} = nothing, 
        seconds::Float32 = 1.0f0,
        minimum_value = -1.0f0,
        maximum_value = 1.0f0,
        path::String)

    fig = Figure()

    ax = Axis(fig[1, 1], aspect = 1.0f0, title = "Total Field", xlabel = "Distance (m)", ylabel = "Distance (m)")

    frames = Int(round(FRAMES_PER_SECOND * seconds))
    tspan = range(tspan[1], tspan[end], frames)

    CairoMakie.record(fig, path, tspan, framerate = FRAMES_PER_SECOND) do t
        empty!(ax)
        heatmap!(ax, dim.x, dim.y, u_total(t), colormap = :ice, colorrange = (minimum_value, maximum_value))
        
        if !isnothing(design)
            mesh!(ax, design(t))
        end
    end

    return nothing
end

function render!(policy::AbstractPolicy, env::WaveEnv; kwargs...)

    RLBase.reset!(env)

    design_times = [time(env)]
    designs = [cpu(env.total_dynamics.design(time(env)))]

    tspan0, u_incident_0, u_scattered_0 = cpu(env(policy(env)))

    sol = Dict(
        :tspan => [tspan0],
        :incident => [displacement.(u_incident_0)],
        :scattered => [displacement.(u_scattered_0)]
    )

    while !is_terminated(env)
        tspan, u_incident, u_scattered = cpu(env(policy(env)))

        push!(sol[:tspan], tspan[2:end])
        push!(sol[:incident], displacement.(u_incident[2:end]))
        push!(sol[:scattered], displacement.(u_scattered[2:end]))

        push!(design_times, time(env))
        push!(designs, cpu(env.total_dynamics.design(time(env))))
    end

    tspans = vcat(sol[:tspan]...)
    u_incident = vcat(sol[:incident]...)
    u_scattered = vcat(sol[:scattered]...)
    u_total = u_incident .+ u_scattered

    design_times = vcat(design_times...)
    designs = vcat(designs...)
    
    u_total = linear_interpolation(tspans, u_total)
    d = linear_interpolation(design_times, designs, extrapolation_bc = Flat())

    render!(
        dim = cpu(env.dim), 
        tspan = tspans, 
        u_total = u_total,
        design = d; 
        kwargs...)
end

# function render!(dim::OneDim, u::AbstractArray{Float32, 3}; path::String)
function render!(dim::OneDim, u::AbstractMatrix{Float32}; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1], title = "Latent Wave Displacement Animation", xlabel = "Space (m)", ylabel = "Displacement (m)")

    xlims!(ax, dim.x[1], dim.x[end])

    # min_u = minimum(u[:, 1, :])
    # max_u = maximum(u[:, 1, :])
    min_u = minimum(u)
    max_u = maximum(u)

    if max_u ≈ min_u
        ylims!(ax, -0.10f0, 0.10f0)
    else
        ylims!(ax, min_u, max_u)
    end

    # CairoMakie.record(fig, path, axes(u, 3), framerate = 60) do i
    CairoMakie.record(fig, path, axes(u, 2), framerate = 60) do i
        empty!(ax)
        # lines!(ax, dim.x, u[:, 1, i], color = :blue)
        lines!(ax, dim.x, u[:, i], color = :blue)
    end
end