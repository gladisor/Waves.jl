const FRAMES_PER_SECOND = 24

function plot_solution!(nrows::Int, ncols::Int, dim::TwoDim, u::AbstractArray{Float32, 4}; path::String, field::Int = 1)
    
    fig = Figure()
    layout = fig[1, 1] = GridLayout(nrows, ncols)

    steps = size(u, ndims(u))
    n = nrows * ncols
    idx = Int.(round.(LinRange(1, steps, n)))

    for i in 1:nrows
        for j in 1:ncols
            k = (i-1) * ncols + j
            ax = Axis(layout[i, j], aspect = 1.0f0)
            heatmap!(ax, dim.x, dim.y, u[:, :, field, idx[k]], colormap = :ice)
        end
    end

    save(path, fig)
end

function plot_wave(dim::OneDim, wave::AbstractVector{Float32}; ylims::Tuple = (-1.0f0, 1.0f0))
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0)
    xlims!(ax, dim.x[1], dim.x[end])
    ylims!(ax, ylims...)
    lines!(ax, dim.x, wave)
    return fig, ax
end

function plot_wave(dim::OneDim, wave::AbstractMatrix{Float32}; kwargs...)
    return plot_wave(dim, wave[:, 1]; kwargs...)
end

function plot_wave(dim::TwoDim, wave::AbstractMatrix{Float32})
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0)
    heatmap!(ax, dim.x, dim.y, wave, colormap = :ice)
    return fig, ax
end

function plot_wave(dim::TwoDim, wave::AbstractArray{Float32, 3})
    return plot_wave(dim, wave[:, :, 1])
end



function CairoMakie.mesh!(ax::Axis, config::Scatterers)
    for i ∈ axes(config.pos, 1)
        mesh!(ax, Circle(Point(config.pos[i, :]...), config.r[i]), color = :gray)
    end
end

function plot_sigma!(episode_data::EpisodeData; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, vcat(episode_data.tspans...), vcat(episode_data.sigmas...))
    save(path, fig)
    return nothing
end

function plot_episode_data!(episode_data::EpisodeData; cols::Int, path::String)

    fig = Figure(resolution = (1920, 1080))

    for i in axes(episode_data.states, 1)
        dim = episode_data.states[i].dim
        wave = episode_data.states[i].wave_total
        design = episode_data.states[i].design

        row = (i - 1) ÷ cols
        col = (i - 1) % cols + 1

        ax = Axis(fig[row, col], aspect = 1.0f0)
        heatmap!(ax, dim.x, dim.y, wave[:, :, 1], colormap = :ice)
        mesh!(ax, design)
    end

    save(path, fig)
    return nothing
end

function plot_sigma!(model::AbstractWaveControlModel, episode::EpisodeData; path::String)
    pred_sigmas = cpu([model(gpu(s), gpu(a)) for (s, a) in zip(episode.states, episode.actions)])

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, vcat(episode.tspans...), vcat(episode.sigmas...), color = :blue)
    lines!(ax, vcat(episode.tspans...), vcat(pred_sigmas...), color = :orange)
    save(path, fig)
    return nothing
end

function plot_action_distribution!(
    model::WaveControlModel,
    policy::RandomDesignPolicy, 
    env::WaveEnv; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1])

    for _ in 1:10
        lines!(ax, cpu(model(state(env), gpu(policy(env)))))
    end

    save(path, fig)
end

function render!(dim::TwoDim, tspan::Vector{Float32}, u::Extrapolation, design::Union{Extrapolation, Nothing} = nothing; seconds::Float32 = 1.0f0, path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0)

    frames = Int(round(FRAMES_PER_SECOND * seconds))
    tspan = range(tspan[1], tspan[end], frames)

    record(fig, path, tspan, framerate = FRAMES_PER_SECOND) do t
        empty!(ax)
        heatmap!(ax, dim.x, dim.y, u(t)[:, :, 1], colormap = :ice)
        if !isnothing(design)
            mesh!(ax, design(t))
        end
    end

    return nothing
end

function render!(policy::AbstractPolicy, env::WaveEnv; kwargs...)

    reset!(env)

    tspan0, u0 = env(policy(env))

    tspans = [tspan0]
    us = [u0]

    while !is_terminated(env)
        tspan, u = env(policy(env))

        push!(tspans, tspan[2:end])
        push!(us, u[2:end])
    end

    tspans = vcat(tspans...)
    us = vcat(us...)

    sol = linear_interpolation(tspans, us)

    render!(env.dim, tspans, sol; kwargs...)
end

function render!(dim::OneDim, u::AbstractArray{Float32, 3}; path::String)
    fig, ax = plot_wave(dim, u[:, :, 1])

    record(fig, path, axes(u, 3), framerate = 60) do i
        empty!(ax)
        lines!(ax, dim.x, u[:, 1, i], color = :blue)
    end
end

function render_latent_wave!(dim::OneDim, model::WaveControlModel, s::WaveEnvState, action::AbstractDesign; path::String)
    z = cpu(model.iter(encode(model, s, action)))
    render!(dim, z, path = path)
end