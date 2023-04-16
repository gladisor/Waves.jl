export plot_solution!

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

function render!(dim::OneDim, u::AbstractArray{Float32, 3}; path::String)
    fig, ax = plot_wave(dim, u[:, :, 1])

    record(fig, path, axes(u, 3), framerate = 60) do i
        empty!(ax)
        lines!(ax, dim.x, u[:, 1, i], color = :blue)
    end
end

function render!(dim::TwoDim, tspan::Vector{Float32}, u::Extrapolation, design::Extrapolation; seconds::Float32 = 1.0f0)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0)

    frames = Int(round(FRAMES_PER_SECOND * seconds))
    tspan = range(tspan[1], tspan[end], frames)

    record(fig, "vid.mp4", tspan, framerate = FRAMES_PER_SECOND) do t
        empty!(ax)
        heatmap!(ax, dim.x, dim.y, u(t)[:, :, 1], colormap = :ice)
        mesh!(ax, design(t))
    end
end