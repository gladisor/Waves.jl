export WavePlot, interpolate, render!

mutable struct WavePlot
    fig::Figure
    ax::Axis
end

function WavePlot(dim::OneDim)
    fig = Figure(resolution = (1920, 1080))
    ax = Axis(fig[1, 1])
    xlims!(ax, dim.x[1], dim.x[end])
    ylims!(ax, -1.0, 1.0)
    return WavePlot(fig, ax)
end

function WavePlot(dim::TwoDim)
    fig = Figure(resolution = (1920, 1080))
    ax = Axis(fig[1, 1], aspect = AxisAspect(1.0), title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)")
    xlims!(ax, dim.x[1], dim.x[end])
    ylims!(ax, dim.y[1], dim.y[end])
    return WavePlot(fig, ax)
end

function CairoMakie.mesh!(ax::Axis, cyl::Cylinder)
    mesh!(ax, Circle(Point(cyl.pos...), cyl.r), color = :gray)
end

function render!(
    sol::WaveSol{OneDim}, design::Union{DesignTrajectory, Nothing} = nothing; path::String, fps::Int = 24, n_frames = nothing)

    p = WavePlot(sol.dim)

    wave_interp = linear_interpolation(sol.t, sol.u)

    if !isnothing(design)
        design_interp = linear_interpolation(sol.t, design.traj)
    end

    if isnothing(n_frames)
        n_frames = Int(round(fps * sol.t[end]))
    end
    t = collect(range(sol.t[1], sol.t[end], n_frames))

    record(p.fig, path, 1:n_frames, framerate = fps) do i
        empty!(p.ax)
        lines!(p.ax, sol.dim.x, wave_interp(t[i])[:, 1], color = :blue, linewidth = 3)
        if !isnothing(design)
            mesh!(p.ax, design_interp(t[i]))
        end
    end
end

"""
Renders an animation of a wave solution.
"""
function render!(
        sol::WaveSol{TwoDim}, 
        design::Union{DesignTrajectory, Nothing} = nothing; 
        path::String,
        fps::Int = 24,
        n_frames = nothing)

    p = WavePlot(sol.dim)

    wave_interp = linear_interpolation(sol.t, sol.u)

    if !isnothing(design)
        design_interp = linear_interpolation(sol.t, design.traj)
    end

    if isnothing(n_frames)
        n_frames = Int(round(fps * sol.t[end]))
    end

    t = collect(range(sol.t[1], sol.t[end], n_frames))

    record(p.fig, path, 1:n_frames, framerate = fps) do i
        empty!(p.ax)
        heatmap!(p.ax, sol.dim.x, sol.dim.y, wave_interp(t[i])[:, :, 1], colormap = :ice)
        if !isnothing(design)
            mesh!(p.ax, design_interp(t[i]))
        end
    end
end