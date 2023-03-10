export WavePlot, plot_wave!, render!

mutable struct WavePlot
    fig::Figure
    ax::Axis
end

function WavePlot(dim::OneDim)
    fig = Figure(resolution = (1920, 1080))
    ax = Axis(fig[1, 1], title = "1D Wave", xlabel = "X (m)")
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

function plot_design!(p::WavePlot, cyl::Cylinder)
    mesh!(p.ax, cyl)
end

function plot_wave!(p::WavePlot, dim::OneDim, wave::AbstractMatrix{Float32})
    lines!(p.ax, dim.x, displacement(wave), color = :blue, linewidth = 3)
end

function plot_wave!(p::WavePlot, dim::TwoDim, wave::AbstractArray{Float32, 3})
    heatmap!(p.ax, dim.x, dim.y, displacement(wave), colormap = :ice)
end

"""
Renders an animation of a wave solution.
"""
function render!(
        sol::WaveSol, 
        design::Union{DesignTrajectory, Nothing} = nothing; path::String, fps::Int = 24)

    p = WavePlot(sol.dim)

    wave_interp = linear_interpolation(sol.t, sol.u)

    if !isnothing(design)
        design_interp = linear_interpolation(sol.t, design.traj)
    end

    n_frames = Int(round(fps * sol.t[end]))
    t = collect(range(sol.t[1], sol.t[end], n_frames))

    record(p.fig, path, 1:n_frames, framerate = fps) do i

        empty!(p.ax)
        plot_wave!(p, sol.dim, wave_interp(t[i]))

        if !isnothing(design)
            plot_design!(p, design_interp(t[i]))
        end
    end
end