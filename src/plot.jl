export WavePlot, interpolate, render!

mutable struct WavePlot
    fig::Figure
    ax::Axis
end

function WavePlot(dim::OneDim)
    fig = Figure(resolution = (1920, 1080))
    ax = Axis(fig[1, 1], aspect = AxisAspect(1.0))
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
    mesh!(ax, Circle(Point(cyl.x, cyl.y), cyl.r), color = :gray)
end

"""
Renders an animation of a wave solution.
"""
function render!(
        sol::WaveSol{TwoDim}, 
        design::Union{DesignTrajectory, Nothing} = nothing; 
        path::String)

    p = WavePlot(sol.dim)
    record(p.fig, path, 1:length(sol)) do i
        empty!(p.ax)
        heatmap!(p.ax, sol.dim.x, sol.dim.y, sol[i][:, :, 1], colormap = :ice)
        if !isnothing(design)
            mesh!(p.ax, design[i])
        end
    end
end