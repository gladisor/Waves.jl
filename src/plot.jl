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

function interpolate(dim::AbstractDim, sol::ODESolution, dt::Float64)::WaveSol
    t = collect(sol.prob.tspan[1]:dt:sol.prob.tspan[end])
    return WaveSol(dim, t, sol(t).u)
end

"""
Goes through a Vector of DesignInterpolator and creates a vector of AbstractDesign
interpolated at the specified dt
"""
function interpolate(designs::Vector{DesignInterpolator}, dt::Float64)
    design_trajectory = typeof(first(designs).initial)[]

    for i ∈ axes(designs, 1)

        design = designs[i]
        t = collect(design.ti:dt:design.tf)

        for j ∈ axes(t, 1)
            push!(design_trajectory, designs[i](t[j]))
        end

        pop!(design_trajectory)
    end

    last_design = last(designs)

    push!(design_trajectory, last_design(last_design.tf))

    return design_trajectory
end

function CairoMakie.mesh!(ax::Axis, cyl::Cylinder)
    mesh!(ax, Circle(Point(cyl.x, cyl.y), cyl.r), color = :gray)
end

"""
Renders an animation of a wave solution.
"""
function render!(
        sol::WaveSol{TwoDim}, 
        designs::Union{Vector{<:AbstractDesign}, Nothing} = nothing; 
        path::String)

    p = WavePlot(sol.dim)
    record(p.fig, path, 1:length(sol)) do i
        empty!(p.ax)
        heatmap!(p.ax, sol.dim.x, sol.dim.y, sol[i][:, :, 1], colormap = :ice)
        if !isnothing(designs)
            mesh!(p.ax, designs[i])
        end
    end
end