const RESOLUTION = (1920, 1080)
const FONT_SIZE = 30

struct Wave{D <: AbstractDim}
    dim::D
    u::AbstractArray{Float64}
end

"""
Replacing GLMakie save with easier syntax
"""
function save(fig::GLMakie.Figure, path::String)
    GLMakie.save(path, fig)
end

"""
Quick plotting function for ez visualizations
"""
function plot(x::Vector, y::Vector)
    fig = GLMakie.Figure(resolution = RESOLUTION, fontsize = FONT_SIZE)
    ax = GLMakie.Axis(fig[1, 1], title = "Plot", xlabel = "x", ylabel = "y")
    GLMakie.lines!(ax, x, y, color = :blue)
    return fig
end

"""
Create a figure for plotting and animating waves in one dimension
"""
function plot(dim::OneDim)
    fig = GLMakie.Figure(resolution = RESOLUTION, fontsize = FONT_SIZE)
    ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X (m)", ylabel = "Displacement (m)")
    GLMakie.xlims!(ax, dim.x[1], dim.x[end])
    GLMakie.ylims!(ax, -1.0, 1.0)
    return fig
end

function plot!(fig::GLMakie.Figure, wave::Wave{OneDim})
    GLMakie.lines!(fig.content[1], wave.dim.x, wave.u[:, 1], color = :blue, linewidth = 3)
end

"""
Creating figure for two dimensional wave plotting
"""
function plot(dim::TwoDim)
    fig = GLMakie.Figure(resolution = RESOLUTION, fontsize = FONT_SIZE)
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Displacement (m)")
    GLMakie.xlims!(ax, dim.x[1], dim.x[end])
    GLMakie.ylims!(ax, dim.y[1], dim.y[end])
    GLMakie.zlims!(ax, -1.0, 5.0)
    return fig
end

function plot!(fig::GLMakie.Figure, wave::Wave{TwoDim})
    GLMakie.surface!(fig.content[1], wave.dim.x, wave.dim.y, wave.u[:, :, 1], colormap = :ice)
end

"""
Creating figure for three dimensional wave plotting
"""
function plot(dim::ThreeDim)
    fig = GLMakie.Figure(resolution = RESOLUTION, fontsize = FONT_SIZE)
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "3D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Z (m)")
    GLMakie.xlims!(ax, dim.x[1], dim.x[end])
    GLMakie.ylims!(ax, dim.y[1], dim.y[end])
    GLMakie.zlims!(ax, dim.z[1], dim.z[end])
    return fig
end

function plot!(fig::GLMakie.Figure, cyl::Cylinder)
    GLMakie.mesh!(fig.content[1], GLMakie.GeometryBasics.Cylinder3{Float32}(GLMakie.Point3f(cyl.x, cyl.y, -1.0), GLMakie.Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

"""
Constructs a WaveSol which is interpolated from an ODESolution
"""
function interpolate(sol::ODESolution, dim::AbstractDim, dt::Float64)
    u = typeof(sol.prob.u0)[]
    t = collect(sol.prob.tspan[1]:dt:sol.prob.tspan[end])

    for i ∈ axes(t, 1)
        push!(u, sol(t[i]))
    end

    return WaveSol(dim, t, u)
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

"""
Renders an animation of a wave solution.
"""
function render!(sol::WaveSol, design_trajectory::Union{Vector{<:AbstractDesign}, Nothing} = nothing; path::String)
    fig = Waves.plot(sol.dim)

    GLMakie.record(fig, path, 1:length(sol)) do i
        GLMakie.empty!(fig.content[1].scene)
        if !isnothing(design_trajectory)
            Waves.plot!(fig, design_trajectory[i])
        end
        wave = Waves.Wave(sol.dim, sol.u[i])
        Waves.plot!(fig, wave)
    end
end
