export plot, plot!, save, render!

function plot(dim::OneDim)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X (m)", ylabel = "Displacement (m)")
    GLMakie.xlims!(ax, dim.x[1], dim.x[end])
    GLMakie.ylims!(ax, -1.0, 1.0)
    return fig
end

function plot(dim::TwoDim)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Displacement (m)")
    GLMakie.xlims!(ax, dim.x[1], dim.x[end])
    GLMakie.ylims!(ax, dim.y[1], dim.y[end])
    GLMakie.zlims!(ax, -1.0, 5.0)
    return fig
end

function plot!(fig::GLMakie.Figure, wave::Wave{TwoDim})
    GLMakie.surface!(fig.content[1], wave.dim.x, wave.dim.y, wave.u, colormap = :ice)
end

function plot(wave::Wave{TwoDim})
    dim = wave.dim
    fig = plot(dim)
    plot!(fig, wave)
    return fig
end

function plot(dim::ThreeDim)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "3D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Z (m)")
    GLMakie.xlims!(ax, dim.x[1], dim.x[end])
    GLMakie.ylims!(ax, dim.y[1], dim.y[end])
    GLMakie.zlims!(ax, dim.z[1], dim.z[end])
    return fig
end

function save(fig::GLMakie.Figure, path::String)
    GLMakie.save(path, fig)
end

function plot!(fig::GLMakie.Figure, cyl::Cylinder)
    GLMakie.mesh!(fig.content[1], GLMakie.GeometryBasics.Cylinder3{Float32}(GLMakie.Point3f(cyl.x, cyl.y, -1.0), GLMakie.Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

function render!(sol, dim::OneDim; path, dt = 0.1)

    fig = plot(dim)

    n = Int(round((sol.prob.tspan[end] - sol.prob.tspan[1]) / dt))

    GLMakie.record(fig, path, 1:n) do i
        t = dt * i
        GLMakie.empty!(fig.content[1].scene)
        GLMakie.lines!(fig.content[1], dim.x, sol(t)[:, 1], color = :blue, linewidth = 2)
    end

    return nothing
end

function render!(sol, dim::TwoDim; path, dt = 0.1, design = nothing)

    fig = plot(dim)

    n = Int(round((sol.prob.tspan[end] - sol.prob.tspan[1]) / dt))

    GLMakie.record(fig, path, 1:n) do i
        t = dt * i
        GLMakie.empty!(fig.content[1].scene)
        if !isnothing(design)
            plot!(fig, design(t))
        end
        GLMakie.surface!(fig.content[1], dim.x, dim.y, sol(t)[:, :, 1], colormap = :ice, linewidth = 2)
    end

    return nothing
end

function render!(sol, dim::ThreeDim; path::String, dt = 0.1, design = nothing)

    fig = plot(dim)
    n = Int(round((sol.prob.tspan[end] - sol.prob.tspan[1]) / dt))

    GLMakie.record(fig, path, 1:n) do i
        t = dt * i
        GLMakie.empty!(fig.content[1].scene)
        if !isnothing(design)
            plot!(fig, design(t))
        end

        GLMakie.volume!(fig.content[1], dim.x, dim.y, dim.z, sol(t)[:, :, :, 1], colormap = GLMakie.Reverse(:ice))
    end

    return nothing
end