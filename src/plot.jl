export WavePlot, plot_wave!, render!, plot_comparison!, plot_loss!

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

function plot_design!(p::WavePlot, design::AbstractDesign)
    mesh!(p.ax, design)
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
        design::Union{DesignTrajectory, Nothing} = nothing; 
        path::String, tmax::Float32 = 5.0f0)

    p = WavePlot(sol.dim)

    wave_interp = linear_interpolation(sol.t, sol.u)

    if !isnothing(design)
        design_interp = linear_interpolation(sol.t, design.traj)
    end

    # n_frames = Int(round(fps * sol.t[end]))

    n_frames = Int(round(24 * tmax))
    t = collect(range(sol.t[1], sol.t[end], n_frames))

    record(p.fig, path, 1:n_frames) do i

        empty!(p.ax)
        plot_wave!(p, sol.dim, wave_interp(t[i]))

        if !isnothing(design)
            plot_design!(p, design_interp(t[i]))
        end
    end
end

function Waves.render!(traj::Trajectory; kwargs...)
    states = traj.traces.state[2:end]
    actions = traj.traces.action[1:end-1]

    design = DesignTrajectory[]

    for (s, a) ∈ zip(states, actions)
        interp = DesignInterpolator(s.design, a, s.sol.total.t[1], s.sol.total.t[end])
        dt = DesignTrajectory(interp, length(s.sol.total)-1)
        push!(design, dt)
    end

    sol = WaveSol([s.sol.total for s ∈ states]...)
    design = DesignTrajectory(design...)

    render!(sol, design; kwargs...)
end

function render!(policy::AbstractPolicy, env::WaveEnv; kwargs...)
    traj = episode_trajectory(env)
    agent = Agent(policy, traj)
    run(agent, env, StopWhenDone())
    render!(traj; kwargs...)
end

function plot_comparison!(dim, y_true, y_pred; path::String)
    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = AxisAspect(1.0))
    heatmap!(ax1, dim.x, dim.y, y_true[:, :, 1, end], colormap = :ice)
    ax2 = Axis(fig[1, 2], aspect = AxisAspect(1.0))
    heatmap!(ax2, dim.x, dim.y, y_pred[:, :, 1, end], colormap = :ice)
    ax3 = Axis(fig[2, 1], aspect = AxisAspect(1.0))
    heatmap!(ax3, dim.x, dim.y, y_true[:, :, 1, end ÷ 2], colormap = :ice)
    ax4 = Axis(fig[2, 2], aspect = AxisAspect(1.0))
    heatmap!(ax4, dim.x, dim.y, y_pred[:, :, 1, end ÷ 2], colormap = :ice)
    ax5 = Axis(fig[3, 1], aspect = AxisAspect(1.0))
    heatmap!(ax5, dim.x, dim.y, y_true[:, :, 1, 1], colormap = :ice)
    ax6 = Axis(fig[3, 2], aspect = AxisAspect(1.0))
    heatmap!(ax6, dim.x, dim.y, y_pred[:, :, 1, 1], colormap = :ice)
    save(path, fig)
end

function plot_loss!(train_loss::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Gradient Update", ylabel = "Loss", title = "Training Loss", aspect = 1.0)
    lines!(ax, train_loss, linewidth = 3)
    save(path, fig)
end

function plot_loss!(train_loss::Vector{Float32}, test_loss::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Epoch", ylabel = "Average Loss", title = "Average Loss per Epoch", aspect = 1.0)
    lines!(ax, train_loss, linewidth = 3, color = :blue, label = "Training")
    lines!(ax, test_loss, linewidth = 3, color = :orange, label = "Testing")
    axislegend(ax)
    save(path, fig)
end