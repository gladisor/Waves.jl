function plot_solution!(dim::OneDim, tspan::Vector{Float32}, u::AbstractArray{Float32, 3}; path::String)
    fig = Figure()
    ax1 = Axis(fig[1, 1], title = "Displacement of Solution Over Time", ylabel = "Time (s)", xlabel = "Distance (m)")
    heatmap!(ax1, dim.x, tspan, u[:, 1, :], colormap = :ice)
    save(path, fig)
end

function plot_wave!(dim::OneDim, wave::AbstractMatrix{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, dim.x, wave[:, 1])
    lines!(ax, dim.x, wave[:, 2])
    save(path, fig)
end