export WaveSol, render!

struct WaveSol{D <: AbstractDim}
    wave::Wave{D}
    dims::Vector
    tspan::Tuple
    data::AbstractArray
end

function WaveSol(sim::WaveSim)
    return WaveSol(sim.wave, dims(sim), tspan(sim), get_data(sim))
end

function Base.display(sol::WaveSol)
    println(typeof(sol))
end

function Base.:-(sol::WaveSol, other::WaveSol)::WaveSol
    data = sol.data .- other.data
    return WaveSol(sol.wave, sol.dims, sol.tspan, data)
end

function render!(sol::WaveSol{OneDim}; path::String)
    fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
    ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X", ylabel = "Y")

    GLMakie.xlims!(ax, getbounds(sol.wave.dim.x)...)
    GLMakie.ylims!(ax, -1.0, 1.0)

    x = sol.dims[1]

    GLMakie.record(fig, path, axes(sol.data, 1)) do i
        GLMakie.empty!(ax.scene)
        GLMakie.lines!(ax, x, sol.data[i], linestyle = nothing, linewidth = 5, color = :blue)
    end

    return nothing
end

function Waves.render!(
        sol::WaveSol{TwoDim}; 
        path::String, 
        design::Union{Vector{<: AbstractDesign}, Nothing} = nothing)
        
    fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
    ax = GLMakie.Axis3(fig[1,1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X", ylabel = "Y", zlabel = "Z")

    GLMakie.xlims!(ax, getbounds(sol.wave.dim.x)...)
    GLMakie.ylims!(ax, getbounds(sol.wave.dim.y)...)
    GLMakie.zlims!(ax, -1.0, 4.0)

    x, y = sol.dims

    GLMakie.record(fig, path, axes(sol.data, 1)) do i
        GLMakie.empty!(ax.scene)
        GLMakie.surface!(ax, x, y, sol.data[i], colormap = :ice, shading = false)
        if !isnothing(design)
            GLMakie.mesh!(ax, design[i])
        end
    end

    return nothing
end

function energy(x::AbstractArray)::Float64
    return sum(x .^ 2)
end

function energy(sol::WaveSol)::Vector
    return map(energy, sol.data)
end

function plot_energy!(;sol_inc::WaveSol, sol_sc::Union{WaveSol, Nothing} = nothing, path)
    inc_energy = energy(sol_inc)

    tick_length = length(inc_energy)
    old_ticks = collect(1:100:tick_length)
    new_ticks = collect(range(0, sol_inc.tspan[end], length = length(old_ticks)))

    fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 50)
    ax = GLMakie.Axis(fig[1, 1], 
        title = "Scattered Wave Energy Over Time",
        xlabel = "Time", ylabel = "Wave Energy: Σx²",
        xticks = (old_ticks,  string.(new_ticks)))

    GLMakie.lines!(ax, inc_energy, linewidth = 8, label = "Incident")

    if !isnothing(sol_sc)
        sc_energy = energy(sol_sc)
        GLMakie.lines!(ax, sc_energy, linewidth = 8, label = "Scattered")
    end

    GLMakie.Legend(fig[1, 2], ax, "Wave")
    GLMakie.save(path, fig)
    return nothing
end