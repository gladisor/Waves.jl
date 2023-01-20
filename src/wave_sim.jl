export WaveSim, get_data, render!, reset!, WaveSol, energy, plot_energy!

mutable struct WaveSim{D <: AbstractDim}
    sys::PDESystem
    disc::MOLFiniteDifference
    grid::Dict
    prob::ODEProblem
    iter::ODEIntegrator
    wave::Wave{D}
    dt::Real
end

function WaveSim(;
        wave::Wave, 
        design::Union{ParameterizedDesign, Nothing} = nothing,
        ic::InitialCondition, 
        open::Bool,
        tmax::Real, 
        ambient_speed::Real, 
        n::Int, 
        dt::Real)

    ps = [wave.speed => ambient_speed]

    if isnothing(design)
        eq = wave_equation(wave)
    else
        eq = wave_equation(wave, design)
        ps = vcat(ps, design_parameters(design, design.design, 0.0, tmax))
    end

    bcs = [
        wave.u(dims(wave)..., 0.0) ~ ic(wave)
        ]

    if open
        bcs = vcat(bcs, open_boundary(wave))
    else
        bcs = vcat(bcs, closed_boundary(wave))
    end

    @named sys = PDESystem(eq, bcs, get_domain(wave, tmax = tmax), spacetime(wave), [signature(wave)], ps)

    disc = wave_discretizer(wave, n)
    prob = discretize(sys, disc)
    grid = get_discrete(sys, disc)
    iter = init(prob, Tsit5(), advance_to_tstop = true, saveat = dt)
    return WaveSim(sys, disc, grid, prob, iter, wave, dt)
end

function Waves.step!(sim::WaveSim, dt::Real)
    add_tstop!(sim.iter, sim.iter.t + dt)
    step!(sim.iter)
end

function Waves.step!(sim::WaveSim)
    step!(sim.iter)
end

function Base.display(sim::WaveSim)
    display(typeof(sim))
end

function reset!(sim::WaveSim)
    reinit!(sim.iter)
    return nothing
end

function get_data(sim::WaveSim)
    return sim.iter.sol[sim.grid[signature(sim.wave)]]
end

function state(sim::WaveSim)
    return sim.iter[sim.grid[Waves.signature(sim.wave)]]
end

function dims(sim::WaveSim)::Vector
    return [collect(sim.grid[d]) for d ∈ dims(sim.wave)]
end

function set_t0!(sim::WaveSim, t0)
    sim.iter.p[end-1] = t0
    return nothing
end

function set_tf!(sim::WaveSim, tf)
    sim.iter.p[end] = tf
    return nothing
end

function set_design_params!(sim::WaveSim, dp)
    sim.iter.p[2:end-2] .= dp
    return nothing
end

function set_wave_speed!(sim::WaveSim, speed)
    sim.iter.p[1] = speed
    return nothing
end

function get_tmax(sim::WaveSim)
    return sim.prob.tspan[end]
end

function current_time(sim::WaveSim)
    return sim.iter.t
end

struct WaveSol{D <: AbstractDim}
    wave::Wave{D}
    dims::Vector
    tspan::Tuple
    data::AbstractArray
end

function WaveSol(sim::WaveSim)
    return WaveSol(sim.wave, dims(sim), sim.prob.tspan, get_data(sim))
end

function Base.display(sol::WaveSol)
    display(typeof(sol))
end

function Base.:-(sol::WaveSol, other::WaveSol)
    data = sol.data .- other.data
    return WaveSol(sol.wave, sol.dims, sol.tspan, data)
end

function render!(sol::WaveSol{OneDim}; path::String)
    fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
    ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X", ylabel = "Y")

    GLMakie.xlims!(ax, getbounds(sol.wave.dim.x)...)
    GLMakie.ylims!(ax, -1.0, 1.0)

    x = sol.dims[1]
    data = sol.data

    GLMakie.record(fig, path, axes(data, 1)) do i
        GLMakie.empty!(ax.scene)
        GLMakie.lines!(ax, x, data[i], linestyle = nothing, linewidth = 5, color = :blue)
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
    data = sol.data

    GLMakie.record(fig, path, axes(data, 1)) do i
        GLMakie.empty!(ax.scene)
        GLMakie.surface!(ax, x, y, data[i], colormap = :ice, shading = false)
        if !isnothing(design)
            GLMakie.mesh!(ax, design[i])
        end
    end

    return nothing
end

function energy(x::AbstractArray)
    return sum(x .^ 2)
end

function energy(sol::WaveSol)
    return map(energy, sol.data)
end

function plot_energy!(sol_inc::WaveSol, sol_sc::WaveSol; path)
    inc_energy = energy(sol_inc)
    sc_energy = energy(sol_sc)
    tick_length = length(inc_energy)
    old_ticks = collect(1:100:tick_length)
    new_ticks = collect(range(0, sol_inc.tspan[end], length = length(old_ticks)))

    fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 50)
    ax = GLMakie.Axis(fig[1, 1], 
        title = "Scattered Wave Energy Over Time",
        xlabel = "Time", ylabel = "Wave Energy: Σx²",
        xticks = (old_ticks,  string.(new_ticks)))

    GLMakie.lines!(ax, inc_energy, linewidth = 8, label = "Incident")
    GLMakie.lines!(ax, sc_energy, linewidth = 8, label = "Scattered")

    GLMakie.Legend(fig[1, 2], ax, "Wave")
    GLMakie.save(path, fig)
    return nothing
end