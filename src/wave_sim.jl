export WaveSim, get_data, render!, reset!, WaveSol

mutable struct WaveSim{D <: AbstractDim}
    sys::PDESystem
    disc::MOLFiniteDifference
    grid::Dict
    prob::ODEProblem
    iter::ODEIntegrator
    wave::Wave{D}
end

function WaveSim(;wave::Wave, ic::InitialCondition, t_max::Real, speed::Real, n::Int, dt::Real)
    @named sys = PDESystem(
        wave_equation(wave),
        conditions(wave, ic),
        get_domain(wave, t_max = t_max),
        spacetime(wave), 
        [signature(wave)], 
        [wave.speed => speed])

    disc = MOLFiniteDifference([Pair.(dims(wave), n)...], wave.t)
    prob = discretize(sys, disc)
    grid = get_discrete(sys, disc)
    iter = init(prob, Tsit5(), advance_to_tstop = true, saveat = dt)
    return WaveSim(sys, disc, grid, prob, iter, wave)
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

struct WaveSol{D <: AbstractDim}
    wave::Wave{D}
    grid::Dict
    sol::ODESolution
end

function WaveSol(sim::WaveSim)
    return WaveSol(sim.wave, sim.grid, sim.iter.sol)
end

function Base.display(sol::WaveSol)
    display(typeof(sol))
end

function get_data(sol::WaveSol)
    return sol.sol[sol.grid[signature(sol.wave)]]
end

function dims(s::Union{WaveSim, WaveSol})::Vector
    return [collect(s.grid[d]) for d ∈ dims(s.wave)]
end

function render!(sol::WaveSol{OneDim}; path::String)
    fig = Figure(resolution = (1920, 1080), fontsize = 20)
    ax = Axis(fig[1, 1], title = "1D Wave", xlabel = "X", ylabel = "Y")

    xlims!(ax, getbounds(sol.wave.dim.x)...)
    ylims!(ax, -1.0, 1.0)

    x = dims(sol)[1]
    data = get_data(sol)

    record(fig, path, axes(data, 1)) do i
        empty!(ax.scene)
        lines!(ax, x, data[i], linestyle = nothing, linewidth = 5, color = :blue)
    end

    return nothing
end

function render!(sim::WaveSim{TwoDim}; path::String)
    fig = Figure(resolution = (1920, 1080), fontsize = 20)
    
    ax = Axis3(fig[1,1], aspect = (1, 1, 1), perspectiveness = 0.5,
        title = "2D Wave", xlabel = "X", ylabel = "Y", zlabel = "Z")

    xlims!(ax, getbounds(sim.wave.dim.x)...)
    ylims!(ax, getbounds(sim.wave.dim.y)...)
    zlims!(ax, 0.0, 5.0)

    x, y = dims(sol)
    data = get_data(sol)

    record(fig, path, axes(data, 1)) do i
        empty!(ax.scene)
        surface!(ax, x, y, data[i], colormap = :ice, shading = false)
    end

    return nothing
end