export WaveSim, get_sol, render!, reset!

mutable struct WaveSim{D <: AbstractDim}
    sys::PDESystem
    disc::MOLFiniteDifference
    grid::Dict
    prob::ODEProblem
    iter::ODEIntegrator
    wave::Wave{D}
end

function WaveSim(;wave::Wave, ic::InitialCondition, t_max::Real, speed::Real, n::Int)
    @named sys = PDESystem(
        wave_equation(wave),
        conditions(wave, ic),
        get_domain(wave, t_max = t_max),
        spacetime(wave), 
        [signature(wave)], 
        [wave.speed => speed])

    disc = MOLFiniteDifference([Pair.(Waves.dims(wave), n)...], wave.t)
    prob = discretize(sys, disc)
    grid = get_discrete(sys, disc)
    iter = init(prob, Tsit5(), advance_to_tstop = true, saveat = 0.05)
    return WaveSim(sys, disc, grid, prob, iter, wave)
end

function Waves.step!(sim::WaveSim, dt::Real)
    add_tstop!(sim.iter, sim.iter.t + dt)
    step!(sim.iter)
end

function Waves.step!(sim::WaveSim)
    step!(sim.iter)
end

function get_sol(sim::WaveSim)
    return sim.iter.sol[sim.grid[signature(sim.wave)]]
end

function Waves.dims(sim::WaveSim)::Vector
    return [collect(sim.grid[d]) for d ∈ dims(sim.wave)]
end

function Base.display(sim::WaveSim)
    display(typeof(sim))
end

function render!(sim::WaveSim{OneDim}; path::String)
    fig = Figure(resolution = (1920, 1080), fontsize = 20)
    ax = Axis(fig[1, 1], title = "1D Wave", xlabel = "X", ylabel = "Y")

    xlims!(ax, getbounds(sim.wave.dim.x)...)
    ylims!(ax, -1.0, 1.0)

    x = dims(sim)[1]
    sol = get_sol(sim)

    record(fig, path, axes(sol, 1)) do i
        empty!(ax.scene)
        lines!(ax, x, sol[i], linestyle = nothing, linewidth = 5, color = :blue)
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

    x, y = dims(sim)
    sol = get_sol(sim)

    record(fig, path, axes(sol, 1)) do i
        empty!(ax.scene)
        surface!(ax, x, y, sol[i], colormap = :ice, shading = false)
    end

    return nothing
end

function reset!(sim::WaveSim)
    reinit!(sim.iter)
    return nothing
end