export WaveSim, propagate!

mutable struct WaveSim{D <: AbstractDim}
    wave::Wave{D}
    grid::Dict
    iter::ODEIntegrator
    dt::Real
end

function WaveSim(;wave::Wave, ic::InitialCondition, tmax::Real, n::Int, dt::Real, ambient_speed::Real = 1.0, design::Union{ParameterizedDesign, Nothing} = nothing, boundary::WaveBoundary = OpenBoundary())

    ps = [wave.speed => ambient_speed]

    if isnothing(design)
        eq = wave_equation(wave)
    else
        eq = wave_equation(wave, design)
        ps = vcat(ps, design_parameters(design, design.design, 0.0, tmax))
    end

    bcs = [
        wave.u(dims(wave)..., 0.0) ~ ic(wave), 
        boundary(wave)...
        ]

    @named sys = PDESystem(eq, bcs, get_domain(wave, tmax = tmax), spacetime(wave), [signature(wave)], ps)
    disc = wave_discretizer(wave, n)
    iter = init(discretize(sys, disc), Tsit5(), advance_to_tstop = true, saveat = dt)
    return WaveSim(wave, get_discrete(sys, disc), iter, dt)
end

function propagate!(sim::WaveSim)
    step!(sim.iter)
    return nothing
end

function Base.display(sim::WaveSim)
    println(typeof(sim))
end

function reset!(sim::WaveSim)
    reinit!(sim.iter)
    return nothing
end

function get_data(sim::WaveSim)
    return sim.iter.sol[sim.grid[signature(sim.wave)]]
end

function state(sim::WaveSim)
    return sim.iter[sim.grid[signature(sim.wave)]]
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

function tspan(sim::WaveSim)
    return sim.iter.sol.prob.tspan
end

function get_tmax(sim::WaveSim)
    return tspan(sim)[end]
end

function current_time(sim::WaveSim)
    return sim.iter.t
end