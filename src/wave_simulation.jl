export WaveSimulation, ∫ₜ, animate!

mutable struct WaveSimulation{W <: AbstractWave}
    wave::W
    prob::ODEProblem
end

function WaveSimulation(wave::Wave1D; ic::Function, C::Function, n::Int, p = [])
    x, t, u = wave.x, wave.t, wave.u
    t_min, _ = getbounds(t)
    bcs = [u(x, t_min) ~ ic(x), boundary_conditions(wave)...]

    domain = [
        x ∈ getbounds(x),
        t ∈ getbounds(t),]

    @named sys = PDESystem(wave_equation(wave, C), bcs, domain, [x, t], [u(x, t)], p)
    disc = MOLFiniteDifference([x => n], t)
    prob = discretize(sys, disc)
    return WaveSimulation(wave, prob)
end

function WaveSimulation(wave::Wave2D; ic::Function, C::Function, n::Int, p = [])
    x, y, t, u = wave.x, wave.y, wave.t, wave.u
    t_min, _ = getbounds(t)
    bcs = [u(x, y, t_min) ~ ic(x, y), boundary_conditions(wave)...]

    domain = [
        x ∈ getbounds(x),
        y ∈ getbounds(y),
        t ∈ getbounds(t)]

    @named sys = PDESystem(wave_equation(wave, C), bcs, domain, [x, y, t], [u(x, y, t)], p)
    disc = MOLFiniteDifference([x => n, y => n], t)
    prob = discretize(sys, disc)
    return WaveSimulation(wave, prob)
end

function ∫ₜ(sim::WaveSimulation{Wave1D}; dt)::WaveSolution1D
    sol = solve(sim.prob, Tsit5(), saveat = dt)
    x = collect(sol.ivdomain[2])
    t = sol.t
    u = sol[sim.wave.u(sim.wave.x, sim.wave.t)]
    return WaveSolution1D(x, t, u)
end

function ∫ₜ(sim::WaveSimulation{Wave1D}; dt)::WaveSolution1D
    sol = solve(sim.prob, Tsit5(), saveat = dt)
    x = collect(sol.ivdomain[2])
    t = sol.t
    u = sol[sim.wave.u(sim.wave.x, sim.wave.t)]
    return WaveSolution1D(x, t, u)
end

function ∫ₜ(sim::WaveSimulation{Wave2D}; dt)::WaveSolution2D
    sol = solve(sim.prob, Tsit5(), saveat = dt)
    x = collect(sol.ivdomain[2])
    y = collect(sol.ivdomain[3])
    t = sol.t
    u = sol[sim.wave.u(sim.wave.x, sim.wave.y, sim.wave.t)]
    return WaveSolution2D(x, y, t, u)
end