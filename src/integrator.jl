export WaveIntegrator, integrate

"""
Comprised of two functions:
    f(u, dyn) = du/dt
    integration_function(f, u, dyn)


    wave:                   current state of all fields of the wave
    f:                      function which approximates the change in the wave over time
    integration_function:   scheme for integrating to the next timestep
    dyn:                    dynamics of the wave
"""
mutable struct WaveIntegrator
    wave::AbstractArray{Float32}
    f::Function
    integration_function::Function
    dyn::WaveDynamics
end

"""
Resets the dynaimcs time to zero
"""
function reset!(iter::WaveIntegrator)
    reset!(iter.dyn)
end

"""
Updates the wave to its next state according to the dynamics
increments the system time.
"""
function step!(iter::WaveIntegrator)
    iter.wave = iter.wave .+ iter.integration_function(iter.f, iter.wave, iter.dyn)
    iter.dyn.t += 1
end

"""
Performs n integration steps and returns a solution containing the trajectory
of the wave over those steps.
"""
function integrate(iter::WaveIntegrator, n::Int)
    t = Float32[time(iter.dyn)]
    u = displacement(iter.wave)
    sol = typeof(u)[u]
    tf = iter.dyn.t + n

    while iter.dyn.t < tf
        step!(iter)
        push!(t, time(iter.dyn))
        push!(sol, displacement(iter.wave))
    end

    return WaveSol(iter.dyn.dim, t, sol)
end

function Flux.gpu(iter::WaveIntegrator)
    return WaveIntegrator(
        gpu(iter.wave),
        iter.f,
        iter.integration_function,
        gpu(iter.dyn)
    )
end

function Flux.cpu(iter::WaveIntegrator)
    return WaveIntegrator(
        cpu(iter.wave),
        iter.f,
        iter.integration_function,
        cpu(iter.dyn)
    )
end