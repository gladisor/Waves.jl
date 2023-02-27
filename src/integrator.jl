export WaveIntegrator, integrate, runge_kutta

"""
Comprised of two functions:
    f(u, dyn) = du/dt
    integration_function(f, u, dyn)
"""
mutable struct WaveIntegrator
    wave::Wave
    f::Function
    integration_function::Function
    dyn::WaveDynamics
end

function step!(iter::WaveIntegrator)
    iter.wave = iter.integration_function(iter.f, iter.wave, iter.dyn)
    iter.dyn.t += 1
end

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

function runge_kutta(f::Function, wave::Wave, dyn::WaveDynamics)
    h = dyn.dt
    t = dyn.t * h

    k1 = f(wave,                   t,            dyn) ## Euler
    k2 = f(wave + 0.5f0 * h * k1, t + 0.5f0 * h, dyn) ## Midpoint
    k3 = f(wave + 0.5f0 * h * k2, t + 0.5f0 * h, dyn)
    k4 = f(wave +         h * k3, t +         h, dyn) ## Endpoint

    return wave + 1/6f0 * h * (k1 + 2*k2 + 2*k3 + k4)
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