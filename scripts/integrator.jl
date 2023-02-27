using Waves
using Waves: runge_kutta
using CairoMakie

"""
Comprised of two functions:
    f(u, dyn) = du/dt
    integration_function(f, u, dyn)
"""
mutable struct WaveIntegrator
    f::Function
    integration_function::Function
    dyn::WaveDynamics
end

function Base.step(iter::WaveIntegrator, wave::Wave)
    u = iter.integration_function(iter.f, wave, iter.dyn)
    iter.dyn.t += 1
    return u
end

function integrate(iter::WaveIntegrator, wave::Wave, n::Int)
    t = Float32[time(iter.dyn)]
    u = displacement(wave)
    sol = typeof(u)[u]
    tf = iter.dyn.t + n

    while iter.dyn.t < tf
        wave = step(iter, wave)
        push!(t, time(iter.dyn))
        push!(sol, displacement(wave))
    end

    return WaveSol(iter.dyn.dim, t, sol)
end

function integrate(iter::WaveIntegrator, u, n::Int)
    t = Float32[time(iter.dyn)]
    sol = typeof(u)[u]
    tf = iter.dyn.t + n

    while iter.dyn.t < tf
        u = iter.integration_function(iter.f, u, iter.dyn)
        iter.dyn.t += 1
        push!(t, time(iter.dyn))
        push!(sol, u)
    end

    return WaveSol(iter.dyn.dim, t, sol)
end

pml_width = 1.0f0
pml_scale = 100.0f0
ambient_speed = 2.0f0
dt = 0.01f0
n = 600

dim = TwoDim(5.0f0, 300)

cyl = Cylinder([0.0f0, -3.0f0], 1.0f0, 0.1f0)
action = Cylinder([0.0f0, 3.0f0], 0.0f0, 0.0f0)

wave = Wave(dim, 6)
ic = Pulse(dim, 0.0f0, 0.0f0, 5.0f0)

wave = ic(wave)
dyn = WaveDynamics(design = cyl, dim = dim, pml_width = pml_width, pml_scale = pml_scale, ambient_speed = ambient_speed, dt = dt)
dyn.design = DesignInterpolator(cyl, action, 0.0f0, dyn.dt * n)
iter = WaveIntegrator(split_wave_pml, runge_kutta, dyn)

@time sol = integrate(iter, wave, n)

wave = ic(wave)
dyn = WaveDynamics(design = cyl, dim = dim, pml_width = pml_width, pml_scale = pml_scale, ambient_speed = ambient_speed, dt = dt)
dyn.design = DesignInterpolator(cyl, action, 0.0f0, dyn.dt * n)
iter = WaveIntegrator(split_wave_pml, runge_kutta, dyn)

;
@time sol = integrate(iter, wave.u, n)
;
@time render!(sol, path = "sol_wave.mp4")
@time render!(sol, path = "sol_u.mp4")