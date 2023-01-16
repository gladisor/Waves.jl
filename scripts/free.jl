using ModelingToolkit
using Waves

function absorbing_condition(wave::Wave{OneDim})
    x, t, u = Waves.unpack(wave)
    x_min, x_max = getbounds(x)

    Dx = Differential(x)
    Dt = Differential(t)

    return [
        Dt(u(x_min, t)) - wave.speed * Dx(u(x_min, t)) ~ 0.,
        Dt(u(x_max, t)) + wave.speed * Dx(u(x_max, t)) ~ 0.]
end

function absorbing_condition(wave::Wave{TwoDim})
    x, y, t, u = Waves.unpack(wave)
    x_min, x_max = getbounds(x)
    y_min, y_max = getbounds(y)

    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)
    # Dxx = Differential(x) ^ 2
    # Dyy = Differential(y) ^ 2
    # Dtt = Differential(t) ^ 2

    return [
        Dt(u(x_min, y, t)) - wave.speed * Dx(u(x_min, y, t)) ~ 0.,
        Dt(u(x_max, y, t)) + wave.speed * Dx(u(x_max, y, t)) ~ 0.,
        Dt(u(x, y_min, t)) - wave.speed * Dy(u(x, y_min, t)) ~ 0.,
        Dt(u(x, y_max, t)) + wave.speed * Dy(u(x, y_max, t)) ~ 0.]
end

function Waves.boundary_conditions(wave::Wave)::Vector{Equation}
    return vcat(absorbing_condition(wave), [Waves.time_condition(wave)])
end

## simulation hyperparameters
gs = 5.0
kwargs = Dict(:tmax => 20.0, :speed => 1.0, :n => 30, :dt => 0.05)
dp = (0.0, 0.0, 0.7, 0.2)

## important objects
wave = Wave(
    dim = OneDim(-gs, gs)
    # dim = TwoDim(-gs, gs, -gs, gs)
    )

pulse = GaussianPulse(intensity = 1.0)
sim = WaveSim(wave = wave, ic = pulse; kwargs...)
Waves.step!(sim)
sol = WaveSol(sim)

render!(sol, path = "test_2D.mp4")