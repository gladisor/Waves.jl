using ModelingToolkit
using Waves
using Waves: unpack, dirichlet, neumann, time_condition

"""
Wave equation in three dimensions.
"""
function Waves.wave_equation(wave::Wave{ThreeDim}, C::Function)::Equation
    x, y, z, t, u = unpack(wave)

    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2
    Dtt = Differential(t)^2

    return Dtt(u(x, y, z, t)) ~ C(x, y, z, t)^2 * (Dxx(u(x, y, z, t)) + Dyy(u(x, y, z, t)) + Dzz(u(x, y, z, t)))
end

function Waves.dirichlet(wave::Wave{ThreeDim})::Vector{Equation}
    x, y, z, t, u = unpack(wave)
    x_min, x_max = getbounds(x)
    y_min, y_max = getbounds(y)
    z_min, z_max = getbounds(z)

    return [
        u(x_min, y, z, t) ~ 0., 
        u(x_max, y, z, t) ~ 0.,
        u(x, y_min, z, t) ~ 0.,
        u(x, y_max, z, t) ~ 0.,
        u(x, y, z_min, t) ~ 0.,
        u(x, y, z_max, t) ~ 0.]
end

function Waves.neumann(wave::Wave{ThreeDim})::Vector{Equation}
    x, y, z, t, u = unpack(wave)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    x_min, x_max = getbounds(x)
    y_min, y_max = getbounds(y)
    z_min, z_max = getbounds(z)

    return [
        Dx(u(x_min, y, z, t)) ~ 0., 
        Dx(u(x_max, y, z, t)) ~ 0.,
        Dy(u(x, y_min, z, t)) ~ 0.,
        Dy(u(x, y_max, z, t)) ~ 0.,
        Dz(u(x, y, z_min, t)) ~ 0.,
        Dz(u(x, y, z_max, t)) ~ 0.
        ]
end

function Waves.time_condition(wave::Wave{ThreeDim})::Equation
    x, y, z, t, u = unpack(wave)
    Dt = Differential(t)
    return Dt(u(x, y, z, 0.0)) ~ 0.
end

gs = 3.0
wave = Wave(dim = ThreeDim(-gs, gs, -gs, gs, -gs, gs))
pulse = GaussianPulse(intensity = 1.0)

sim = WaveSim(
    wave = wave,
    ic = pulse,
    speed = 1.0,
    t_max = 5.0,
    n = 30,
    dt = 0.05)