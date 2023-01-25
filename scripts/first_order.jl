using Waves
using Waves: get_domain, spacetime, signature, wave_discretizer, wave_equation, unpack
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
c = 1.0

dim = OneDim(size = gs)
wave = Wave(dim = dim)

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(), 
    :boundary => ClosedBoundary(), 
    :ambient_speed => c, 
    :tmax => 20.0, :n => 100, :dt => 0.05)

ps = [
    wave.speed => kwargs[:ambient_speed]
]

eq = wave_equation(wave)
@variables v(..), w(..)
x, t, u = unpack(wave)
Dx = Differential(x)
Dxx = Differential(x)^2
Dt = Differential(t)
Dtt = Differential(t)^2

x, t, u = unpack(wave)
x_min, x_max = getbounds(x)

bcs = [
    u(x, 0.) ~ kwargs[:ic](wave), 
    u(x_min, t) ~ 0., 
    u(x_max, t) ~ 0., 
    Dx(u(x_min, t)) ~ 0.,
    Dx(u(x_max, t)) ~ 0.,
    Dt(u(x, 0.)) ~ 0.,
    v(x, 0.) ~ 0.,
    ]

eqs = [
    v(x, t) ~ Dt(u(x, t))
    Dt(v(x, t)) ~ Dxx(u(x, t))
    # Dtt(u(x, t)) ~ Dxx(u(x, t))
    ]

@named sys = PDESystem(
    eqs, 
    bcs, 
    get_domain(wave, tmax = kwargs[:tmax]), 
    spacetime(wave), 
    [u(x, t), v(x, t)], 
    ps)

disc = wave_discretizer(wave, kwargs[:n])
iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])

propagate!(sim)
render!(WaveSol(sim), path = "first_order.mp4")
