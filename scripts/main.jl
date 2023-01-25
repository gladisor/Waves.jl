using Waves
using Waves: get_domain, spacetime, signature, wave_discretizer, wave_equation, unpack
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
c = 1.0

pml_width = 2.0
pml_start = gs - pml_width
c = 1.0
ω = 1.0
k = ω / c

function σₓ(x)
    x_pml = abs(x) - pml_start
    return IfElse.ifelse(x_pml > 0.0, x_pml / pml_width, 0.0)
end


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
    Dx(u(x_min, t)) ~ 0., v(x_min, t) ~ 0.,
    Dx(u(x_max, t)) ~ 0., v(x_max, t) ~ 0.,
    Dt(u(x, 0.)) ~ 0., w(x, 0.) ~ 0., v(x, 0.) ~ 0.,
    # w(x, 0.0) ~ 0.0,
    # v(x_min, t) ~ 0.0,
    # v(x_max, t) ~ 0.0,
    # v(x_min, t) ~ 0.0,
    # v(x_max, t) ~ 0.0,
    # w(x_min, t) ~ 0.0,
    # w(x_max, t) ~ 0.0,
    # Dx(v(x_min, t)) ~ 0.0,
    # Dx(v(x_max, t)) ~ 0.0,
    # Dx(w(x_min, t)) ~ 0.0,
    # Dx(w(x_max, t)) ~ 0.0
    ]

eqs = [
    v(x, t) ~ Dx(u(x, t)),
    w(x, t) ~ Dt(u(x, t)),
    Dt(w(x, t)) ~ Dx(v(x, t))
    # Dx(u(x, t)) ~ v(x, t),
    # Dt(w(x, t)) ~ c^2 * Dx(v(x, t))
    # Dtt(u(x, t)) ~ Dxx(u(x, t))
]

@named sys = PDESystem(
    eqs, 
    bcs, 
    get_domain(wave, tmax = kwargs[:tmax]), 
    spacetime(wave), 
    [u(x, t), v(x, t), w(x, t)], 
    ps)

disc = wave_discretizer(wave, kwargs[:n])
iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])

propagate!(sim)
render!(WaveSol(sim), path = "simple.mp4")
