using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 20.0
dim = OneDim(size = gs)
wave = Wave(dim = dim)

x, t, u = Waves.unpack(wave)
x_min, x_max = getbounds(x)
Dx = Differential(x)
Dxx = Differential(x)^2
Dt = Differential(t)
Dtt = Differential(t)^2

@variables v(..), w(..)

ps = [wave.speed => 2.0]


pml_width = 10.0
pml_start = gs - pml_width

function σₓ(x)
    x_pml = abs(x) - pml_start
    return IfElse.ifelse(x_pml > 0.0, x_pml / pml_width, 0.0)
end

eq = [
    v(x, t) ~ Dt(u(x, t)),
    Dt(v(x, t)) + σₓ(x) * v(x, t) ~ Dxx(u(x, t))
    # Dtt(u(x, t)) + σₓ(x) * Dt(u(x, t)) ~ wave.speed * Dxx(u(x, t))
]

bcs = [
    u(x, 0.0) ~ exp(-x^2),
    v(x, 0.0) ~ 0.0,
    # Dx(u(x_min, t)) ~ Dt(u(x_min, t)),
    # Dx(u(x_max, t)) ~ Dt(u(x_max, t)),
    # Dt(u(x, 0.0)) ~ 0.0,
    ]

@named sys = PDESystem(
    eq, 
    bcs, 
    Waves.get_domain(wave, tmax = 40.0), 
    Waves.spacetime(wave), 
    [u(x, t), v(x, t)], ps)

disc = Waves.wave_discretizer(wave, 100)
iter = init(discretize(sys, disc), Tsit5(), advance_to_tstop = true, saveat = 0.05)
sim = WaveSim(wave, get_discrete(sys, disc), iter, 0.05)
propagate!(sim)

sol = WaveSol(sim)
render!(sol, path = "1d.mp4")