using Waves
using Waves: wave_equation, wave_speed, unpack, get_domain, spacetime, signature, wave_discretizer
using ModelingToolkit: Equation, Differential, @named, PDESystem
using MethodOfLines
using OrdinaryDiffEq: Tsit5, init
using IfElse

gs = 15.0
pml_width = 10.0
pml_start = gs - pml_width
c = 1.0
ω = 1.0
k = ω / c

function σₓ(x)
    x_pml = abs(x) - pml_start
    return IfElse.ifelse(x_pml > 0.0, x_pml / pml_width, 0.0)^2
end

function Waves.wave_equation(wave::Wave{OneDim}, C::Function)
    x, t, u = unpack(wave)
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    Dt = Differential(t)
    s_x = 1/(1 + 1im * σₓ(x)) ^ 2
    eq = Dtt(u(x, t)) + σₓ(x) * Dt(u(x, t)) ~ s_x * C(x, t) ^ 2 * Dxx(u(x, t))
    return eq[1]
end

dim = OneDim(size = gs)
wave = Wave(dim = dim)

kwargs = Dict(:wave => wave, :ic => GaussianPulse(), :boundary => ClosedBoundary(), :ambient_speed => c, :tmax => 20.0, :n => 200, :dt => 0.05)
sim = WaveSim(;kwargs...)
propagate!(sim)
render!(WaveSol(sim), path = "pml_$(pml_width).mp4")

