using Waves
using Waves: wave_equation, wave_speed, unpack
using ModelingToolkit: Equation, Differential
using IfElse

gs = 5.0
pml_width = 1.0
pml_start = gs - pml_width

# σ = -3/4*log(1e-12) * pml_width # σ_0
λ = 1.0          # Wavelength (arbitrary unit)
k = 2*π/λ        # Wave number

dim = OneDim(size = gs)
wave = Wave(dim = dim)

# function Waves.wave_equation(wave::Wave{OneDim}, C::Function)::Equation
#     x, t, u = unpack(wave)
#     Dxx = Differential(x)^2
#     Dtt = Differential(t)^2

#     x_pml = abs(x) - pml_start
#     s_x = IfElse.ifelse(x_pml > 0, 1 + (1im * σ / k) * x_pml, 1.0)
#     x̂ = IfElse.ifelse(x_pml > 0, x_pml + 1im * σ/k, x)
#     c_pml = IfElse.ifelse(x_pml > 0, 1im * σ * x_pml / k, 0.0)

#     return Dtt(u(x, t)) ~ (C(x, t) + c_pml) ^ 2 * (1/(s_x^2)) * Dxx(u(x̂, t))
# end

function Waves.wave_equation(wave::Wave{OneDim}, C::Function)
    x, t, u = unpack(wave)
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    x_pml = x -> max(abs(x) - pml_start, 0.0)

    return Dtt(u(x, t)) ~ 1 / (1 + 1im *(x_pml(x) / pml_width) / (k * C(x, t))) ^ 2 * (C(x, t) * (1 + 1im *(x_pml(x) / pml_width))) ^ 2 * Dxx(u(x, t))
end

kwargs = Dict(:wave => wave, :ic => GaussianPulse(), :boundary => ClosedBoundary(), :ambient_speed => 2.0, :tmax => 10.0, :n => 100, :dt => 0.05)
sim_tot = WaveSim(;kwargs...)

propagate!(sim_tot)
render!(WaveSol(sim_tot), path = "test.mp4")
