using DifferentialEquations
using Waves
using Waves: AbstractDesign
import GLMakie

include("configuration.jl")

gs = 8.0
Δ = 0.3
pml_width = 2.0
pml_scale = 10.0
dim = TwoDim(gs, Δ)
tspan = (0.0, 20.0)

M = 8
r = 0.5
c = 0.2
config = Configuration(dim, M = M, r = r, c = c, offset = pml_width)
final = Configuration(dim, M = M, r = r, c = c, offset = pml_width)
action = final - config

design = DesignInterpolator(config, action, tspan...)
C = WaveSpeed(dim, 1.0, design)

u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)
pml = build_pml(dim, pml_width) * pml_scale
p = [Δ, C, pml]

@time prob = ODEProblem(split_wave!, u0, tspan, p)
@time sol = solve(prob, Midpoint())
@time render!(sol, dim, path = "vid.mp4", design = design)
