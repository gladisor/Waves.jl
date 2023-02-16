using GLMakie
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using ReinforcementLearning

using Waves
using Waves: AbstractDesign

include("../src/env.jl")

gs = 10.0
dx = 0.05
dt = 0.05
ambient_speed = 2.0
pml_width = 4.0
pml_scale = 20.0
tspan = (0.0, 20.0)

dim = TwoDim(gs, dx)
g = grid(dim)
u = pulse(dim)
u0 = cat(u, zeros(size(u)..., 2), dims = 3)

grad = gradient(dim.x)
design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.2), Cylinder(0.0, 0.0, 0.0, 0.0), tspan...)
C_inc = SpeedField(dim, ambient_speed)
C = SpeedField(dim, ambient_speed, design)
pml = build_pml(dim, pml_width, pml_scale)

prob_inc = ODEProblem(wave!, u0, tspan, [grad, C_inc, pml])
prob = ODEProblem(wave!, u0, tspan, [grad, C, pml])

@time sol_inc = solve(prob_inc, ORK256(), dt = dt)
reward_signal = ScatteredFlux(sol_inc, WaveFlux(dim, circle_mask(dim, 6.0)))

iter = init(
    prob,
    ORK256(), 
    advance_to_tstop = true, 
    dt = dt
    )

env = WaveEnv(iter, C, 0.5, reward_signal)
designs = Vector{DesignInterpolator}()

@time while !is_terminated(env)
    action = Cylinder(rand(action_space(env))..., 0.0, 0.0)
    env(action)
    println("Time: $(env.iter.t)")
    push!(designs, env.C.design)
end

sol = Waves.interpolate(env.iter.sol, dim, 0.1)
design_interp = Waves.interpolate(designs, 0.1)
Waves.render!(sol, design_interp, path = "env_dx=$(dx)_dt=$(dt).mp4")