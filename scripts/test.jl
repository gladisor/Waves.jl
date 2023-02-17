println("Loading libraries")
import Flux
## diffeq
using DifferentialEquations
## rl
using ReinforcementLearning
## local source
using Waves

gs = 10.0
dx = 0.025
dt = 0.025
ambient_speed = 1.0
pml_width = 4.0
pml_scale = 20.0
tspan = (0.0, 20.0)

println("Constructing components")
dim = TwoDim(gs, dx)
u = pulse(dim)
u0 = cat(u, zeros(size(u)..., 2), dims = 3) |> Flux.gpu
grad = gradient(dim.x) |> Flux.gpu
design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.2), Cylinder(0.0, 0.0, 0.0, 0.0), tspan...)
C_inc = SpeedField(dim, ambient_speed)
C = SpeedField(dim, ambient_speed, design)
pml = build_pml(dim, pml_width, pml_scale) |> Flux.gpu
prob_inc = ODEProblem(wave!, u0, tspan, [grad, C_inc, pml])
prob = ODEProblem(wave!, u0, tspan, [grad, C, pml])

println("Solving")
@time sol_inc = solve(prob_inc, ORK256(), dt = dt)

reward_signal = ScatteredFlux(sol_inc, WaveFlux(dim, circle_mask(dim, 6.0)))
iter = init(prob, ORK256(),  advance_to_tstop = true,  dt = dt)

env = WaveEnv(iter, C, 0.5, reward_signal)
designs = Vector{DesignInterpolator}()

@time while !is_terminated(env)
    action = Cylinder(rand(action_space(env))..., 0.0, 0.0)
    env(action)
    println("Time: $(env.iter.t)")
    push!(designs, env.C.design)
end

design_interp = interpolate(designs, 0.1)
sol_inc_interp = interpolate(dim, sol_inc, 0.1)
sol_tot_interp = interpolate(dim, env.iter.sol, 0.1)
sol_sc = sol_tot_interp - sol_inc_interp

render!(sol_tot_interp, design_interp, path = "sol_tot.mp4")
render!(sol_inc_interp, path = "sol_inc.mp4")
render!(sol_sc, design_interp, path = "sol_sc.mp4")

flux_inc = reward_signal.flux(sol_inc_interp)
flux_sc = reward_signal.flux(sol_sc)

fig = Figure(resolution = (1920, 1080), fontsize = 40)
ax = Axis(fig[1, 1], title = "Acoustic Flux", xlabel = "Time (s)", ylabel = "Flux")
lines!(ax, sol_inc_interp.t, flux_inc, label = "Incident", color = :blue, linewidth = 3)
lines!(ax, sol_sc.t, flux_sc, label = "Scattered", color = :red, linewidth = 3)
axislegend(ax)
save("flux.png", fig)
