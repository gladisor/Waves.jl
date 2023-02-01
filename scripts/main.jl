using Waves
using Waves: perturb, spacetime, signature, wave_discretizer, wave_equation, get_domain
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
import GLMakie

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

# design = Design(Configuration([0.0], [-3.0], [0.5], [0.0]))
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.05)
design = nothing

ps = [wave.speed => kwargs[:ambient_speed]]

if !isnothing(design)
    ps = vcat(ps, Waves.design_parameters(design, design.design, 0.0, kwargs[:tmax]))
end

C = WaveSpeed(wave, nothing)
@variables v(..), energy(..)

x, y, t, u = Waves.unpack(wave)
Dx = Differential(x); Dxx = Differential(x) ^ 2
Dy = Differential(y); Dyy = Differential(y) ^ 2
Dt = Differential(t)

eq = [
    v(x, y, t) ~ Dt(u(x, y, t))
    Dt(v(x, y, t)) ~ C(x, y, t) ^ 2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))

    # wave_equation(wave, C),
    # energy(x, y, t) ~ (Dxx(Dt(u(x, y, t))) + Dyy(Dt(u(x, y, t)))),
    ]

bcs = [
    wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave), 
    kwargs[:boundary](wave)...,
    v(x, y, 0.0) ~ 0.]
    
@named sys = PDESystem(
    eq, bcs, get_domain(wave, tmax = kwargs[:tmax]), spacetime(wave), 
    [
        signature(wave), 
        v(x, y, t)
        # energy(x, y, t),
    ], ps)

disc = wave_discretizer(wave, kwargs[:n])
@time iter = init(discretize(sys, disc), Tsit5(), advance_to_tstop = true, saveat = kwargs[:dt])

println("Build sim")
@time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
propagate!(sim)
sol = WaveSol(sim)
render!(sol, path = "vid.mp4")

# env = WaveEnv(sim, design, 20)

# steps = Vector{Configuration}([env.design.design])

# @time while !is_terminated(env)
#     action = Configuration(dim, M = length(env.design.design.x), r = 0.0) / 5
#     [push!(steps, s) for s ∈ perturb(env, action)]
# end

# sol = WaveSol(env)

# E = vec(sum(cat(sim.iter.sol[sim.grid[energy(x, y, t)]]..., dims = 3), dims = (1, 2)))

# tick_length = length(E)
# old_ticks = collect(1:100:tick_length)
# new_ticks = collect(range(0, Waves.tspan(sim)[end], length = length(old_ticks)))

# fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 50)
# ax = GLMakie.Axis(fig[1, 1], 
#     title = "Energy Flux",
#     xlabel = "Time", ylabel = "Wave Energy: Σ∇ ̇u",
#     xticks = (old_ticks,  string.(new_ticks)))

# GLMakie.lines!(ax, E, linewidth = 8, label = "Energy")
# # GLMakie.lines!(ax, F, linewidth = 8, label = "Flux")
# GLMakie.Legend(fig[1, 2], ax, "Wave")
# GLMakie.save("energy.png", fig)

