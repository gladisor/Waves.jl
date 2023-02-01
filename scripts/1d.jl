using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
dim = OneDim(size = gs)
wave = Wave(dim = dim)
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 100, :dt => 0.05)

C = WaveSpeed(wave, nothing)
@variables energy(..)

x, t, u = Waves.unpack(wave)
Dx = Differential(x); Dxx = Differential(x) ^ 2
Dt = Differential(t); Dtt = Differential(t) ^ 2

eq = [
    Dtt(u(x, t)) ~ C(x, t) ^ 2 * Dxx(u(x, t)),
    energy(x, t) ~ Dtt(u(x, t))
    ]

bcs = [wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave), kwargs[:boundary](wave)...]
ps = [wave.speed => kwargs[:ambient_speed]]
@named sys = PDESystem(
    eq, bcs, get_domain(wave, tmax = kwargs[:tmax]), spacetime(wave), 
    [
        signature(wave), 
    ], ps)

disc = wave_discretizer(wave, kwargs[:n])
@time iter = init(discretize(sys, disc), Tsit5(), advance_to_tstop = true, saveat = kwargs[:dt])
@time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
propagate!(sim)
sol = WaveSol(sim)
render!(sol, path = "vid.mp4")

E = map(x -> sum(x.^2), sol.data)
fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 50)
ax = GLMakie.Axis(fig[1, 1], title = "Energy", xlabel = "Time", ylabel = "Wave Energy")
GLMakie.lines!(ax, E, linewidth = 8, label = "Energy")
GLMakie.Legend(fig[1, 2], ax, "Wave")
GLMakie.save("energy.png", fig)
