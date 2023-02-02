using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 100, :dt => 0.1)
# sim = WaveSim(;kwargs...)
# propagate!(sim)
# sol = WaveSol(sim)
# render!(sol, path = "1d.mp4")

C = WaveSpeed(wave, nothing)
@variables ∂x(..)

x, y, t, u = Waves.unpack(wave)
Dx = Differential(x); Dxx = Differential(x) ^ 2
Dy = Differential(y); Dyy = Differential(y) ^ 2
Dt = Differential(t); Dtt = Differential(t) ^ 2

eq = [
    Dtt(u(x, y, t)) ~ C(x, y, t) ^ 2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t))),
    # ∂x(x, t) ~ Dx(u(x, t))
    ]

bcs = [
    wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave), 
    kwargs[:boundary](wave)...]
    
ps = [wave.speed => kwargs[:ambient_speed]]
@named sys = PDESystem(
    eq, bcs, Waves.get_domain(wave, tmax = kwargs[:tmax]), Waves.spacetime(wave), 
    [
        Waves.signature(wave), 
        # ∂x(x, t)
    ], ps)

disc = Waves.wave_discretizer(wave, kwargs[:n])
@time iter = init(discretize(sys, disc), Tsit5(), advance_to_tstop = true, saveat = kwargs[:dt])
@time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
propagate!(sim)
sol = WaveSol(sim)
render!(sol, path = "2d.mp4")

# x_der = sim.iter.sol[sim.grid[∂x(x, t)]]
# points = hcat([x_der[i][[1, end]] for i ∈ axes(x_der, 1)]...)
# n = [-1.0, 1.0]
# flux = vec(n' * points)
# savefig(plot(flux), "flux.png")
# # E = map(x -> sum(x.^2), sol.data)
# # fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 50)
# # ax = GLMakie.Axis(fig[1, 1], title = "Energy", xlabel = "Time", ylabel = "Wave Energy")
# # GLMakie.lines!(ax, E, linewidth = 8, label = "Energy")
# # GLMakie.Legend(fig[1, 2], ax, "Wave")
# # GLMakie.save("energy.png", fig)
