using Waves
using Waves: get_domain, spacetime, signature, wave_discretizer, wave_equation, unpack
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
c = 1.0

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

eqs = [
    # w(x, t) ~ Dx(u(x, t)),
    v(x, t) ~ Dt(u(x, t)),
    # Dt(v(x, t)) ~ Dx(w(x, t)) # first order
    Dt(v(x, t)) ~ Dxx(u(x, t)) # first order in time second order in space

    ]

bcs = [
    u(x, 0.) ~ kwargs[:ic](wave), 
    u(x_min, t) ~ 0., 
    u(x_max, t) ~ 0., 
    Dx(u(x_min, t)) ~ 0.,
    Dx(u(x_max, t)) ~ 0.,
    # Dt(u(x, 0.)) ~ 0.,
    v(x_min, t) ~ 0.,
    v(x_max, t) ~ 0.,
    Dx(v(x_min, t)) ~ 0.,
    Dx(v(x_max, t)) ~ 0.,
    v(x, 0.) ~ 0.,
    
    # w(x_min, t) ~ 0.,
    # w(x_max, t) ~ 0.,
    ]

@named sys = PDESystem(
    eqs, 
    bcs, 
    get_domain(wave, tmax = kwargs[:tmax]), 
    spacetime(wave), 
    [u(x, t), v(x, t)],
    ps)

disc = wave_discretizer(wave, kwargs[:n])
iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])

propagate!(sim)
render!(WaveSol(sim), path = "first_order.mp4")

fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X", ylabel = "Y")

der = sim.iter.sol[sim.grid[v(x, t)]]
x_axis = collect(sim.grid[x])

GLMakie.xlims!(ax, -gs, gs)
GLMakie.ylims!(ax, -1.0, 1.0)

GLMakie.record(fig, "der.mp4", axes(der, 1)) do i
    GLMakie.empty!(ax.scene)
    GLMakie.lines!(ax, x_axis, der[i], linestyle = nothing, linewidth = 5, color = :blue)
end
