using Waves
using Waves: get_domain, spacetime, signature, wave_discretizer, wave_equation, unpack, WaveBoundary
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
import GLMakie

gs = 5.0
c = 1.0
pml_width = 2.0
pml_start = gs - pml_width

function σₓ(x)
    x_pml = abs(x) - pml_start
    return IfElse.ifelse(x_pml > 0.0, x_pml / pml_width, 0.0)
end

dim = OneDim(size = gs)
wave = Wave(dim = dim)

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(), 
    :boundary => MinimalBoundary(), 
    :ambient_speed => c, 
    :tmax => 20.0, :n => 21, :dt => 0.1)

ps = [wave.speed => kwargs[:ambient_speed]]

eq = wave_equation(wave)
@variables v(..), w(..)
x, t, u = unpack(wave)
Dx = Differential(x); Dxx = Differential(x)^2
Dt = Differential(t); Dtt = Differential(t)^2

x_min, x_max = getbounds(x)

eqs = [
    # Dt(u(x, t)) ~ Dx(v(x, t)),
    # Dt(v(x, t)) ~ Dx(u(x, t))

    v(x, t) ~ Dx(u(x, t)),
    # w(x, t) ~ Dx(u(x, t)),
    Dtt(u(x, t)) ~ Dx(v(x, t)),
    ]

bcs = [

    ]

@named sys = PDESystem(
    eqs, 
    bcs, 
    get_domain(wave, tmax = kwargs[:tmax]), 
    spacetime(wave), 
    [
        u(x, t), 
        v(x, t), 
        # w(x, t)
        ],
    ps)

disc = wave_discretizer(wave, kwargs[:n])
iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])

# sim = WaveSim(;kwargs...)
propagate!(sim)
sol = WaveSol(sim)
render!(sol, path = "vid.mp4")

# fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
# ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X", ylabel = "Y")

# # ∂t = sim.iter.sol[sim.grid[v(x, t)]]
# # ∂x = sim.iter.sol[sim.grid[w(x, t)]]
# x_axis = collect(sim.grid[x])

# GLMakie.xlims!(ax, -gs, gs)
# GLMakie.ylims!(ax, -1.0, 1.0)

# GLMakie.record(fig, "der.mp4", axes(sol.data, 1)) do i
#     GLMakie.empty!(ax.scene)
#     GLMakie.lines!(ax, x_axis, sol.data[i], linestyle = nothing, linewidth = 5, color = :blue)
#     GLMakie.lines!(ax, x_axis, ∂t[i], linestyle = nothing, linewidth = 5, color = :orange)
#     GLMakie.lines!(ax, x_axis, ∂x[i], linestyle = nothing, linewidth = 5, color = :green)
# end
