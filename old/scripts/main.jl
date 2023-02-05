using Waves
using Waves: perturb, spacetime, signature, wave_discretizer, wave_equation, get_domain
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
import GLMakie

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

design = Design(Configuration([0.0], [-3.0], [0.5], [0.0]))
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.1)
# design = nothing

ps = [wave.speed => kwargs[:ambient_speed]]

if !isnothing(design)
    ps = vcat(ps, Waves.design_parameters(design, design.design, 0.0, kwargs[:tmax]))
end

C = WaveSpeed(wave, design)
@variables v(..)
@variables ∇²(..)

x, y, t, u = Waves.unpack(wave)
Dx = Differential(x); Dxx = Differential(x) ^ 2
Dy = Differential(y); Dyy = Differential(y) ^ 2
Dt = Differential(t)

eq = [
    v(x, y, t) ~ Dt(u(x, y, t)),
    Dt(v(x, y, t)) ~ C(x, y, t) ^ 2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t))),
    ∇²(x, y, t) ~ Dxx(u(x, y, t)) + Dyy(u(x, y, t))
    ]

bcs = [
    wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave), 
    kwargs[:boundary](wave)...,
    v(x, y, 0.0) ~ 0.
    ]

println("Building PDESystem")
@named sys = PDESystem(
    eq, bcs, get_domain(wave, tmax = kwargs[:tmax]), spacetime(wave), 
    [
        signature(wave), 
        v(x, y, t),
        ∇²(x, y, t)
    ], ps)

disc = wave_discretizer(wave, kwargs[:n])
println("Discretizing")
@time prob = discretize(sys, disc)
println("Building Iterator")

@time iter = init(
    prob, 
    Tsit5(), 
    advance_to_tstop = true, 
    saveat = kwargs[:dt]
    )

println("Build sim")
@time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
propagate!(sim)
sol = WaveSol(sim)



function circular_region(center::Tuple, radius::Real, x_axis::Vector, y_axis::Vector)
    region = zeros(length(x_axis), length(y_axis))

    for i ∈ axes(region, 1)
        for j ∈ axes(region, 2)
            if (center[1] - x_axis[i]) ^ 2 + (center[2] - y_axis[j]) ^ 2 <= radius ^ 2
                region[i, j] = 1.0
            end
        end
    end

    return region
end

region = circular_region((0.0, 0.0), 4.0, dims(sim)...)
laplacian = Waves.extract(sim, sim.grid[∇²(x, y, t)])

flux = [sum(region * laplacian[i]) for i ∈ axes(laplacian ,1)]

render!(sol, path = "vid.mp4")

tick_length = length(flux)
old_ticks = collect(1:100:tick_length)
new_ticks = collect(range(0, Waves.tspan(sim)[end], length = length(old_ticks)))

fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 50)
ax = GLMakie.Axis(fig[1, 1], 
    title = "Flux across a boundary over time",
    xlabel = "Time", ylabel = "Flux",
    xticks = (old_ticks,  string.(new_ticks)))

GLMakie.lines!(ax, flux, linewidth = 8, label = "Flux")
GLMakie.save("flux2d.png", fig)

