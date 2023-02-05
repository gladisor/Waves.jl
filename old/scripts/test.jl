using Waves
import GLMakie

function energy(sim::WaveSim)
    return sum(state(sim) .^ 2)
end

function Waves.reset!(env::WaveEnv{TwoDim, Configuration})
    reset!(env, M = length(env.design.design.x))
end

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)
design0 = Configuration([0.0], [0.0], [0.5], [0.0])
design = Design(design0)
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.05)

@time sim = WaveSim(;design = design, kwargs...)
sim_inc = WaveSim(;kwargs...)
propagate!(sim_inc)
sol_inc = WaveSol(sim_inc)

env = WaveEnv(sim, design, 20)
reset!(env)
env.design.design = Configuration([0.0], [0.0], [0.5], [0.15])

trajectory = Vector{Configuration}([env.design.design])

while !is_terminated(env)
    action = Configuration(dim, M = 1, r = 0.0) / 5
    steps = Waves.perturb(env, action)
    [push!(trajectory, s) for s ∈ steps]
end

sol_tot = WaveSol(env)
sol = sol_tot - sol_inc
render!(sol, design = trajectory, path = "env_sc.mp4")
E = map(x -> sum(x.^2), sol.data)
fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 50)
ax = GLMakie.Axis(fig[1, 1], title = "Energy", xlabel = "Time", ylabel = "Wave Energy")
GLMakie.lines!(ax, E, linewidth = 8, label = "Energy")
GLMakie.Legend(fig[1, 2], ax, "Wave")
GLMakie.save("energy.png", fig)
