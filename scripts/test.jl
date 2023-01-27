using Waves
using Waves: WaveBoundary, perturb, AbstractDesign
using ModelingToolkit: @named, getbounds
import GLMakie

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), 
    :boundary => MinimalBoundary(), 
    :ambient_speed => 1.0, 
    :tmax => 20.0, :n => 21, :dt => 0.05)

function Base.:/(cyl::Cylinder, n::Real)
    return Cylinder(cyl.x/n, cyl.y/n, cyl.r, cyl.c)
end

struct Configuration <: AbstractDesign
    scatterers::Vector
end

function GLMakie.mesh!(ax::GLMakie.Axis3, config::Configuration)
    for i ∈ axes(config.scatterers, 1)
        GLMakie.mesh!(ax, config.scatterers[i])
    end

    return nothing
end

function Base.:+(config::Configuration, action::Configuration)
    return Configuration([
        config.scatterers[i] + action.scatterers[i] for i ∈ axes(config.scatterers, 1)
    ])
end

function Base.:/(config::Configuration, n::Real)
    return Configuration([
        config.scatterers[i] / n for i ∈ axes(config.scatterers, 1)
    ])
end

function Waves.perturb(config::Configuration, action::Configuration, dim::TwoDim)
    return Configuration([
        perturb(config.scatterers[i], action.scatterers[i], dim) for i ∈ axes(config.scatterers, 1)
    ])
end

function Configuration(dim::TwoDim; M::Int, r::Real, c::Real)
    return Configuration([
        Cylinder(dim, r = r, c = c) for i ∈ 1:M
    ])
end

function Configuration(;M::Int, name)
    scatterers = []

    for i ∈ 1:M
        cyl = Cylinder(name = "$(name)_cyl$i")
        push!(scatterers, cyl)
    end

    return Configuration(scatterers)
end

function Base.range(start::Configuration, stop::Configuration, length::Int)
    steps = []

    for i ∈ axes(start.scatterers, 1)
        push!(
            steps,
            range(start.scatterers[i], stop.scatterers[i], length)
            )
    end

    steps = hcat(steps...)
    return [Configuration(steps[i, :]) for i ∈ axes(steps, 1)]
end

fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
ax = GLMakie.Axis3(fig[1,1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X", ylabel = "Y", zlabel = "Z")

GLMakie.xlims!(ax, getbounds(wave.dim.x)...)
GLMakie.ylims!(ax, getbounds(wave.dim.y)...)
GLMakie.zlims!(ax, -1.0, 4.0)

M = 10
design = Configuration(dim, M = M, r = 0.5, c = 0.0)
action = Configuration(dim, M = M, r = 0.0, c = 0.0) / 5
new_design = perturb(design, action, dim)
steps = range(design, new_design, 20)

GLMakie.record(fig, "config.mp4", axes(steps, 1)) do i
    GLMakie.empty!(ax.scene)
    GLMakie.mesh!(ax, steps[i])
end

# GLMakie.mesh!(ax, new_design)
# GLMakie.save("config.png", fig)

# @named initial = Configuration(M = M)


# design = ParameterizedDesign(Cylinder(0.0, 0.0, 0.5, 0.0))
# sim = WaveSim(design = design; kwargs...)

# env = WaveEnv(sim = sim, design = design, design_steps = 20)
# design_trajectory = Vector{typeof(env.design.design)}([env.design.design])

# while !is_terminated(env)
#     action = Cylinder(env.sim.wave.dim, r = 0.0, c = 0.0)
#     action = Cylinder(action.x/2, action.y/2, action.r, action.c)
#     steps = perturb(env, action)
#     [push!(design_trajectory, s) for s in steps]
# end

# render!(WaveSol(env), design = design_trajectory, path = "test.mp4")

