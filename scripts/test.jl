using Waves
using Waves: WaveBoundary, perturb, AbstractDesign

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), 
    :boundary => MinimalBoundary(), 
    :ambient_speed => c, 
    :tmax => 20.0, :n => 21, :dt => 0.05)

struct Configuration <: AbstractDesign
    scatterers::Vector{<: AbstractDesign}
end

function GLMakie.mesh!(ax::GLMakie.Axis3, config::Configuration)
    for i ∈ axes(config.scatterers, 1)
        GLMakie.mesh!(ax, config.scatterers[i])
    end

    return nothing
end

fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
ax = GLMakie.Axis3(fig[1,1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X", ylabel = "Y", zlabel = "Z")

GLMakie.xlims!(ax, getbounds(sol.wave.dim.x)...)
GLMakie.ylims!(ax, getbounds(sol.wave.dim.y)...)
GLMakie.zlims!(ax, -1.0, 4.0)

design = Configuration([Cylinder(dim, r = 0.5), Cylinder(dim, r = 0.5)])

GLMakie.mesh!(ax, design)
GLMakie.save("config.png", fig)

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

