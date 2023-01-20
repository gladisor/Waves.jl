using ModelingToolkit
using Waves
import GLMakie

struct Configuration <: AbstractDesign
    scatterers::Vector{Cylinder}
end

function Configuration(;M::Int, name)
    scatterers = []

    for i ∈ 1:M
        cyl = Cylinder(name = "$(name)_cyl$i")
        push!(scatterers, cyl)
    end

    return Configuration(scatterers)
end

function Waves.interpolate(initial::Configuration, final::Configuration, t::Real)
    return Configuration([Waves.interpolate(initial.scatterers[i], final.scatterers[i], t) for i ∈ axes(initial.scatterers, 1)])
end

function Waves.wave_speed(wave::Wave{TwoDim}, design::ParameterizedDesign{Configuration})::Function
    C = (x, y, t) -> begin
        design = Waves.interpolate(design.initial, design.final, Waves.get_t_norm(design, t))
        is_hit = [(x, y)] .∈ design.scatterers
        cyl_speeds = hcat(design_parameters.(pd.design.scatterers)...)'[:, end]
        return is_hit' * cyl_speeds + (1 - sum(is_hit)) * wave.speed
    end

    return C
end

function Waves.design_parameters(config::Configuration)
    return vcat(design_parameters.(config.scatterers)...)
end

function GLMakie.mesh!(ax::GLMakie.Axis3, config::Configuration)
    for i ∈ axes(config.scatterers, 1)
        GLMakie.mesh!(ax, config.scatterers[i])
    end
end

function Base.:+(config::Configuration, action::Configuration)
    return Configuration([config.scatterers[i] + action.scatterers[i] for i ∈ axes(config.scatterers, 1)])
end

function adjust(config::Configuration, action::Configuration, dim::TwoDim)
    
    new_scatterers = []
    for i ∈ axes(config.scatterers)
        scatterer = adjust(config.scatterers[i], action.scatterers[i], dim)
        push!(new_scatterers, scatterer)
    end

    return Configuration(new_scatterers)
end

function Base.range(start::Configuration, stop::Configuration, length::Int)
    scatterers = [range(start.scatterers[i], stop.scatterers[i], length) for i ∈ axes(start.scatterers, 1)]
    scatterers = collect(zip(scatterers...))
    return map(x -> Configuration([x...]), scatterers)
end

gs = 5.0
dim = TwoDim(gs)
design = Cylinder(0.0, 0.0)
action = Cylinder(1.0, 1.0, 0.0, 0.0)

new_design = adjust(design, action, dim)
new_design = adjust(new_design, action, dim)
new_design = adjust(new_design, action, dim)
new_design = adjust(new_design, action, dim)

# kwargs = Dict(:tmax => 40.0, :speed => 1.0, :n => 21, :dt => 0.05)
# design = Configuration([Cylinder(-1.0, 1.0, 0.5, 0.0), Cylinder(-1.0, -1.0, 0.5, 0.0), Cylinder(-2.0, 2.0, 0.5, 0.0), Cylinder(-4.0, -4.0, 0.5, 0.0)])
# wave_tot = Wave(dim = TwoDim(gs), free = true)
# pulse = GaussianPulse(intensity = 1.0, loc = [2.5, 2.5])

# env = WaveEnv(
#     sim = WaveSim(wave = wave_tot, design = pd, ic = pulse; kwargs...), 
#     design = ParameterizedDesign(design, M = length(design.scatterers)), 
#     design_steps = 40)

# steps = Vector{typeof(env.design.design)}([env.design.design])

# ## run env with random control
# while !is_terminated(env)
#     action = Configuration([Cylinder(randn(), randn(), 0.0, 0.0) for i ∈ axes(design.scatterers, 1)])
#     [push!(steps, s) for s ∈ perturb(env, action)]
# end

# sol_tot = WaveSol(env)
# render!(sol_tot, design = steps, path = "config.mp4")

# ## render total field
# wave_inc = Wave(dim = TwoDim(gs), free = true)
# sim_inc = WaveSim(wave = wave_inc, ic = pulse; kwargs...)
# Waves.step!(sim_inc)
# sol_inc = WaveSol(sim_inc)
# sol_sc = sol_tot - sol_inc

# plot_energy!(sol_inc, sol_sc, path = "config.png")
