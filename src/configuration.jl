export Configuration

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

function Waves.interpolate(initial::Configuration, final::Configuration, t::Num)
    return Configuration([
        interpolate(initial.scatterers[i], final.scatterers[i], t) for i ∈ axes(initial.scatterers, 1)
    ])
end

function Base.:∈(xy::Tuple, config::Configuration)
    return [xy ∈ config.scatterers[i] for i ∈ axes(config.scatterers, 1)]
end

function speeds(config::Configuration)
    return [config.scatterers[i].c for i ∈ axes(config.scatterers, 1)]
end

function Waves.ParameterizedDesign(design::Configuration)
    @named initial = Configuration(M = length(design.scatterers))
    @named final = Configuration(M = length(design.scatterers))

    return Waves.ParameterizedDesign(design, initial, final)
end

function Waves.wave_speed(wave::Wave{TwoDim}, design::Waves.ParameterizedDesign{Configuration})::Function
    C = (x, y, t) -> begin
        design = Waves.interpolate(design.initial, design.final, Waves.get_t_norm(design, t))
        in_design = (x, y) ∈ design
        return in_design' * speeds(design) + (1 - sum(in_design)) * wave.speed
    end

    return C
end

function Waves.design_parameters(config::Configuration)
    return vcat(design_parameters.(config.scatterers)...)
end