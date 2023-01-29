export Configuration

struct Configuration <: AbstractDesign
    x
    y
    r
    c
end

function Configuration(dim::TwoDim; M::Int, r::Real = 1.0, c::Real = 0.0)
    x_min, x_max = getbounds(dim.x)
    y_min, y_max = getbounds(dim.y)

    r = repeat([r], M)
    c = repeat([c], M)

    x = rand.(Uniform.(x_min .+ r, x_max .- r))
    y = rand.(Uniform.(y_min .+ r, y_max .- r))
    return Configuration(x, y, r, c)
end

function Configuration(;name::Symbol, M::Int)
    x = Symbol(name, "_x")
    y = Symbol(name, "_y")
    r = Symbol(name, "_r")
    c = Symbol(name, "_c")

    ps = @parameters $(x)[1:M], $(y)[1:M], $(r)[1:M], $(c)[1:M]
    return Configuration(collect.(ps)...)
end

function Waves.design_parameters(config::Configuration)
    return vcat(config.x, config.y, config.r, config.c)
end

function Base.range(start::Configuration, stop::Configuration, length::Int)

    x′ = collect(range(start.x, stop.x, length))
    y′ = collect(range(start.y, stop.y, length))
    r′ = collect(range(start.r, stop.r, length))
    c′ = collect(range(start.c, stop.c, length))

    return Configuration.(x′, y′, r′, c′)
end

function GLMakie.mesh!(ax::GLMakie.Axis3, config::Configuration)
    cyls = Cylinder.(config.x, config.y, config.r, config.c)
    for cyl ∈ cyls
        GLMakie.mesh!(ax, cyl)
    end
    
    return nothing
end

function (C::WaveSpeed{TwoDim, Design{Configuration}})(x, y, t)
    design = C.design

    t′ = (t - design.ti) / (design.tf - design.ti)
    x′ = Waves.interpolate.(design.initial.x, design.final.x, t′)
    y′ = Waves.interpolate.(design.initial.y, design.final.y, t′)
    r′ = Waves.interpolate.(design.initial.r, design.final.r, t′)
    c′ = Waves.interpolate.(design.initial.c, design.final.c, t′)

    inside = @. (x - x′) ^ 2 + (y - y′) ^ 2 < r′^2
    any_inside = min(sum(inside), 1.0)

    # return IfElse.ifelse(count > 0.0, (inside' * c′) / count, wave.speed)
    return inside' * c′ + (1 - any_inside) * C.wave.speed
end

function Design(config::Configuration)
    return Design(config, M = length(config.x))
end