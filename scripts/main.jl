using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using Distributions: Uniform
import GLMakie

mutable struct Design{D <: AbstractDesign}
    design::D
    initial::D
    final::D
    ti::Num
    tf::Num
end

function Design(design::AbstractDesign; kwargs...)
    @named initial = typeof(design)(;kwargs...)
    @named final = typeof(design)(;kwargs...)
    @parameters ti, tf
    return Design(design, initial, final, ti, tf)
end

function Waves.design_parameters(design::Design)
    return Waves.design_parameters(design.design)
end

function Waves.design_parameters(design::Design, new_design::AbstractDesign, ti, tf)
    return [
        (design_parameters(design.initial) .=> design_parameters(design))...,
        (design_parameters(design.final) .=> design_parameters(new_design))...,
        design.ti => ti,
        design.tf => tf
    ]
end

function Waves.wave_speed(wave::Wave{TwoDim}, design::Design{Cylinder})::Function

    C = (x, y, t) -> begin
        t′ = (t - design.ti) / (design.tf - design.ti)
        x′, y′, r′, c′ = Waves.interpolate.(design_parameters(design.initial), design_parameters(design.final), t′)
        inside = (x - x′) ^ 2 + (y - y′) ^ 2 < r′ ^ 2
        inside * c′ + (1 - inside) * wave.speed
    end
    
    return C
end

function Waves.wave_equation(wave::Wave, design::Design)::Equation
    return Waves.wave_equation(wave, Waves.wave_speed(wave, design))
end

include("configuration.jl")

mutable struct WaveSpeed{Dm <: AbstractDim, Dz <: Union{Design{<: AbstractDesign}, Nothing}}
    wave::Wave{Dm}
    design::Dz
end

function WaveSpeed(wave::Wave)
    return WaveSpeed(wave, nothing)
end

function (C::WaveSpeed{TwoDim, Nothing})(x, y, t)
    return C.wave.speed
end

function (C::WaveSpeed{TwoDim, Design{Cylinder}})(x, y, t)
    design = C.design

    t′ = (t - design.ti) / (design.tf - design.ti)
    x′, y′, r′, c′ = Waves.interpolate.(design_parameters(design.initial), design_parameters(design.final), t′)
    inside = (x - x′) ^ 2 + (y - y′) ^ 2 < r′^2
    return inside * c′ + (1 - inside) * wave.speed
end

function (C::WaveSpeed{TwoDim, Design{Configuration}})(x, y, t)
    design = C.design

    t′ = (t - design.ti) / (design.tf - design.ti)
    x′ = Waves.interpolate.(design.initial.x, design.final.x, t′)
    y′ = Waves.interpolate.(design.initial.y, design.final.y, t′)
    r′ = Waves.interpolate.(design.initial.r, design.final.r, t′)
    c′ = Waves.interpolate.(design.initial.c, design.final.c, t′)

    inside = @. (x - x′) ^ 2 + (y - y′) ^ 2 < r′^2
    count = sum(inside)

    # return IfElse.ifelse(count > 0.0, (inside' * c′) / count, wave.speed)
    return inside' * c′ + (1 - sum(inside)) * wave.speed
end

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

M = 4

design = Design(Configuration(
    [-2.0, -2.0, -2.0, -4.0], 
    [-3.0, 0.0, 3.0, 3.0], 
    [0.5, 0.5, 0.5, 0.5], 
    [0.1, 0.1, 0.1, 0.1]), M = M)

new_design = Configuration(
    [2.0, 2.0, 2.0, 4.0], 
    [-3.0, 0.0, 3.0, 3.0], 
    [0.5, 0.5, 0.5, 0.5], 
    [0.1, 0.1, 0.1, 0.1])

C = WaveSpeed(wave, design)

x, y, t, u = Waves.unpack(wave)
Dx = Differential(x); Dxx = Differential(x)^2; Dy = Differential(y); Dyy = Differential(y)^2
Dt = Differential(t); Dtt = Differential(t)^2

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), :boundary => MinimalBoundary(), 
    :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.05)

ps = [wave.speed => kwargs[:ambient_speed]]

C = WaveSpeed(wave, design)
eq = Dtt(u(x, y, t)) ~ C(x, y, t) ^ 2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))
ps = vcat(ps, Waves.design_parameters(design, new_design, 0.0, kwargs[:tmax]))

bcs = [wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave), kwargs[:boundary](wave)...]

println("Build sys"); @time @named sys = PDESystem(eq, bcs, Waves.get_domain(wave, tmax = kwargs[:tmax]), Waves.spacetime(wave), [Waves.signature(wave)], ps)
disc = Waves.wave_discretizer(wave, kwargs[:n])
println("Build iter"); @time iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
println("Build sim"); @time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
println("propagate!"); @time propagate!(sim)

sol = WaveSol(sim)
steps = range(design.design, new_design, length(sol))
println("render!"); @time render!(sol, design = steps, path = "2d.mp4")