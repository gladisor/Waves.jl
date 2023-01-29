export WaveSpeed

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

    return IfElse.ifelse(count > 0.0, (inside' * c′) / count, wave.speed)
    # return inside' * c′ + (1 - sum(inside)) * wave.speed
end