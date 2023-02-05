export WaveSpeed

mutable struct WaveSpeed{Dm <: AbstractDim, Dz <: Union{Design{<: AbstractDesign}, Nothing}}
    wave::Wave{Dm}
    design::Dz
end

function WaveSpeed(wave::Wave)
    return WaveSpeed(wave, nothing)
end

function (C::WaveSpeed{OneDim, Nothing})(x, t)
    return C.wave.speed
end

function (C::WaveSpeed{TwoDim, Nothing})(x, y, t)
    return C.wave.speed
end