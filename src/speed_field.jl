export SpeedField

mutable struct SpeedField
    dim::AbstractDim
    g::AbstractArray
    ambient_speed::AbstractArray
    design::Union{DesignInterpolator, Nothing}
end

function SpeedField(dim::AbstractDim, ambient_speed::Float32, design::Union{DesignInterpolator, Nothing} = nothing)
    return SpeedField(dim, grid(dim), ones(size(dim)) * ambient_speed, design)
end

function (C::SpeedField)(t::Float32)
    if isnothing(C.design)
        return C.ambient_speed
    else
        return speed(C.design(t), C.g, C.ambient_speed)
    end
end

function Flux.gpu(C::SpeedField)
    return SpeedField(
        gpu(C.dim),
        gpu(C.g),
        gpu(C.ambient_speed),
        gpu(C.design)
    )
end