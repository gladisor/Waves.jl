export SpeedField

mutable struct SpeedField{Dim <: AbstractDim, Des <: Union{DesignInterpolator, Nothing}}
    dim::Dim
    g::AbstractArray
    ambient_speed::AbstractArray
    design::Des
end

function SpeedField(dim::AbstractDim, ambient_speed::Float64, design::Union{DesignInterpolator, Nothing} = nothing)
    return SpeedField(dim, grid(dim), ones(size(dim)) * ambient_speed, design)
end

function (C::SpeedField)(t::Float64)
    if isnothing(C.design)
        return C.ambient_speed
    else
        return speed(C.design(t), C.g, C.ambient_speed)
    end
end