export WaveSpeed

struct WaveSpeed{D <: AbstractDim}
    dim::D
    C0::AbstractArray
    design::Union{DesignInterpolator, Nothing}
end

function WaveSpeed(dim::AbstractDim, C0::Float64, design::Union{DesignInterpolator, Nothing} = nothing)
    return WaveSpeed(dim, ones(size(dim)) * C0, design)
end

function (C::WaveSpeed)(t::Float64)
    if isnothing(C.design)
        return C.C0
    else
        return speed(C.dim, C.design(t), C.C0)
    end
end