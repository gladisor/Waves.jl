export Wave

struct Wave{D <: AbstractDim}
    dim::D
    u::AbstractArray{Float64}
end