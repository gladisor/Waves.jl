export Wave, displacement

struct Wave{D <: AbstractDim}
    u::AbstractArray{Float32}
end

function Wave(dim::AbstractDim, fields::Int = 1)
    u = zeros(Float32, size(dim)..., fields)
    return Wave{typeof(dim)}(u)
end

function field(wave::Wave{TwoDim}, i::Int)
    return view(wave.u, :, :, i)
end

function displacement(wave::Wave{TwoDim})
    return field(wave, 1)
end

function Base.:+(wave1::Wave{D}, wave2::Wave{D}) where D <: AbstractDim
    return Wave{D}(wave1.u .+ wave2.u)
end

function Base.:*(wave::Wave{D}, n::Number) where D <: AbstractDim
    return Wave{D}(n * wave.u)
end

function Base.:*(n::Number, wave::Wave)
    return wave * n
end