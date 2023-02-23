export pulse, Pulse

function pulse(dim::OneDim, x::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    u = exp.(- intensity * (dim.x .- x) .^ 2)
    return cat(u, zeros(Float32, size(u)), dims = 2)
end

function pulse(dim::TwoDim, x::Float32 = 0.0f0, y::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    pos = [x ;;; y]
    g = grid(dim)
    u = exp.(-intensity * dropdims(sum((g .- pos) .^ 2, dims = 3), dims = 3))
    return cat(u, zeros(Float32, size(u)..., 2), dims = 3)
end

struct Pulse <: InitialCondition
    pos::Vector{Float32}
    intensity::Float32
end

function (p::Pulse)(dim::AbstractDim)
    return pulse(dim, p.pos..., p.intensity)
end