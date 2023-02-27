export pulse, Pulse

struct Pulse <: InitialCondition
    pos::AbstractVector{Float32}
    intensity::Float32
end

function (pulse::Pulse)(g::AbstractVector{Float32})
    return exp.(- pulse.intensity * (g .- pulse.pos) .^ 2)
end

function (p::Pulse)(g::AbstractArray{Float32, 3})
    pos = reshape(p.pos, 1, 1, size(p.pos)...)
    u = exp.(- p.intensity * dropdims(sum((g .- pos) .^ 2, dims = 3), dims = 3))
    # z = dropdims(sum(g, dims = 3), dims = 3) * 0.0f0
    # u = cat(u, z, z, dims = 3)
    return u
end

function Flux.gpu(p::Pulse)
    return Pulse(gpu(p.pos), p.intensity)
end

function Flux.cpu(p::Pulse)
    return Pulse(cpu(p.pos), p.intensity)
end