export pulse, Pulse

struct Pulse{D <: AbstractDim} <: InitialCondition
    mesh_grid::AbstractArray{Float32}
    pos::AbstractArray{Float32}
    intensity::Float32
end

function Pulse(dim::OneDim, x::Float32, intensity::Float32)
    return Pulse{OneDim}(grid(dim), [x], intensity)
end

function Pulse(dim::TwoDim, x::Float32, y::Float32, intensity::Float32)
    return Pulse{TwoDim}(grid(dim), [x, y], intensity)
end

function (pulse::Pulse{OneDim})()
    return exp.(- pulse.intensity * (pulse.mesh_grid .- pulse.pos) .^ 2)
end

function (pluse::Pulse{OneDim})(wave::Wave{OneDim})
    u = pulse()
    z = pulse.mesh_grid * 0.0f0
    z = repeat(z, 1, size(wave.u, 2) - 1)
    return Wave{OneDim}(cat(u, z, dims = 2))
end

function (pulse::Pulse{TwoDim})()
    pos = reshape(pulse.pos, 1, 1, size(pulse.pos)...)
    u = exp.(- pulse.intensity * dropdims(sum((pulse.mesh_grid .- pos) .^ 2, dims = 3), dims = 3))
    return u
end

function (pulse::Pulse{TwoDim})(wave::Wave{TwoDim})
    u = pulse()
    z = dropdims(sum(pulse.mesh_grid, dims = 3), dims = 3) * 0.0f0
    z = repeat(z, 1, 1, size(wave.u, 3) - 1)
    return Wave{TwoDim}(cat(u, z, dims = 3))
end

function Flux.gpu(pulse::Pulse{D}) where D <: AbstractDim
    return Pulse{D}(gpu(pulse.mesh_grid), gpu(pulse.pos), pulse.intensity)
end

function Flux.cpu(pulse::Pulse{D}) where D <: AbstractDim
    return Pulse{D}(cpu(pulse.mesh_grid), cpu(pulse.pos), pulse.intensity)
end