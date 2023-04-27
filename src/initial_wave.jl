export Silence, build_pulse, Pulse

function reset!(::AbstractInitialWave)
    return nothing
end

struct Silence{D <: AbstractDim} <: AbstractInitialWave
    grid::AbstractArray{Float32}
end

Flux.@functor Silence

function Silence(dim::D) where D <: AbstractDim
    return Silence{D}(build_grid(dim))
end

function (silence::Silence)(wave::AbstractArray{Float32})
    return wave * 0.0f0
end

function build_pulse(grid::AbstractVector{Float32}, x::Float32, intensity::Float32)
    return exp.(- intensity * (grid .- x) .^ 2)
end

function build_pulse(grid::AbstractArray{Float32, 3}, x::Float32, y::Float32, intensity::Float32)
    pos = [x ;;; y]
    return exp.(-intensity * dropdims(sum((grid .- pos) .^ 2, dims = 3), dims = 3))
end

struct Pulse{D <: AbstractDim} <: AbstractInitialWave
    grid::AbstractArray{Float32}
    pos::AbstractVector{Float32}
    intensity::Float32
end

Flux.@functor Pulse ()

function Pulse(dim::OneDim; x::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    return Pulse{OneDim}(build_grid(dim), [x], intensity)
end

function Pulse(dim::TwoDim; x::Float32 = 0.0f0, y::Float32 = 0.0f0, intensity::Float32 = 0.0f0)
    return Pulse{TwoDim}(build_grid(dim), [x, y], intensity)
end

function (pulse::Pulse{OneDim})(wave::AbstractMatrix{Float32})
    u = build_pulse(pulse.grid, pulse.pos[1], pulse.intensity)
    z = pulse.grid * 0.0f0
    z = repeat(z, 1, size(wave, 2) - 1)
    return cat(u, z, dims = 2)
end

function (pulse::Pulse{TwoDim})(wave::AbstractArray{Float32, 3})
    u = build_pulse(pulse.grid, pulse.pos[1], pulse.pos[2], pulse.intensity)
    z = dropdims(sum(pulse.grid, dims = 3), dims = 3) * 0.0f0
    z = repeat(z, 1, 1, size(wave, 3) - 1)
    return cat(u, z, dims = 3)
end