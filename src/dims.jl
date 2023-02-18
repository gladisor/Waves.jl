export OneDim, TwoDim, ThreeDim, grid

struct OneDim <: AbstractDim
    x::Vector{Float32}
end

struct TwoDim <: AbstractDim
    x::Vector{Float32}
    y::Vector{Float32}
end

struct ThreeDim <: AbstractDim
    x::Vector{Float32}
    y::Vector{Float32}
    z::Vector{Float32}
end

function OneDim(x_min, x_max, Δ)
    return OneDim(collect(x_min:Δ:x_max))
end

function TwoDim(x_min, x_max, y_min, y_max, Δ)
    x = collect(x_min:Δ:x_max)
    y = collect(y_min:Δ:y_max)
    return TwoDim(x, y)
end

function ThreeDim(x_min, x_max, y_min, y_max, z_min, z_max, Δ)
    x = collect(x_min:Δ:x_max)
    y = collect(y_min:Δ:y_max)
    z = collect(z_min:Δ:z_max)
    return ThreeDim(x, y, z)
end

function OneDim(grid_size::Float32, Δ::Float32)
    return OneDim(-grid_size, grid_size, Δ)
end

function OneDim(grid_size::Float32, n::Int)
    return OneDim(collect(range(-grid_size, grid_size, n)))
end

function TwoDim(grid_size::Float32, Δ::Float32)
    return TwoDim(-grid_size, grid_size, -grid_size, grid_size, Δ)
end

function TwoDim(grid_size::Float32, n::Int)
    return TwoDim(
        collect(range(-grid_size, grid_size, n)),
        collect(range(-grid_size, grid_size, n)))
end

function ThreeDim(grid_size::Float32, Δ::Float32)
    return ThreeDim(-grid_size, grid_size, -grid_size, grid_size, -grid_size, grid_size, Δ)
end

function Base.size(dim::OneDim)
    return size(dim.x)
end

function Base.size(dim::TwoDim)
    return (length(dim.x), length(dim.y))
end

function Base.size(dim::ThreeDim)
    return (length(dim.x), length(dim.y), length(dim.z))
end

function grid(dim::TwoDim)
    x = repeat(dim.x, 1, length(dim.y))
    y = repeat(dim.y', length(dim.x))
    g = cat(x, y, dims = 3)
    return g
end

function Flux.gpu(dim::TwoDim)
    return TwoDim(gpu(dim.x), gpu(dim.y))
end

function Flux.cpu(dim::TwoDim)
    return WaveSol(cpu(dim.x), cpu(dim.y))
end