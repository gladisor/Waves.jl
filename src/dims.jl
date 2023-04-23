export OneDim, TwoDim, ThreeDim, build_grid, build_wave, dirichlet

struct OneDim <: AbstractDim
    x::AbstractVector{Float32}
end

Flux.@functor OneDim

struct TwoDim <: AbstractDim
    x::AbstractVector{Float32}
    y::AbstractVector{Float32}
end

Flux.@functor TwoDim

struct ThreeDim <: AbstractDim
    x::AbstractVector{Float32}
    y::AbstractVector{Float32}
    z::AbstractVector{Float32}
end

Flux.@functor ThreeDim

function OneDim(x_min::Float32, x_max::Float32, Δ::Float32)
    return OneDim(collect(x_min:Δ:x_max))
end

function TwoDim(x_min::Float32, x_max::Float32, y_min::Float32, y_max::Float32, Δ::Float32)
    x = collect(x_min:Δ:x_max)
    y = collect(y_min:Δ:y_max)
    return TwoDim(x, y)
end

function ThreeDim(x_min::Float32, x_max::Float32, y_min::Float32, y_max::Float32, z_min::Float32, z_max::Float32, Δ::Float32)
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

function Base.ndims(dim::AbstractDim)
    return length(size(dim))
end

function build_grid(dim::OneDim)
    return dim.x
end

"""
Returns a 3 dimensional array (x, y, 2) where x and y are the number
of discretization points on each axis. The last dimension specifies the
x or y coordinate in the space. This array is useful in operations
which would normally involve a double for loop.
"""
function build_grid(dim::TwoDim)
    x = repeat(dim.x, 1, length(dim.y))
    y = repeat(dim.y', length(dim.x))
    g = cat(x, y, dims = 3)
    return g
end

function Base.one(dim::OneDim)
    return dim.x .^ 0.0f0
end

function Base.one(dim::TwoDim)
    return (dim.x * dim.y') .^ 0.0f0
end

function build_wave(dim::AbstractDim; fields::Int)
    return zeros(Float32, size(dim)..., fields)
end

function dirichlet(dim::OneDim)
    bc = ones(Float32, size(dim)[1])
    bc[[1, end]] .= 0.0f0
    return bc
end

function dirichlet(dim::TwoDim)
    bc = one(dim)
    bc[:, 1] .= 0.0f0
    bc[1, :] .= 0.0f0
    bc[:, end] .= 0.0f0
    bc[end, :] .= 0.0f0
    return bc
end