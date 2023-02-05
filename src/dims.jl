export OneDim, TwoDim, ThreeDim, dims

struct OneDim <: AbstractDim
    x::Vector{Float64}
end

struct TwoDim <: AbstractDim
    x::Vector{Float64}
    y::Vector{Float64}
end

struct ThreeDim <: AbstractDim
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
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

function OneDim(grid_size::Float64, Δ::Float64)
    return OneDim(-grid_size, grid_size, Δ)
end

function TwoDim(grid_size::Float64, Δ::Float64)
    return TwoDim(-grid_size, grid_size, -grid_size, grid_size, Δ)
end

function ThreeDim(grid_size::Float64, Δ::Float64)
    return ThreeDim(-grid_size, grid_size, -grid_size, grid_size, -grid_size, grid_size, Δ)
end