export build_gradient, ∂x, ∂y
export build_normal

const FORWARD_DIFF_COEF = [-3.0f0, 4.0f0, -1.0f0]
const BACKWARD_DIFF_COEF = [1.0f0, -4.0f0, 3.0f0]
const CENTRAL_DIFF_COEF = [-1.0f0, 1.0f0]

"""
Function for constructing a sparse gradient Matrix for a one dimensional scalar field.
"""
function gradient(x::Vector{Float32})
    grad = zeros(Float32, size(x, 1), size(x, 1))
    Δ = (x[end] - x[1]) / (length(x) - 1)

    grad[[1, 2, 3], 1] .= FORWARD_DIFF_COEF ## left boundary edge case
    grad[[end-2, end-1, end], end] .= BACKWARD_DIFF_COEF ## right boundary edge case

    for i ∈ 2:(size(grad, 2) - 1)
        grad[[i - 1, i + 1], i] .= CENTRAL_DIFF_COEF
    end

    return sparse((grad / (2 * Δ))')
end

function build_gradient(dim::AbstractDim)
    return gradient(dim.x)
end

"""
Constructs the sparse matrix which performs a second spatial derivative.
"""
function laplacian(x::Vector{Float32})
    laplace = zeros(Float32, length(x), length(x))
    dx = (x[end] - x[1]) / (length(x) - 1)

    laplace[1, [1, 2, 3, 4]] .= [2.0f0, -5.0f0, 4.0f0, -1.0f0] / dx^3
    laplace[end, [end-3, end-2, end-1, end]] .= [-1.0f0, 4.0f0, -5.0f0, 2.0f0] / dx^3

    for i ∈ 2:(size(laplace, 2) - 1)
        laplace[i, [i-1, i, i+1]] .= [1.0f0, -2.0f0, 1.0f0] / dx^2
    end

    return sparse(laplace)
end

∂x(∇::AbstractMatrix, u::Union{AbstractVector, AbstractMatrix}) = ∇ * u
∂y(∇::AbstractMatrix, u::AbstractMatrix) = (∇ * u')'
divergence(∇::AbstractMatrix{Float32}, u::AbstractMatrix{Float32}) = ∂x(∇, u) .+ ∂y(∇, u)

function build_normal(x::AbstractVector{Float32}, μ::AbstractVector{Float32}, σ::AbstractVector, a::AbstractVector)
    μ = permutedims(μ)
    σ = permutedims(σ)
    a = permutedims(a)
    f = (1.0f0 ./ (σ * sqrt(2.0f0 * π))) .* a .* exp.(- ((x .- μ) .^ 2) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 2), dims = 2)
end

function build_normal(x::AbstractArray{Float32, 3}, μ::AbstractMatrix, σ::AbstractVector, a::AbstractVector)
    μ = permutedims(μ[:, :, :, :], (3, 4, 2, 1))
    σ = permutedims(σ[:, :, :], (2, 3, 1))
    a = permutedims(a[:, :, :], (2, 3, 1))
    f = (1.0f0 ./ (2.0f0 * π * σ .^ 2)) .* a .* exp.(-dropdims(sum((x .- μ) .^ 2, dims = 3), dims = 3) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 3), dims = 3)
end