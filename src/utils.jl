export build_normal, flatten_repeated_last_dim
export LinearInterpolation

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

function flatten_repeated_last_dim(x::AbstractArray{Float32})

    last_dim = size(x, ndims(x))
    first_component = selectdim(x, ndims(x), 1)
    second_component = selectdim(selectdim(x, ndims(x) - 1, 2:size(x, ndims(x) - 1)), ndims(x), 2:last_dim)
    new_dims = (size(second_component)[1:end-2]..., prod(size(second_component)[end-1:end]))

    return cat(
        first_component,
        reshape(second_component, new_dims),
        dims = ndims(x) - 1)
end

function flatten_repeated_last_dim(x::Vector{<:AbstractMatrix{Float32}})
    return hcat(flatten_repeated_last_dim.(x)...)
end


"""
mask:   (sequence x sequence)
x:      (sequence x batch)
y:      (features x sequence x batch)
"""
struct PolynomialInterpolation
    mask::AbstractMatrix
    x::AbstractMatrix
    y::AbstractArray
end

Flux.@functor PolynomialInterpolation
Flux.trainable(interp::PolynomialInterpolation) = (;interp.y)

function PolynomialInterpolation(x::AbstractArray, y::AbstractArray)
    mask = I(size(x, 1))
    return PolynomialInterpolation(mask, x, y)
end

function (interp::PolynomialInterpolation)(t::AbstractVector{Float32})
    scale = Flux.unsqueeze(maximum(abs.(interp.x), dims = 1), 1)
    n = interp.mask .+ (.!interp.mask) .* Flux.unsqueeze(interp.x .- permutedims(t), 2)
    numer = Flux.prod(n ./ scale .+ 1f-5, dims = 1)

    T = Flux.unsqueeze(interp.x, 2) .- Flux.unsqueeze(interp.x, 1)
    d = T .+ interp.mask
    denom = Flux.prod(d ./ scale .+ 1f-5, dims = 1)
    coef = numer ./ denom
    return dropdims(sum(interp.y .* coef, dims = 2), dims = 2)
end

function linear_interp(X::AbstractMatrix{Float32}, Y::AbstractArray{Float32, 3}, x::AbstractVector{Float32})
    x_row = permutedims(x)

    d = X .- x_row
    ΔYΔX = diff(Y, dims = 2) ./ Flux.unsqueeze(diff(d, dims = 1), 1)
    l = X[1:end-1, :]
    r = X[2:end, :]

    ## short circut evaluation used to cover edge case when x is the final X point
    final_step = r .== r[[end], :] .== x_row
    mask = (l .<= x_row .< r) .|| final_step

    x0 = sum(X[1:end-1, :] .* mask, dims = 1)
    y0 = dropdims(sum(Y[:, 1:end-1, :] .* Flux.unsqueeze(mask, 1), dims = 2), dims = 2)
    dydx = dropdims(sum(ΔYΔX .* Flux.unsqueeze(mask, 1), dims = 2), dims = 2)

    return y0 .+ (permutedims(x) .- x0) .* dydx
end

struct LinearInterpolation
    X::AbstractMatrix{Float32}
    Y::AbstractArray{Float32, 3}
end

Flux.@functor LinearInterpolation

function (interp::LinearInterpolation)(x::AbstractVector)
    return linear_interp(interp.X, interp.Y, x)
end