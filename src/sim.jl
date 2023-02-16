export gradient, split_wave!, wave!

"""
Function for constructing a gradient operator for a one dimensional scalar field.
"""
function gradient(x::Vector)
    grad = zeros(size(x, 1), size(x, 1))
    Δ = (x[end] - x[1]) / (length(x) - 1)

    grad[[1, 2, 3], 1] .= [-3.0, 4.0, -1.0] ## left boundary edge case
    grad[[end-2, end-1, end], end] .= [1.0, -4.0, 3.0] ## right boundary edge case

    for i ∈ 2:(size(grad, 2) - 1)
        grad[[i - 1, i + 1], i] .= [-1.0, 1.0]
    end

    return sparse((grad / (2 * Δ))')
end

"""
Obtains the first derivative of a a one dimensional function using central differences for
points in the interior and forward / backward differences for the boundary points.
"""
function ∇(u::Vector, Δ::Float64)::Vector
    du = zeros(size(u))

    for i ∈ 2:(size(u, 1) - 1)
        du[i] = (u[i + 1] - u[i - 1]) / (2 * Δ)
    end

    du[1] = (u[2] - u[1]) / Δ
    du[end] = (u[end] - u[end - 1]) / Δ

    return du
end

"""
Computes the first derivative of the rows of a matrix. This is assumed to be the x axis of
the two dimensional function.
"""
function ∇x(u::Matrix, Δ::Float64)::Matrix
    dx_u = zeros(size(u))

    for i ∈ axes(u, 1)
        dx_u[:, i] .= ∇(u[:, i], Δ)
    end

    return dx_u
end

function ∇x(u::Array{Float64, 3}, Δ::Float64)::Array{Float64, 3}
    dx_u = zeros(size(u))

    for i ∈ axes(u, 3)
        dx_u[:, :, i] .= ∇x(u[:, :, i], Δ)
    end

    return dx_u
end

"""
Obtains the first derivative of the columns of the matrix. This is assumed to be the y axis
of the function.
"""
function ∇y(u::Matrix, Δ::Float64)::Matrix
    dy_u = zeros(size(u))

    for i ∈ axes(u, 2)
        dy_u[i, :] .= ∇(u[i, :], Δ)
    end

    return dy_u
end

function ∇y(u::Array{Float64, 3}, Δ::Float64)::Array{Float64, 3}
    dy_u = zeros(size(u))

    for i ∈ axes(u, 3)
        dy_u[i, :, :] .= ∇y(u[i, :, :], Δ)
    end

    return dy_u
end

function ∇z(u::Array{Float64, 3}, Δ::Float64)::Array{Float64, 3}
    dz_u = zeros(size(u))

    for i ∈ axes(u, 3)
        dz_u[i, :, :] .= ∇x(u[i, :, :], Δ)
    end

    return dz_u
end

"""
The split_wave! formulation of the second order wave equation. This specific function is for the
one dimensional case.
"""
function split_wave!(du::Matrix{Float64}, u::Matrix{Float64}, p, t)
    Δ, C, pml = p

    U = u[:, 1]
    V = u[:, 2]

    du[:, 1] .= C(t) .* ∇(V, Δ) .- U .* pml
    du[:, 2] .= ∇(U, Δ) .- V .* pml
end

"""
This is the split_wave! formulation of the second order acoustic wave equation for a two dimensional plane.
"""
function split_wave!(du::Array{Float64, 3}, u::Array{Float64, 3}, p, t)
    Δ, C, pml = p

    U = u[:, :, 1]
    Vx = u[:, :, 2]
    Vy = u[:, :, 3]

    du[:, :, 1] .= C(t) .* (∇x(Vx, Δ) .+ ∇y(Vy, Δ)) .- U .* pml
    du[:, :, 2] .= ∇x(U, Δ) .- Vx .* pml
    du[:, :, 3] .= ∇y(U, Δ) .- Vy .* pml
end

function split_wave!(du::Array{Float64, 4}, u::Array{Float64, 4}, p, t)
    Δ, C, pml = p

    U = u[:, :, :, 1]
    Vx = u[:, :, :, 2]
    Vy = u[:, :, :, 3]
    Vz = u[:, :, :, 4]

    du[:, :, :, 1] .= C(t) .* (∇x(Vx, Δ) .+ ∇y(Vy, Δ) .+ ∇z(Vz, Δ)) .- U .* pml
    du[:, :, :, 2] .= ∇x(U, Δ) .- Vx .* pml
    du[:, :, :, 3] .= ∇y(U, Δ) .- Vy .* pml
    du[:, :, :, 4] .= ∇z(U, Δ) .- Vz.* pml
end

function wave!(du::AbstractArray{<: AbstractFloat, 3}, u::AbstractArray{<: AbstractFloat, 3}, p, t::AbstractFloat)
    grad, C, pml = p

    U = u[:, :, 1]
    Vx = u[:, :, 2]
    Vy = u[:, :, 3]

    du[:, :, 1] .= C(t) .* (grad * Vx .+ (grad * Vy')') .- U .* pml
    du[:, :, 2] .= (grad * U) .- Vx .* pml
    du[:, :, 3] .= (grad * U')' .- Vy .* pml
end