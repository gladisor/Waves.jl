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
        dx_u[i, :] .= ∇(u[i, :], Δ)
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
        dy_u[:, i] .= ∇(u[:, i], Δ)
    end

    return dy_u
end