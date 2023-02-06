using DifferentialEquations
import GLMakie

using Waves

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

    du[:, 1] .= C .* ∇(V, Δ) .- U .* pml
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

using Waves: AbstractDim

struct WaveSpeed{D <: AbstractDim}
    dim::D
    C0::AbstractArray
    design::Union{DesignInterpolator, Nothing}
end

function WaveSpeed(dim::AbstractDim, C0::Float64, design::Union{DesignInterpolator, Nothing} = nothing)
    return WaveSpeed(dim, ones(size(dim)) * C0, design)
end

function (C::WaveSpeed)(t::Float64)
    if isnothing(C.design)
        return C.C0
    else
        return C.C0 .- speed(C.dim, C.design(t))
    end
end

function build_pml(dim::ThreeDim, width::Float64)
    x, y, z = abs.(dim.x), abs.(dim.y), abs.(dim.z)

    start_x = x[end] - width
    start_y = y[end] - width
    start_z = z[end] - width

    pml = zeros(size(dim))

    for i ∈ axes(pml, 1)
        for j ∈ axes(pml, 2)
            for k ∈ axes(pml, 3)
                depth = maximum([x[i] - start_x, y[j] - start_y, z[k] - start_z, 0.0]) / width
                pml[i, j, k] = depth
            end
        end
    end

    return pml .^ 2
end

gs = 5.0
Δ = 0.5
pml_width = 2.0
pml_scale = 10.0
dim = ThreeDim(gs, Δ)
tspan = (0.0, 20.0)

# cyli = Cylinder(-3.0, 0.0, 2.0, 0.0)
# cylf = Cylinder(3.0, 0.0, 2.0, 0.0)
# design = DesignInterpolator(cyli, cylf - cyli, tspan...)

C = WaveSpeed(dim, 1.0, nothing)
u, v = gaussian_pulse(dim, 0.0, 0.0, 0.0, 1.0)
pml = build_pml(dim, pml_width) * pml_scale
u0 = cat(u, v, dims = (ndims(u) + 1))
p = [Δ, C, pml]

@time prob = ODEProblem(split_wave!, u0, tspan, p)
@time sol = solve(prob, Midpoint(thread = OrdinaryDiffEq.True()))
render!(sol, dim, path = "vid3d.mp4")

# g = grid(dim)
# mask = map(xy -> xy[1]^2 + xy[2]^2 <= 9, g)

# flux = Float64[]
# for t ∈ range(tspan..., 200)
#     push!(
#         flux, 
#         sum((∇x(∇x(sol(t)[:, :, 1], Δ), Δ) .+ ∇y(∇y(sol(t)[:, :, 1], Δ), Δ)) .* mask)
#         )
# end

# save(plot(collect(range(tspan..., 200)), flux), "flux.png")