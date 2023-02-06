using DifferentialEquations
import GLMakie

using Waves
using Waves: AbstractDim, TwoDim

"""
Obtains the first derivative of a a one dimensional function using central differences for
points in the interior and forward / backward differences for the boundary points.
"""
function ∇(u::Vector, Δ::Float64)::Vector
    du = zeros(size(u))

    for i ∈ 2:(size(u, 1) - 1)
        du[i] = (u[i + 1] - u[i - 1]) / (2 * Δ)
    end

    # for i ∈ 3:(size(u, 1) - 2) ## second order centeral difference
    #     du[i] = (-u[i + 2] + 8 * u[i + 1] - 8 * u[i - 1] + u[i - 2]) / (12 * Δ)
    # end

    du[1] = (u[2] - u[1]) / Δ
    # du[2] = (u[3] - u[1]) / (2 * Δ)
    # du[end - 1] = (u[end] - u[end - 2]) / (2 * Δ)
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
    display(t)
    Δ, C, pml = p

    U = u[:, :, 1]
    Vx = u[:, :, 2]
    Vy = u[:, :, 3]

    du[:, :, 1] .= (1.0 .- C(t)) .* (∇x(Vx, Δ) .+ ∇y(Vy, Δ)) .- U .* pml
    du[:, :, 2] .= ∇x(U, Δ) .- Vx .* pml
    du[:, :, 3] .= ∇y(U, Δ) .- Vy .* pml
end

function render!(sol, x, n; path)

    fig = GLMakie.Figure()
    ax = GLMakie.Axis(fig[1, 1])

    GLMakie.xlims!(x[1], x[end])
    GLMakie.ylims!(-1.0, 1.0)

    dt = (sol.prob.tspan[end] - sol.prob.tspan[1]) / n

    GLMakie.record(fig, path, 1:n) do i
        GLMakie.empty!(ax.scene)
        GLMakie.lines!(ax, x, sol(i * dt)[:, 1], color = :blue, linewidth = 2)
    end

    return nothing
end

function render!(sol, x, y, n; path)

    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Displacement (m)")

    GLMakie.xlims!(x[1], x[end])
    GLMakie.ylims!(y[1], y[end])
    GLMakie.zlims!(-1.0, 4.0)

    dt = (sol.prob.tspan[end] - sol.prob.tspan[1]) / n

    GLMakie.record(fig, path, 1:n) do i
        GLMakie.empty!(ax.scene)
        GLMakie.surface!(ax, x, y, sol(i * dt)[:, :, 1], colormap = :ice, linewidth = 2)
    end

    return nothing
end

function plane_wave(x::Vector, y::Vector, x_pos::Float64, intensity::Float64)
    u0 = zeros(length(x), length(y))

    for i ∈ axes(u0, 1)
        for j ∈ axes(u0, 2)
            u0[i, j] = exp(- intensity * (x[i] - x_pos) ^ 2)
        end
    end

    return u0
end

function plot(x::Vector, u::Vector)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X (m)", ylabel = "Displacement (m)")
    GLMakie.xlims!(ax, x[1], x[end])
    GLMakie.ylims!(ax, -1.0, 1.0)
    GLMakie.lines!(ax, x, u, color = :blue)
    return fig
end

"""
Assuming an x axis which is symmetric build a vector which contains zeros in the
interior and slowly scales from zero to one at the edges.
"""
function build_pml(x::Vector, width::Float64)
    start = maximum(x) - width ## define the starting x value of the pml
    pml = abs.(x) 
    pml[pml .< start] .= 0.0 ## sets points which are in the interior to be zero
    pml[pml .> 0.0] .= (pml[pml .> 0.0] .- start) ./ width ## scales the points between zero and one
    pml = pml .^ 2 ## squishes the function to be applied more gradually
    return pml
end

function build_pml(dim::TwoDim, width::Float64)
    x, y = abs.(dim.x), abs.(dim.y)

    start_x = x[end] - width
    start_y = y[end] - width

    pml = zeros(length(x), length(y))

    for i ∈ axes(pml, 1)
        for j ∈ axes(pml, 2)
            depth = maximum([x[i] - start_x, y[j] - start_y, 0.0]) / 2
            pml[i, j] = depth
        end
    end

    return pml .^ 2
end

abstract type Scatterer end

struct Cylinder <: Scatterer
    x
    y
    r
    c
end

function GLMakie.mesh!(ax::GLMakie.Axis3, cyl::Cylinder)
    GLMakie.mesh!(ax, GLMakie.GeometryBasics.Cylinder3{Float32}(GLMakie.Point3f(cyl.x, cyl.y, -1.0), GLMakie.Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

function Base.:∈(xy::Tuple, cyl::Cylinder)
    return ((xy[1] - cyl.x) ^ 2 + (xy[2] - cyl.y) ^ 2) <= cyl.r ^ 2
end

function Base.:+(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x + cyl2.x, cyl1.y + cyl2.y, cyl1.r + cyl2.r, cyl1.c + cyl2.c)
end

function Base.:-(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x - cyl2.x, cyl1.y - cyl2.y, cyl1.r - cyl2.r, cyl1.c - cyl2.c)
end

function Base.:*(cyl::Cylinder, m::Float64)
    return Cylinder(cyl.x * m, cyl.y * m, cyl.r * m, cyl.c * m)
end

function Base.:*(m::Float64, cyl::Cylinder)
    return cyl * m
end

function speed(dim::TwoDim, cyl::Cylinder)
    C = zeros(size(dim))

    for i ∈ axes(C, 1)
        for j ∈ axes(C, 2)
            if (dim.x[i], dim.y[j]) ∈ cyl
                C[i, j] = 0.8
            end
        end
    end

    return C
end

function gaussian_pulse(dim::TwoDim, x, y, intensity)

    u = zeros(length(dim.x), length(dim.y))

    for i ∈ axes(u, 1)
        for j ∈ axes(u, 2)
            u[i, j] = exp(- intensity * ((dim.x[i] - x) ^ 2 + (dim.y[j] - y) ^ 2))
        end
    end

    v = zeros(size(u)..., 2)

    return u, v
end

struct Wave{D <: AbstractDim}
    dim::D
    u::AbstractArray{Float64}
end

function plot(dim::TwoDim)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Displacement (m)")
    GLMakie.xlims!(ax, dim.x[1], dim.x[end])
    GLMakie.ylims!(ax, dim.y[1], dim.y[end])
    GLMakie.zlims!(ax, -1.0, 5.0)
    return fig
end

function plot!(fig::GLMakie.Figure, wave::Wave{TwoDim})
    GLMakie.surface!(fig.content[1], wave.dim.x, wave.dim.y, wave.u, colormap = :ice)
end

function plot(wave::Wave{TwoDim})
    dim = wave.dim
    fig = plot(dim)
    plot!(fig, wave)
    return fig
end

function save(fig::GLMakie.Figure, path::String)
    GLMakie.save(path, fig)
end

function plot!(fig::GLMakie.Figure, cyl::Cylinder)
    GLMakie.mesh!(fig.content[1], cyl)
end

struct DesignInterpolator
    initial::Scatterer
    Δ::Scatterer
    ti::Float64
    tf::Float64
end

function (interp::DesignInterpolator)(t::Float64)
    t = (t - interp.ti) / (interp.tf - interp.ti)
    return interp.initial + t * interp.Δ
end

gs = 10.0
Δ = 0.2

dim = TwoDim(gs, Δ)
pml_width = 2.0
pml_scale = 10.0

tspan = (0.0, 20.0)

cyli = Cylinder(-3.0, 3.0, 1.0, 0.0)
cylf = Cylinder(3.0, -3.0, 1.0, 0.0)
design = DesignInterpolator(cyli, cylf - cyli, tspan...)
C = t -> speed(dim, design(t))
pml = build_pml(dim, pml_width) * pml_scale
u, v = gaussian_pulse(dim, 2.5, 2.5, 1.0)
u0 = cat(u, v, dims = (ndims(u) + 1))
p = [Δ, C, pml]

@time prob = ODEProblem(split_wave!, u0, tspan, p)
@time sol = solve(prob, Midpoint())
render!(sol, dim.x, dim.y, 200, path = "vid.mp4")

flux = Float64[]
for t ∈ range(tspan..., 200)
    push!(
        flux, 
        sum(∇x(∇x(sol(t)[:, :, 1], Δ), Δ) .+ ∇y(∇y(sol(t)[:, :, 1], Δ), Δ))
        )
end

save(plot(collect(range(tspan..., 200)), flux), "flux.png")