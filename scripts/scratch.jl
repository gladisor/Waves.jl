using DifferentialEquations
import GLMakie

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

    du[:, 1] .= sqrt.(C) .* ∇(V, Δ) .- U .* pml
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

    du[:, :, 1] .= C .* (∇x(Vx, Δ) .+ ∇y(Vy, Δ)) .- U .* pml
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

function gaussian_pulse(x::Vector, x′::Float64, intensity::Float64)::Vector
    return exp.( - intensity * (x .- x′) .^ 2)
end

function gaussian_pulse(x::Vector, y::Vector, x′::Float64, y′::Float64, intensity::Float64)

    u = zeros(length(x), length(y))

    for i ∈ axes(x, 1)
        for j ∈ axes(y, 1)
            u[i, j] = exp(- intensity * ((x[i] - x′) ^ 2 + (y[j] - y′) ^ 2))
        end
    end

    return u
end

function plot(x::Vector, u::Vector)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X (m)", ylabel = "Displacement (m)")
    GLMakie.xlims!(ax, x[1], x[end])
    GLMakie.ylims!(ax, -1.0, 1.0)
    GLMakie.lines!(ax, x, u, color = :blue)
    return fig
end

function plot(x::Vector, y::Vector, u::Matrix)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Displacement (m)")
    GLMakie.xlims!(ax, x[1], x[end])
    GLMakie.ylims!(ax, y[1], y[end])
    GLMakie.zlims!(ax, -1.0, 5.0)
    GLMakie.surface!(ax, x, y, u, colormap = :ice)
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

function scatterer(x, x′)
    return max.(-(x .- x′) .^ 10 .+ 1.0, 0.0)
end

gs = 10.0
Δ = 0.1

pml_width = 2.0
pml_scale = 10.0

pulse_intensity = 10.0
pulse_x = 5.0
pulse_y = 5.0

x = collect(-gs:Δ:gs)
y = collect(-gs:Δ:gs)
tmax = 20.0

pml = build_pml(x, pml_width) * pml_scale
pml = (pml .+ pml') ./ 2
u = gaussian_pulse(x, y, pulse_x, pulse_y, pulse_intensity)
v = zeros(size(u)..., 2)
u0 = cat(u, v, dims = (ndims(u) + 1))
tspan = (0.0, tmax)

# C = 1.0 .- (scatterer(x, -10.0) .+ scatterer(x, 10.0))
C = ones(size(u))
p = [Δ, C, pml]
@time prob = ODEProblem(split_wave!, u0, tspan, p)
@time sol = solve(prob, RK4())
render!(sol, x, y, 200, path = "vid.mp4")
# @time render2d!(sol, x, y, 200, path = "vid2d.mp4")