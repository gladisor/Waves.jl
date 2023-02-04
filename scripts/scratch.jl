using DifferentialEquations
import GLMakie

function ∇(u::Vector, Δ::Float64)::Vector
    du = zeros(size(u))

    for i ∈ 2:(size(u, 1) - 1)
        du[i] = (u[i + 1] - u[i - 1]) / (2 * Δ)
    end

    du[1] = (u[2] - u[1]) / Δ
    du[end] = (u[end] - u[end - 1]) / Δ

    return du
end

function ∇ₓ(u::Matrix, Δ::Float64)::Matrix
    dx_u = zeros(size(u))

    for i ∈ axes(u, 1)
        dx_u[i, :] .= ∇(u[i, :], Δ)
    end

    return dx_u
end

# function second_centered_diff(u::Vector, dx::Float64)::Vector
#     du = zeros(size(u))

#     for i ∈ 2:(size(u, 1) - 1)
#         du[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ^ 2)
#     end

#     du[1] = (u[3] - 2 * u[2] + u[1]) / (dx ^ 2)
#     du[end] = (u[end] - 2 * u[end-1] + u[end-2]) / (dx ^ 2)

#     return du
# end

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

function wave!(du, u, p, t)
    C = p[1]
    du[:, 2] .= second_centered_diff(u[:, 1], dx) # dt v
    du[:, 1] .= C^2 * u[:, 2] # dt u
end

function split_wave!(du, u, p, t)
    Δ, C, pml = p

    U = u[:, 1]
    V = u[:, 2]

    du[:, 1] .= C * ∇(V, Δ) .- U .* pml
    du[:, 2] .= ∇(U, Δ) .- V .* pml
end

struct Point{D}
    dim::D
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

function plot(x::Vector, y::Vector, u::Matrix)
    fig = GLMakie.Figure(resolution = (1920, 1080))
    ax = GLMakie.Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Displacement (m)")
    GLMakie.xlims!(ax, x[1], x[end])
    GLMakie.ylims!(ax, y[1], y[end])
    GLMakie.zlims!(ax, -1.0, 5.0)
    GLMakie.surface!(ax, x, y, u, colormap = :ice)
    return fig
end

gs = 5.0
dx = 0.1
Δ = 0.1
x = collect(-gs:Δ:gs)
y = collect(-gs:Δ:gs)

# u = gaussian_pulse(x, 0.0, 1.0)
# u′ = ∇(u, Δ)
u = gaussian_pulse(x, y, 0.0, 0.0, 1.0)
v = zeros(size(u))
# u0 = hcat(u, v)

fig = plot(x, y, u)
GLMakie.save("surf.png", fig)

# tspan = (0.0, 20.0)

# pml_width = 1.0
# pml_start = gs - pml_width
# pml = abs.(x)
# pml[pml .< pml_start] .= 0.0
# pml[pml .> 0.0] .= (pml[pml .> 0.0] .- pml_start) ./ pml_width
# pml = 30 * pml .^ 2

# p = [Δ, 4.0, pml]

# prob = ODEProblem(split_wave!, u0, tspan, p)
# sol = solve(prob, RK4())
# render!(sol, x, 200, path = "vid.mp4")

