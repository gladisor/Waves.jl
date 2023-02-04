using DifferentialEquations
import GLMakie

function centered_diff(u::Vector, dx::Float64)::Vector

    du = zeros(size(u))

    for i ∈ 2:(size(u, 1) - 1)
        du[i] = (u[i + 1] - u[i - 1]) / (2 * dx)
    end

    du[1] = (u[2] - u[1]) / dx
    du[end] = (u[end] - u[end - 1]) / dx


    return du
end

function second_centered_diff(u::Vector, dx::Float64)::Vector
    du = zeros(size(u))

    for i ∈ 2:(size(u, 1) - 1)
        du[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ^ 2)
    end

    du[1] = (u[3] - 2 * u[2] + u[1]) / (dx ^ 2)
    du[end] = (u[end] - 2 * u[end-1] + u[end-2]) / (dx ^ 2)

    return du
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

function wave!(du, u, p, t)
    C = p[1]
    du[:, 2] .= second_centered_diff(u[:, 1], dx) # dt v
    du[:, 1] .= C^2 * u[:, 2] # dt u
end

function split_wave!(du, u, p, t)
    C, pml = p

    du[:, 1] .= centered_diff(u[:, 2], dx) .- u[:, 1] .* pml
    du[:, 2] .= centered_diff(u[:, 1], dx) .- u[:, 2] .* pml
end

gs = 10.0
dx = 0.1
x = collect(-gs:dx:gs)
gaussian_pulse(x) = exp.(- x .^ 2)
u = gaussian_pulse(x)
v = zeros(size(u))

u0 = hcat(u, v)
tspan = (0.0, 20.0)

pml_width = 3.0
pml_start = gs - pml_width
pml = abs.(x)
pml[pml .< pml_start] .= 0.0
pml[pml .> 0.0] .= (pml[pml .> 0.0] .- pml_start) ./ pml_width
p = [1.0, pml]

prob = ODEProblem(split_wave!, u0, tspan, p)
sol = solve(prob, RK4())
render!(sol, x, 200, path = "vid.mp4")

