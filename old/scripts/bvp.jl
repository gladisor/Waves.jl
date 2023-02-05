using DifferentialEquations: BoundaryValueDiffEq, solve
using DifferentialEquations.BoundaryValueDiffEq: BVProblem, GeneralMIRK4, Shooting
using OrdinaryDiffEq: Tsit5
import GLMakie

function second_centered_diff(u::Vector, dx::Float64)::Vector
    du = zeros(size(u))

    for i ∈ 2:(size(u, 1) - 1)
        du[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ^ 2)
    end

    # du[1] = (u[3] - 2 * u[2] + u[1]) / (dx ^ 2)
    # du[end] = (u[end] - 2 * u[end-1] + u[end-2]) / (dx ^ 2)

    return du
end

function wave!(du, u, p, t)
    C = p[1]
    du[:, 2] .= second_centered_diff(u[:, 1], dx) # dt v
    du[:, 1] .= C .^2 .* u[:, 2] # dt u
end

function bc1!(residual, u, p, t)
    residual .= 0.0
    # residual[end, :] .= 0.0
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

gs = 5.0
dx = 0.1
x = collect(-gs:dx:gs)
gaussian_pulse(x) = exp.(- x .^ 2)
u = gaussian_pulse(x)
v = zeros(size(u))
u0 = hcat(u, v)
tspan = (0.0, 20.0)
C = ones(size(u))

C[end-20:end] .= 0.5
p = [C]

prob = BVProblem(wave!, bc1!, u0, tspan, p)
sol = solve(prob, Tsit5())

render!(sol, x, 200, path = "vid.mp4")