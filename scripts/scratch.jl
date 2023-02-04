import GLMakie

function diff_x(u, dx)

    dx_u = zeros(size(u))

    for i ∈ 2:(length(u) - 1)
        dx_u[i] = (u[i + 1] - u[i]) / dx
    end

    return dx_u
end

function diff_xx(u, dx)
    dxx_u = zeros(size(u))

    for i ∈ 2:(length(u) - 1)
        dxx_u[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx^2
    end

    return dxx_u
end

function integrate(u0, v0, dx, dt, n)
    u = u0
    v = v0

    sol = [u]

    for _ ∈ 1:n
        v += diff_xx(u, dx) * dt
        # v += diff_x(diff_x(u, dx), dx) * dt
        u += v * dt
        push!(sol, u)
    end

    return sol
end

function render!(sol)

    fig = GLMakie.Figure()
    ax = GLMakie.Axis(fig[1, 1])

    GLMakie.xlims!(-5.0, 5.0)
    GLMakie.ylims!(-1.0, 1.0)

    GLMakie.record(fig, "vid.mp4", sol) do i
        GLMakie.empty!(ax.scene)
        GLMakie.lines!(ax, x, i, color = :blue, linewidth = 2)
    end

    return nothing
end

gs = 5.0
dx = 0.1
dt = 0.1
n = 200

x = collect(-gs:dx:gs)
gaussian_pulse(x) = exp.(- x .^ 2)
u = gaussian_pulse(x)
v = zeros(size(u))

sol = integrate(u, v, dx, dt, n)

render!(sol)