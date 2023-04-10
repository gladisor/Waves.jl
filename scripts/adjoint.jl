using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, batch, unbatch, pullback, withgradient
using Waves

include("dynamics.jl")
include("plot.jl")

function continuous_backprop(iter::Integrator, wave, adj)

    iter = reverse(iter)
    tspan = build_tspan(iter.ti, iter.dt, iter.steps)

    wave = deepcopy(wave)
    adj = deepcopy(adj)

    for i in axes(tspan, 1)
        du, back = pullback(_wave -> iter(_wave, tspan[i]), wave)
        adj = adj .+ back(adj)[1]
        wave = wave .+ du
    end

    return wave, adj
end

grid_size = 10.f0
elements = 512
ti = 0.0f0
dt = 0.00002f0
steps = 300

ambient_speed = 1531.0f0
# ambient_speed = 1.0f0
pulse_intensity = 1.0f0

dim = OneDim(grid_size, elements)
grad = build_gradient(dim)
pulse = Pulse(dim, 0.0f0, pulse_intensity)
ui = pulse(build_wave(dim, fields = 2)) |> gpu

dynamics = LinearWave(ambient_speed, grad, dirichlet(dim))
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)
iter_reverse = reverse(iter)
tspan = build_tspan(iter.ti, iter.dt, iter.steps)

opt = Momentum(1e-8)
# opt = Momentum(1e-3)

for i in 1:100
    u = iter(ui)
    uf = u[:, :, end]
    e, back = pullback(_uf -> sum(_uf[:, 1] .^ 2), uf)
    adj = back(one(e))[1]

    ui_0, adj_0 = continuous_backprop(iter, uf, adj)

    println(e)

    Flux.Optimise.update!(opt, ui, -adj_0)
end

u = iter(ui)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "vid.mp4", axes(u, 3)) do i
    empty!(ax)
    lines!(ax, dim.x, u[:, 1, i], color = :blue)
    println(i)
end

save("results/u.png", lines(ui[:, 1]))
save("results/v.png", lines(ui[:, 2]))
