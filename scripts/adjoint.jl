using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, batch, unbatch, pullback, withgradient, mean, @adjoint
using Waves

include("dynamics.jl")
include("plot.jl")

function continuous_backprop(iter::Integrator, wave::AbstractMatrix{Float32}, adj::AbstractArray{Float32, 3})
    tspan = build_tspan(iter.ti, iter.dt, iter.steps)
    iter_reverse = reverse(iter)

    wave = deepcopy(wave)

    gs = [adj[:, :, end]]

    for i in reverse(axes(tspan, 1))
        _, back = pullback(_wave -> iter(_wave, tspan[i]), wave)
        a = adj[:, :, i] .+ back(adj[:, :, i])[1]
        push!(gs, a)
        wave = wave .+ iter_reverse(wave, tspan[i])
    end

    return dropdims(sum(batch(gs), dims = 3), dims = 3)
end

Flux.trainable(iter::Integrator) = ()

@adjoint function (iter::Integrator)(ui::AbstractMatrix{Float32})

    u = iter(ui)
    uf = u[:, :, end]

    back = function(adj::AbstractArray{Float32, 3}) 
        return (nothing, continuous_backprop(iter, uf, adj))
    end

    return u, back
end

grid_size = 10.f0
elements = 1024
ti = 0.0f0
dt = 0.00002f0
steps = 100

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

dim = OneDim(grid_size, elements)
grad = build_gradient(dim)
pulse = Pulse(dim, 0.0f0, pulse_intensity)

ui = build_wave(dim, fields = 2)
ui = pulse(ui)

dynamics = LinearWave(ambient_speed, grad, dirichlet(dim))
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

model = Chain(
    Integrator(runge_kutta, dynamics, ti, dt, steps),
    flatten,
    Dense(2 * elements, 2 * elements, tanh),
    Dense(2 * elements, 1),
    vec
    )

y = sin.(2pi*range(0.0f0, 1.0f0, steps + 1))

opt = Adam(1e-5)

ps = Flux.params(model)

for i in 1:50
    loss, gs = withgradient(() -> mse(model(ui), y), ps)
    Flux.Optimise.update!(opt, ps, gs)
    println(loss)
end

sigma, back = pullback(model, ui)
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, y, label = "True")
lines!(ax, sigma, label = "Prediction")
save("y.png", fig)


# for i in 1:100

#     u, back2 = pullback(iter, ui)
#     e, back1 = pullback(_u -> Flux.mean(sum(_u[:, 1, :] .^ 2, dims = 1), dims = 2), u)
#     adj_0 = back2(back1(one(e))[1])[1]

#     # u = iter(ui)
#     # uf = u[:, :, end]
#     # e, back = pullback(_u -> Flux.mean(sum(_u[:, 1, :] .^ 2, dims = 1), dims = 2), u)
#     # adj = back(one(e))[1]
    



#     # adj_0 = continuous_backprop(iter, uf, adj)
#     Flux.Optimise.update!(opt, ui, adj_0) ## low wave speed
#     println(e)
# end

# u = iter(ui)

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)

# record(fig, "u.mp4", axes(u, 3)) do i
#     empty!(ax)
#     lines!(ax, dim.x, u[:, 1, i], color = :blue)
#     println(i)
# end

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, minimum(u[:, 2, :]), maximum(u[:, 2, :]))

# record(fig, "v.mp4", axes(u, 3)) do i
#     empty!(ax)
#     lines!(ax, dim.x, u[:, 2, i], color = :blue)
#     println(i)
# end

# save("results/u.png", lines(dim.x, ui[:, 1], title = "Displacement"))
# save("results/v.png", lines(dim.x, ui[:, 2]))