using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, batch, unbatch, pullback, withgradient, mean
using Waves

include("dynamics.jl")
include("plot.jl")

function standard_gradient(iter::Integrator, ui::AbstractMatrix{Float32})

    e, back = pullback(ui) do _ui
        u = iter(_ui) ## solve from initial condition
        return sum(u[:, 1, end] .^ 2) ## compute energy of final state
    end

    gs = back(one(e))[1] ## obtain gradient
    return e, gs
end

function continuous_backprop(iter::Integrator, wave::AbstractMatrix{Float32}, adj::AbstractMatrix{Float32})
    
    tspan = build_tspan(iter.ti, iter.dt, iter.steps)
    iter_reverse = reverse(iter)

    wave = deepcopy(wave)
    adj = deepcopy(adj)

    for i in reverse(axes(tspan, 1))
        _, back = pullback(_wave -> iter(_wave, tspan[i]), wave)
        du = iter_reverse(wave, tspan[i])

        adj = adj .+ back(adj)[1]
        wave = wave .+ du
    end

    return wave, adj
end

# function continuous_backprop(iter::Integrator, u::AbstractArray{Float32, 3}, adj::AbstractMatrix{Float32})

#     tspan = build_tspan(iter.ti, iter.dt, iter.steps)
#     adj = deepcopy(adj)

#     for i in reverse(2:size(u, 3))
#         _, back = pullback(_wave -> iter(_wave, tspan[i]), u[:, :, i])
#         adj = adj .+ back(adj)[1]
#     end

#     return adj
# end

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

grid_size = 10.f0
elements = 1024
ti = 0.0f0
dt = 0.00002f0
steps = 200

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

dim = OneDim(grid_size, elements)
grad = build_gradient(dim)
pulse = Pulse(dim, 0.0f0, pulse_intensity)

ui = build_wave(dim, fields = 2)
ui = pulse(ui)

dynamics = LinearWave(ambient_speed, grad, dirichlet(dim))
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

opt = Momentum(1e-8)

for i in 1:100
    u = iter(ui)
    uf = u[:, :, end]
    e, back = pullback(_u -> Flux.mean(sum(_u[:, 1, :] .^ 2, dims = 1), dims = 2), u)
    adj = back(one(e))[1]

    adj_0 = continuous_backprop(iter, uf, adj)
    Flux.Optimise.update!(opt, ui, adj_0) ## low wave speed
    println(e)
end

u = iter(ui)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "u.mp4", axes(u, 3)) do i
    empty!(ax)
    lines!(ax, dim.x, u[:, 1, i], color = :blue)
    println(i)
end

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, minimum(u[:, 2, :]), maximum(u[:, 2, :]))

record(fig, "v.mp4", axes(u, 3)) do i
    empty!(ax)
    lines!(ax, dim.x, u[:, 2, i], color = :blue)
    println(i)
end

save("results/u.png", lines(dim.x, ui[:, 1], title = "Displacement"))
save("results/v.png", lines(dim.x, ui[:, 2]))