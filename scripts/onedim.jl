using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, batch, unbatch, pullback, withgradient
using Waves

include("dynamics.jl")
include("plot.jl")

function displacement_gradient(wave::AbstractMatrix{Float32})
    e, back = pullback(_wave -> sum(_wave[:, 1] .^ 2), wave)
    return e, back(one(e))[1]
end

function standard_gradient(iter::Integrator, ui::AbstractMatrix{Float32})

    e, back = pullback(ui) do _ui
        u = iter(_ui) ## solve from initial condition
        return sum(u[:, 1, end] .^ 2) ## compute energy of final state
    end

    gs = back(one(e))[1] ## obtain gradient
    return e, gs
end

function adjoint_sensitivity(iter::Integrator, ui::AbstractMatrix{Float32})
    
    ## compute final time
    tf = iter.ti + iter.steps * iter.dt
    ## build a reversed tspan starting from the final time
    tspan = build_tspan(tf, -iter.dt, iter.steps)

    ## obtain an Integrator that can step backwards
    iter_backward = reverse(iter)

    ## compute forward solution from the current state
    u = iter(ui)

    ## take the last state of the forward solution
    wave = u[:, :, end]

    ## compute the energy gradient of the final state
    # e, back = pullback(_wave -> sum(_wave[:, 1] .^ 2), wave)
    d, adj = displacement_gradient(wave)

    adj = back(one(e))[1]

    for i in axes(tspan, 1)
        t = tspan[i]
        dwave, back = pullback(_wave -> iter_backward(_wave, t), wave)
        dadj = back(adj)[1]

        adj = adj .- dadj
        wave = wave .- dwave
    end

    return d, adj
end

grid_size = 10.f0
elements = 512
ti = 0.0f0
dt = 0.00002f0
steps = 100

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

#=
Running a one dim wave simulation forward in time.
Simple boundary conditions which make walls reflective.
=#
dim = OneDim(grid_size, elements)
grad = build_gradient(dim)
pulse = Pulse(dim, 0.0f0, pulse_intensity)
ui = pulse(build_wave(dim, fields = 2)) |> gpu

dynamics = LinearWave(ambient_speed, grad, dirichlet(dim))
iter = Integrator(runge_kutta, dynamics, ti, dt, steps) |> gpu
iter_reverse = reverse(iter)
tspan = build_tspan(iter.ti, iter.dt, iter.steps)
tspan_reverse = build_tspan(iter_reverse.ti, iter_reverse.dt, iter.steps - 1)

u = iter(ui)
wave = u[:, :, end]
d, adj = displacement_gradient(wave)

dwave, back = pullback(_wave -> iter_reverse(_wave, tspan_reverse[1]), wave)
dadj = back(adj)[1]

plot_wave!(dim, adj, path = "adj.png")
plot_wave!(dim, wave, path = "wave.png")
plot_wave!(dim, dwave, path = "dwave.png")
plot_wave!(dim, dadj, path = "dadj.png")

wave = wave .- dwave
adj = adj .- dadj

dwave, back = pullback(_wave -> iter_reverse(_wave, tspan_reverse[2]), wave)
dadj = back(adj)[1]

plot_wave!(dim, adj, path = "adj-1.png")
plot_wave!(dim, wave, path = "wave-1.png")
plot_wave!(dim, dwave, path = "dwave-1.png")
plot_wave!(dim, dadj, path = "dadj-1.png")

wave = wave .- dwave
adj = adj .- dadj

dwave, back = pullback(_wave -> iter_reverse(_wave, tspan_reverse[3]), wave)
dadj = back(adj)[1]

plot_wave!(dim, adj, path = "adj-2.png")
plot_wave!(dim, wave, path = "wave-2.png")
plot_wave!(dim, dwave, path = "dwave-2.png")
plot_wave!(dim, dadj, path = "dadj-2.png")

####################################################################################
# plot_wave!(dim, ui, path = "ui_original.png")
# plot_wave!(dim, u[:, :, end], path = "uf_original.png")
# plot_solution!(dim, tspan, u, path = "sol_original.png")

# e, gs = standard_gradient(iter, ui)
# plot_wave!(dim, gs, path = "standard_gs.png")

# e, gs = adjoint_sensitivity(iter, ui)
# plot_wave!(dim, gs, path = "adj_sensitivity_gs.png")

# # ################ Standard Gradient:
# opt = Descent(1e-2)
# # opt = Momentum(1e-4)
# opt = Nesterov(1e-4)
# opt = Adam(1e-2)
# # ################ Adjoint Sensitivity:
# opt = Descent(1e-2)

# for i in 1:40
#     # e, gs = standard_gradient(iter, ui)
#     e, gs = adjoint_sensitivity(iter, ui)
#     Flux.Optimise.update!(opt, ui, gs)
#     println(e)
# end

# u = iter(ui)
# plot_wave!(dim, ui, path = "ui_opt.png")
# plot_wave!(dim, u[:, :, end], path = "uf_opt.png")
# plot_solution!(dim, tspan, u, path = "sol_opt.png")
