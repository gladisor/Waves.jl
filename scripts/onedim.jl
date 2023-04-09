using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, batch, unbatch, pullback, withgradient
using Waves

include("dynamics.jl")

function plot_solution!(dim::OneDim, tspan::Vector{Float32}, u::AbstractArray{Float32, 3}; path::String)
    fig = Figure()
    ax1 = Axis(fig[1, 1], title = "Displacement", ylabel = "Time (s)", xlabel = "Distance (m)")
    # ax2 = Axis(fig[1, 2], title = "Velocity", xlabel = "Distance (m)")
    heatmap!(ax1, dim.x, tspan, u[:, 1, :], colormap = :ice)
    # heatmap!(ax2, dim.x, tspan, u[:, 2, :], colormap = :ice)
    save(path, fig)
end

grid_size = 10.f0
elements = 512

ti = 0.0f0
dt = 0.00002f0
steps = 50
ambient_speed = 1531.0f0
pulse_intensity = 1.0f0
pml_width = 2.0f0
pml_scale = ambient_speed * 50.0f0

#=
Running a one dim wave simulation forward in time.
Simple boundary conditions which make walls reflective.
=#
dim = OneDim(grid_size, elements)
tspan = build_tspan(iter.ti, iter.dt, iter.steps)

grad = build_gradient(dim)
pml = build_pml(dim, pml_width, pml_scale)
pulse = Pulse(dim, 0.0f0, pulse_intensity)
ui = pulse(build_wave(dim, fields = 2)) |> gpu


latent_ambient_speed = 1.0f0
latent_dt = 0.01

dynamics = LinearWave(latent_ambient_speed, grad, dirichlet(dim))
iter = Integrator(runge_kutta, dynamics, ti, latent_dt, steps) |> gpu

opt = Descent(0.001)
# opt = Adam(0.01)

ps = Flux.params(ui)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, dim.x, ui[:, 1])
lines!(ax, dim.x, ui[:, 2])
save("ui_original.png", fig)

for i in 1:20

    e, back = pullback(ui) do _ui
        u = iter(_ui)
        return sum(u[:, 1, end] .^ 2)
    end

    gs = back(one(e))[1]
    Flux.Optimise.update!(opt, ui, gs)
    
    println(e)
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, dim.x, ui[:, 1])
lines!(ax, dim.x, ui[:, 2])
save("ui_opt.png", fig)

#=
Computing the energy gradient of the final wave with respect to the
initial wave.
=#

# @time loss, back = pullback(wave) do _wave
#     u = integrate(iter, _wave, 0.0f0, steps)
#     return sum(u[:, 1, end] .^ 2)
# end

# ## Gradient weighted by its loss value
# @time gs = back(loss)[1]

## Plotting the gradient of displacement and velocity seperately
# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "dL/du(0)")
# ax2 = Axis(fig[1, 2], title = "dL/dv(0)")
# lines!(ax1, dim.x, gs[:, 1], label = "Default")
# lines!(ax2, dim.x, gs[:, 2], label = "Default")
# save("gs.png", fig)


# ## Solving from initial wave at time 0.0
# u = integrate(iter, wave, 0.0f0, steps)
# ## Getting final condition of wave simulation
# uf = u[:, :, end]
# ## Computing final time
# tf = steps * dt

# ## Doing backwards in time integration
# iter_backward = Integrator(runge_kutta, dynamics, -dt) |> gpu
# u_backward = integrate(iter_backward, uf, tf, steps)
# @assert u ≈ reverse(u_backward, dims = 3)

# fig = Figure()
# ax1 = Axis(fig[1, 1])
# ax2 = Axis(fig[1, 2])
# heatmap!(ax1, u_backward[:, 1, :], colormap = :ice)
# heatmap!(ax2, u_backward[:, 2, :], colormap = :ice)
# save("u_backward.png", fig)

# loss, back = pullback(_u -> sum(_u[:, 1] .^ 2), uf)
# adj = back(loss)[1]

# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "dL/du(tf)")
# ax2 = Axis(fig[1, 2], title = "dL/dv(tf)")
# lines!(ax1, dim.x, adj[:, 1], label = "Default")
# lines!(ax2, dim.x, adj[:, 2], label = "Default")
# save("adj.png", fig)

# function adjoint_sensitivity(adj::AbstractMatrix{Float32}, iter::Integrator, u::AbstractMatrix{Float32}, tf::Float32, steps::Int)

#     ## create time vector containing all but the first time
#     tspan = build_tspan(tf, iter.dt, steps)[1:end-1]
#     ## create a deepcopy of the adjoint so that the original is not modified
#     adj = deepcopy(adj)

#     for t in tspan
#         ## step the value of u backward one timestep & compute gradient
#         u, back = pullback(_u -> iter(_u, t, iter.dt), u)
#         ## step the adjoint backward one timestep
#         adj = adj .-= back(adj)[1] * iter.dt
#     end

#     return adj
# end

# @time adj_t0 = adjoint_sensitivity(adj, iter_backward, uf, tf, steps)

# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "dL/du(ti)")
# ax2 = Axis(fig[1, 2], title = "dL/dv(ti)")
# lines!(ax1, dim.x, adj_t0[:, 1], label = "Default")
# lines!(ax2, dim.x, adj_t0[:, 2], label = "Default")
# save("adj_t0.png", fig)

# fig = Figure()
# ax1 = Axis(fig[1, 1])
# ax2 = Axis(fig[1, 2])
# lines!(ax1, dim.x, uf[:, 1])
# lines!(ax2, dim.x, uf[:, 2])
# save("uf.png", fig)