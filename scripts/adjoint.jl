using CairoMakie
using Flux
using Flux: @adjoint, pullback, Recur, batch, unbatch, mean, params, trainable, withgradient
using Flux.Zygote: Params, Grads
using Flux.Losses: mse
using Flux.ChainRulesCore: Tangent, rrule, backing, construct, canonicalize
using Waves

include("../src/dynamics.jl")

struct NonLinearWaveDynamics <: AbstractDynamics
    ambient_speed::Float32
    C::Vector{Float32}
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
    pml_scale::Float32
    pml::Vector{Float32}
end

Flux.@functor NonLinearWaveDynamics
Flux.trainable(dynamics::NonLinearWaveDynamics) = (;dynamics.pml)

function (dyn::NonLinearWaveDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    σ = sigmoid.(dyn.pml) * dyn.pml_scale

    b = (dyn.ambient_speed * dyn.C) .^ 2
    du = b .* (dyn.grad * v) .- σ .* u #.* dyn.bc 
    dv = dyn.grad * u .- σ .* v
    return hcat(du, dv)
end

function Flux.ChainRulesCore.rrule(iter::Integrator, ui::AbstractMatrix{Float32})
    println("rrule Integrator")

    u = iter(ui)
    ps = Flux.params(iter.dynamics)
    # trainable_ps = trainable(iter.dynamics)

    function Integrator_back(adj::AbstractArray{Float32, 3})

        println("calling Integrator_back")
        ps_gs, u_gs = continuous_backprop(iter, u, adj, ps)
        # ps_gs = [ps_gs[trainable_ps[p]] for p in keys(trainable_ps)]
        # ps_gs = Dict(keys(trainable_ps) .=> ps_gs)
        # tangent = canonicalize(Tangent{Integrator}(;dynamics = Tangent{typeof(iter.dynamics)}(;ps_gs...)))
        # tangent = NonLinearWaveDynamics(values(ps_gs)...)
        # return tangent, u_gs
        return ps_gs, u_gs
    end

    return u, Integrator_back
end

grid_size = 10.f0
elements = 1024
ti = 0.0f0
dt = 0.00002f0
steps = 100

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

dim = OneDim(grid_size, elements)
pulse = Pulse(dim, 0.0f0, pulse_intensity)

ui = build_wave(dim, fields = 2)
ui = pulse(ui)

C = ones(Float32, size(dim)...)# * ambient_speed
grad = build_gradient(dim)
bc =  dirichlet(dim)
pml_scale = 70000.0f0
pml = build_pml(dim, 1.0f0, 1.0f0)

# dynamics = SplitWavePMLDynamics(nothing, dim, grid(dim), ambient_speed, grad, pml)
dynamics = NonLinearWaveDynamics(ambient_speed, C, grad, bc, pml_scale, pml)
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

u = iter(ui)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "vid.mp4", axes(u, 3)) do i
    empty!(ax)
    lines!(ax, dim.x, u[:, 1, i], color = :blue)
end

opt = Adam(1e-5)
opt2 = Adam(1e-1)

ps = params(iter)

for i in 1:20

    u = iter(ui)
    loss, loss_back = pullback(_u -> mean(sum(_u[:, 1, :] .^ 2, dims = 2)), u)
    adj = loss_back(1.0f0)[1]
    ps_gs, dui = continuous_backprop(iter, u, adj, ps)
    # Flux.Optimise.update!(opt, ui, dui)
    Flux.Optimise.update!(opt2, ps, ps_gs)
    display(loss)
end

u = iter(ui)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "vid.mp4", axes(u, 3)) do i
    empty!(ax)
    lines!(ax, dim.x, u[:, 1, i], color = :blue)
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, iter.dynamics.C)
save("C.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, iter.dynamics.pml)
save("pml.png", fig)

# # model = Chain(
# #     Dense(elements, elements, relu),
# #     Dense(elements, elements, relu),
# #     Integrator(runge_kutta, dynamics, ti, dt, steps),
# #     Flux.flatten,
# #     Dense(2 * elements, 1),
# #     vec)

# # display(model[1].weight)

# # # mlp = Chain(
# # #     vec,
# # #     Dense(2 * elements, 2 * elements, relu),
# # #     Dense(2 * elements, 2 * elements, relu),
# # #     Dense(2 * elements, steps + 1))

# # y = sin.(2pi*range(0.0f0, 1.0f0, steps + 1))
# # larger_x = range(-2.0f0, 2.0f0, 300)

# # opt = Adam(1e-5)
# # ps = Flux.params(model)

# # for i in 1:20
# #     loss, gs = Flux.withgradient(() -> mse(model(ui), y), ps)
# #     Flux.Optimise.update!(opt, ps, gs)
# #     println(loss)
# # end

# # yhat = model(ui)

# # fig = Figure()
# # ax = Axis(fig[1, 1])
# # lines!(ax, y, label = "True")
# # lines!(ax, yhat, label = "Prediction")
# # save("y.png", fig)

# # display(model[1].weight)
