using CairoMakie
using Flux
using Flux: mse
using Optimisers
using Waves

struct NonLinearWaveDynamics <: AbstractDynamics
    ambient_speed::Float32
    C::Vector{Float32}
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
    pml_scale::Float32
    pml::Vector{Float32}
end

Flux.@functor NonLinearWaveDynamics
Flux.trainable(dynamics::NonLinearWaveDynamics) = (;dynamics.C, dynamics.pml)

function (dyn::NonLinearWaveDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    σ = dyn.pml * dyn.pml_scale

    b = (dyn.ambient_speed * dyn.C) .^ 2
    du = b .* (dyn.grad * v) .* dyn.bc .- σ .* u
    dv = dyn.grad * u .- σ .* v
    return hcat(du, dv)
end

function train(model, opt_state, ui, y)

    for i in 1:20
        loss, gs = Flux.withgradient(model -> mse(model(ui), y), model)
        opt_state, model = Optimisers.update(opt_state, model, gs[1])
        println(loss)
    end

    return model
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

C = ones(Float32, size(dim)...)
grad = build_gradient(dim)
bc =  dirichlet(dim)
pml_scale = 70000.0f0
pml = build_pml(dim, 1.0f0, 1.0f0)

dynamics = NonLinearWaveDynamics(ambient_speed, C, grad, bc, pml_scale, pml)
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

model = Chain(
    input = Dense(elements, elements, relu),
    iter = Integrator(runge_kutta, dynamics, ti, dt, steps),
    flatten = Flux.flatten,
    h1 = Dense(2 * elements, 2 * elements, relu),
    h2 = Dense(2 * elements, 1),
    output = vec)

# model = Chain(
#     vec,
#     Dense(2 * elements, 2 * elements, relu),
#     Dense(2 * elements, 2 * elements, relu),
#     Dense(2 * elements, steps + 1))

opt_state = Optimisers.setup(Optimisers.Adam(5e-5), model)
y = sin.(2pi*range(0.0f0, 1.0f0, steps + 1))

model = train(model, opt_state, ui, y)

yhat = model(ui)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, y, label = "True")
lines!(ax, yhat, label = "Prediction")
save("evaluation.png", fig)

u = model[1:2](ui)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "C.mp4", axes(u, 3)) do i
    empty!(ax)
    lines!(ax, dim.x, u[:, 1, i], color = :blue)
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, model[:iter].dynamics.C)
save("C.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, model[:iter].dynamics.pml)
save("pml.png", fig)