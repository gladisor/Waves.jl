using CairoMakie
using Flux
using Flux.Losses: mse
using Waves

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

dynamics = LinearWaveDynamics(ambient_speed, grad, dirichlet(dim))
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

model = Chain(
    Integrator(runge_kutta, dynamics, ti, dt, steps),
    Flux.flatten,
    Dense(2 * elements, 2 * elements, relu),
    Dense(2 * elements, 2 * elements, relu),
    Dense(2 * elements, 1),
    vec)

mlp = Chain(
    vec,
    Dense(2 * elements, 2 * elements, relu),
    Dense(2 * elements, 2 * elements, relu),
    Dense(2 * elements, steps + 1))

y = sin.(2pi*range(0.0f0, 1.0f0, steps + 1))
larger_x = range(-2.0f0, 2.0f0, 300)

opt = Adam(1e-5)
ps = Flux.params(model)

for i in 1:70
    loss, gs = Flux.withgradient(() -> mse(model(ui), y), ps)
    Flux.Optimise.update!(opt, ps, gs)
    println(loss)
end

yhat = model(ui)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, y, label = "True")
lines!(ax, yhat, label = "Prediction")
save("y.png", fig)