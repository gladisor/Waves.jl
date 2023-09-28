using Waves, CairoMakie, Flux, BSON
using Optimisers
Flux.CUDA.allowscalar(false)
Flux.device!(0)
display(Flux.device())

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)"

ep = Episode(path = joinpath(DATA_PATH, "episode1.bson"))
# Waves.visualize(ep, path = "ep.png")
s, a, t, y = Flux.batch.(prepare_data(ep, 100))

model = gpu(BSON.load("tranable_source/checkpoint_step=1900/checkpoint.bson")[:model])
# z = cpu(Waves.generate_latent_solution(model, gpu(s[1, :]), gpu(a[:, [1]]), gpu(t[:, [1]])))
@time y_hat = cpu(model(gpu(s[1, :]), gpu(a[:, [1]]), gpu(t[:, [1]])))
tspan = cpu(t[:, 1])

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, tspan, y_hat[:, 1])
lines!(ax, tspan, y_hat[:, 2])
lines!(ax, tspan, y_hat[:, 3])
save("latent_energy1.png", fig)

# fig = Figure()
# ax = Axis(fig[1, 1])
# dim = cpu(model.iter.dynamics.dim)
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)

# record(fig, "latent.mp4", axes(tspan, 1)) do i
#     empty!(ax)
#     lines!(ax, dim.x, z[:, 1, 1, i], color = :blue)
# end