using Waves, CairoMakie, Flux, BSON
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

ep = Episode(path = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode500.bson")
model = gpu(BSON.load("/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/latent_gs=100.0_pml_width=10.0_nfreq=500/checkpoint_step=3100/checkpoint.bson")[:model])

s, a, t, y = prepare_data(ep, 10)
idx = 50
s, a, t, y = gpu(Flux.batch.((s[[idx]], a[[idx]], t[[idx]], y[[idx]])))

@time y_hat = cpu(model(s, a, t))
y = cpu(y)

tspan = cpu(vec(t))
fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, tspan, y[:, 1], color = :blue)
lines!(ax1, tspan, y_hat[:, 1], color = :orange)

ax2 = Axis(fig[2, 1])
lines!(ax2, tspan, y[:, 2], color = :blue)
lines!(ax2, tspan, y_hat[:, 2], color = :orange)

ax3 = Axis(fig[3, 1])
lines!(ax3, tspan, y[:, 3], color = :blue)
lines!(ax3, tspan, y_hat[:, 3], color = :orange)
save("pred.png", fig)

# fig = Figure()
# ax = Axis(fig[1, 1], aspect = 1.0f0)
# heatmap!(ax, ep.s[end].wave[:, :, end], colormap = :ice)
# save("wave.png", fig)