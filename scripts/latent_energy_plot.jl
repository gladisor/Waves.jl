println("Importing Packages")
using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves

Flux.CUDA.allowscalar(false)

Flux.device!(0)

main_path = "..."
pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=10000.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
no_pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=0.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Loading Models")
pml_model = gpu(BSON.load(pml_model_path)[:model])
no_pml_model = gpu(BSON.load(no_pml_model_path)[:model])
testmode!(pml_model)
testmode!(no_pml_model)

episode = EpisodeData(path = joinpath(data_path, "episode1/episode.bson"))
s, a, t, sigma = prepare_data(episode, length(episode))
s = gpu(s[1])
a = gpu(a[1])
t = gpu(t[1])
tspan = cpu(flatten_repeated_last_dim(t))
# sigma = gpu(sigma[1])

pml_z = cpu(generate_latent_solution(pml_model, s, a, t))
no_pml_z = cpu(generate_latent_solution(no_pml_model, s, a, t))

pml_incident_energy = vec(sum(pml_z[:, 1, :, 1] .^ 2, dims = 1))
no_pml_incident_energy = vec(sum(no_pml_z[:, 1, :, 1] .^ 2, dims = 1))

pml_total_energy = vec(sum(pml_z[:, 2, :, 1] .^ 2, dims = 1))
no_pml_total_energy = vec(sum(no_pml_z[:, 2, :, 1] .^ 2, dims = 1))

pml_scattered_energy = vec(sum((pml_z[:, 2, :, 1] .- pml_z[:, 1, :, 1]) .^ 2, dims = 1))
no_pml_scattered_energy = vec(sum((no_pml_z[:, 2, :, 1] .- no_pml_z[:, 1, :, 1]) .^ 2, dims = 1))

fig = Figure()
ax1 = Axis(fig[1, 1], ylabel = "Incident Energy")
lines!(ax1, tspan, pml_incident_energy, label = "PML")
lines!(ax1, tspan, no_pml_incident_energy, label = "No PML")

ax2 = Axis(fig[2, 1], ylabel = "Total Energy")
lines!(ax2, tspan, pml_total_energy, label = "PML")
lines!(ax2, tspan, no_pml_total_energy, label = "No PML")

ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "Scattered Energy")
lines!(ax3, tspan, pml_scattered_energy, label = "PML")
lines!(ax3, tspan, no_pml_scattered_energy, label = "No PML")

axislegend(ax1, position = :lt)
axislegend(ax2, position = :lt)
axislegend(ax3, position = :lt)
save("latent_energy.png", fig)