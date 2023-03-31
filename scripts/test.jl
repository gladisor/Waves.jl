using Flux
using Waves

include("models.jl")

path = "data/elements=256_speed=2.0_design_steps=20"
# data = load_episode_data.(readdir(joinpath(path, "episode1"), join = true))
# s, a = first(data)

files = readdir(joinpath(path, "episode1"), join = true)
paths = files[occursin.(".jld2", files)]

for p in paths
    println(p)
    data = load_episode_data(p)
end

# data = load_episode_data.(files[occursin.(".jld2", files)])

# dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

# fields = 6
# elements = 256
# h_fields = 64
# z_fields = 2
# h_size = 1024
# design_size = 2 * length(vec(s.design))
# activation = tanh

# total_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
# design_encoder = DesignEncoder(design_size, h_size, elements, activation)
# total_iter = FEMIntegrator(elements, 100; grid_size = 4.0f0, dynamics_kwargs...)
# incident_encoder = WaveEncoder(fields, h_fields, z_fields + 1, activation)
# incident_iter = FEMIntegrator(elements, 100; grid_size = 4.0f0, dynamics_kwargs...)
# incident_mlp = Chain(Dense(256, 256), z -> hcat(tanh.(z[:, [1, 2]]), sigmoid(z[:, 3])))

# mlp = MLP(3 * elements, h_size, 2, 1, activation)

# model = LatentSigmaSeparationModel(total_encoder, incident_encoder, design_encoder, total_iter, incident_iter, incident_mlp, mlp)
# model(s, a)