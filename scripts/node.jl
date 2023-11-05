using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(2)
display(Flux.device())

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")
include("dataset.jl")
include("../src/model/node.jl")

DATA_PATH = "../AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/"
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:10]

h_size = 256
activation = leakyrelu
nfreq = 500 #128
elements = 1024
horizon = 20
batchsize = 32
val_every = 20
val_batches = val_every
epochs = 10
latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0
train_val_split = 0.90 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)

## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]

## preparing DataLoader(s)
data_func = get_energy_data
train_loader = Flux.DataLoader(EpisodeDataset(train_data, horizon, data_func); data_loader_kwargs...)
val_loader = Flux.DataLoader(EpisodeDataset(val_data, horizon, data_func); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

latent_dim = OneDim(latent_gs, elements)
wave_encoder = WaveEncoder(
    build_cnn_base(env, 3, activation, h_size),
    Chain(Dense(h_size, elements)))

design_encoder = DesignEncoder(env, h_size, activation, nfreq, latent_dim)

# mlp = MLP(
#     Dense(2 * elements, h_size, activation), 
#     Dense(h_size, elements, activation))

mlp = Chain(
    Dense(2 * elements, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, elements)
)

params, re = Flux.destructure(mlp)

dyn = NODEDynamics(re)
iter = Integrator(runge_kutta, dyn, env.dt)
model = gpu(NODEEnergyModel(wave_encoder, design_encoder, iter, params, get_dx(latent_dim)))
loss_func = NODEEnergyLoss()

s, a, t, y = gpu(first(val_loader))

@time begin
    loss, back = Flux.pullback(model) do m
        loss_func(m, s, a, t, y)
    end

    gs = back(one(loss))[1]
end;
;