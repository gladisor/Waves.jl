using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")
include("dataset.jl")

DATA_PATH = "../AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/"
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:10]


## spliting data
train_val_split = 0.90 ## choosing percentage of data for val
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]

## preparing DataLoader(s)
batchsize = 32
horizon = 20
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)

data_func = get_rnn_energy_data
train_loader = Flux.DataLoader(EpisodeDataset(train_data, horizon, data_func); data_loader_kwargs...)
val_loader = Flux.DataLoader(EpisodeDataset(val_data, horizon, data_func); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

activation = leakyrelu
wave_encoder = gpu(build_cnn_base(env, 3, activation, 256))
s, a, t, y = gpu(first(val_loader))

z = cat([wave_encoder(s[i, :]) for i in axes(s, 1)]..., dims = 3)

model = gpu(Chain(
    GRU(256 => 256),
    GRU(256 => 1024),
    _z -> reshape(_z, (16, 16, 4, size(z, 2), size(z, 3)))
    ))

decoder = gpu(Chain(
    Conv((3, 3), 4 => 64, activation, pad = SamePad()),
    Conv((3, 3), 64 => 128, activation, pad = SamePad()),
    Upsample((2, 2)),

    Conv((3, 3), 128 => 128, activation, pad = SamePad()),
    Conv((3, 3), 128 => 128, activation, pad = SamePad()),
    Upsample((2, 2)),

    Conv((3, 3), 128 => 128, activation, pad = SamePad()),
    Conv((3, 3), 128 => 128, activation, pad = SamePad()),
    Upsample((2, 2)),

    Conv((3, 3), 128 => 64, activation, pad = SamePad()),
    Conv((3, 3), 64 => 64, activation, pad = SamePad()),
    Conv((1, 1), 64 => 3, w -> 3.0f0 * tanh.(w))))

# model(z)
decoder(model(z)[:, :, :, 1, [1]])