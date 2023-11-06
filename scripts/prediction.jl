using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(2)
display(Flux.device())
include("model_modifications.jl")

@time data = [Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode$i.bson") for i in 468:500]

data_loader_kwargs = Dict(:batchsize => 32, :shuffle => true, :partial => false)

pml_checkpoint = 2300
MODEL_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/trainable_pml_localization_horizon=20_batchsize=32_h_size=256_latent_gs=100.0_pml_width=10.0_nfreq=500/checkpoint_step=$pml_checkpoint/checkpoint.bson"
model = gpu(BSON.load(MODEL_PATH)[:model])

# horizon = collect(20:10:200)
horizon = collect(20:10:200)
pml_error = Vector{Float32}[]

for h in horizon
    val_loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)
    s, a, t, y = gpu(Flux.batch.(first(val_loader)))

    y_sc = y[:, 3, :]
    @time y_hat = model(s, a, t)[:, 3, :]
    error = cpu(vec(Flux.mse(y_sc, y_hat, agg = x -> Flux.mean(x, dims = 1))))
    push!(pml_error, error)

    BSON.bson("pml_error.bson", horizon = horizon, error = pml_error)
end

