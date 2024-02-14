using Waves, Flux, CairoMakie, BSON
using Optimisers

Flux.device!(0)
Flux.CUDA.allowscalar(false)
display(Flux.device())

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
## declaring hyperparameters
activation = relu
h_size = 256
in_channels = 4
nfreq = 500
elements = 1024
horizon = 1
lr = 1f-4
batchsize = 4 #32 ## shorter horizons can use large batchsize
val_every = 20
val_batches = val_every
epochs = 10
latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0
train_val_split = 0.90 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)
latent_dim = OneDim(latent_gs, elements)
dx = get_dx(latent_dim)

## loading environment and data
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:5]

## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]

## preparing DataLoader(s)
train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

s, a, t, y = gpu(Flux.batch.(first(train_loader)))
model = gpu(WaveControlPINN(;env, activation, h_size, nfreq, latent_dim))
loss_func = gpu(WaveControlPINNLoss(env, latent_dim))

opt_state = Optimisers.setup(Optimisers.Adam(5f-4), model)

for i in 1:2000
    loss, back = Flux.pullback(m -> loss_func(m, s, a, t, y), model)
    println("$i, Loss: $loss")
    gs = back(one(loss))[1]
    opt_state, model = Optimisers.update(opt_state, model, gs)
end

# pinn_sol = generate_latent_solution(model, s, a, t)
# y_hat = compute_latent_energy(pinn_sol, get_dx(model.latent_dim))
# z = model.W(s)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(z[:, 5, 1]), label = "Force")
# ax = Axis(fig[2, 1])
# lines!(ax, cpu(z[:, 6, 1]), label = "PML")
# save("parameters.png", fig)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(y[:, 1, 1]))
# lines!(ax, cpu(y_hat[:, 1, 1]))
# ax = Axis(fig[2, 1])
# lines!(ax, cpu(y[:, 2, 1]))
# lines!(ax, cpu(y_hat[:, 2, 1]))
# ax = Axis(fig[3, 1])
# lines!(ax, cpu(y[:, 3, 1]))
# lines!(ax, cpu(y_hat[:, 3, 1]))
# save("energy.png", fig)

# pinn_tot = pinn_sol[:, 1, 1, :]
# pinn_inc = pinn_sol[:, 3, 1, :]
# pinn_sc = pinn_tot .- pinn_inc

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)
# record(fig, "vid.mp4", axes(tspan, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, cpu(pinn_tot[:, i]), color = :blue)
#     lines!(ax, latent_dim.x, cpu(pinn_inc[:, i]), color = :orange)
#     lines!(ax, latent_dim.x, cpu(pinn_sc[:, i]), color = :green)
# end

make_plots(model, (s, a, t, y), path = "")
