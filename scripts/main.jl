println("Importing Packages")
using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves

function optimize(model, s, a, t, y)

    opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(), Optimisers.Adam(lr))
    opt_state = Optimisers.setup(opt, model)

    for i in 1:100
    
        loss, back = Flux.pullback(model) do m
            return Flux.mse(y, m(s, a, t))
        end
    
        @time gs = back(one(loss))[1]
        opt_state, model = Optimisers.update(opt_state, model, gs)
    
        println(loss)
    end

    return model
end

Flux.CUDA.allowscalar(false)

Flux.device!(1)
main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_pulse_intensity=10.0_freq=1000.0/"
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Declaring Hyperparameters")
nfreq = 50
h_size = 256
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 512
horizon = 2
batchsize = 32

pml_width = 5.0f0
pml_scale = 10000.0f0
lr = 1e-4
decay_rate = 1.0f0
k_size = 2
steps = 20
epochs = 500
loss_func = Flux.mse

println("Initializing Model Components")
latent_dim = OneDim(latent_grid_size, latent_elements)

# wave_encoder = build_split_hypernet_wave_encoder(latent_dim = latent_dim, nfreq = nfreq, h_size = h_size, activation = activation)
include("new_model.jl")
wave_encoder = build_wave_encoder(;
    k = (3, 3), 
    in_channels = 3, 
    activation, h_size,
    nfields = 4,
    nfreq = nfreq,
    latent_dim = latent_dim
    )

design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)
mlp = build_scattered_wave_decoder(latent_elements, h_size, k_size, activation)

println("Constructing Model")
model = gpu(ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp))

println("Initializing DataLoaders")
@time begin
    train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:2])
    # val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 101:120])
    train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = -1, partial = false)
    # val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize, partial = false)
end

s, a, t, y = gpu(first(train_loader))
tspan = cpu(flatten_repeated_last_dim(t))

# model = optimize(model, s, a, t, y)

zi = model.wave_encoder(s)
c = ones(Float32, size(latent_dim)..., 1, 1)
dc = zeros(Float32, size(latent_dim)..., 1, 1)

# z = generate_latent_solution(model, s, a, t)

# ## plotting initial conditions
# fig = Figure()
# ax = Axis(fig[1, 1])
# for i in axes(zi, 2)
#     lines!(ax, latent_dim.x, zi[:, i, 1])
# end
# save("freq.png", fig)

# y_hat = model(s, a, t)

# ## plotting prediction of scattered energy versus real
# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, tspan, vec(cpu(y)), label = "true")
# lines!(ax, tspan, vec(cpu(y_hat)), label = "pred")
# axislegend(ax)
# save("signal.png", fig)

# z = generate_latent_solution(model, s, a, t)
# z_u_tot = cpu(z[:, 2, :, 1])'

# ## heatmap of latent solution
# fig = Figure()
# ax = Axis(fig[1, 1])
# heatmap!(ax, tspan, latent_dim.x, z_u_tot, colormap = :ice)
# save("z_tot.png", fig)

# ## rendering latent solution
# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)
# CairoMakie.record(fig, "latent.mp4", axes(tspan, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z_u_tot[i, :], color = :blue)
# end

# z = iter(x, tspan[:, :])
# tspan_matrix = tspan[:, :]

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)

# CairoMakie.record(fig, "z.mp4", axes(z, 4)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 3, 1, i], color = :blue)
#     lines!(ax, latent_dim.x, vec(cpu(dyn.source(tspan[:, :][i, :]))), color = :orange)
# end

# MODEL_PATH = mkpath(joinpath(main_path, "models/RERUN/latent_gs=$(latent_grid_size)_latent_elements=$(latent_elements)_horizon=$(horizon)_nfreq=$(nfreq)_pml=$(pml_scale)_lr=$(lr)_batchsize=$(batchsize)"))
# println(MODEL_PATH)
# println("Training")
# train_loop(
#     model,
#     loss_func = loss_func,
#     train_steps = steps,
#     val_steps = steps,
#     train_loader = train_loader,
#     val_loader = val_loader,
#     epochs = epochs,
#     lr = lr,
#     decay_rate = decay_rate,
#     evaluation_samples = 15,
#     checkpoint_every = 10,
#     path = MODEL_PATH,
#     opt = opt
#     )