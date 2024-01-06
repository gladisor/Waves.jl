using Waves, CairoMakie, Flux, Optimisers, BSON
using ForwardDiff, ReverseDiff, FiniteDiff
using FiniteDifferences

Flux.CUDA.allowscalar(false)
Flux.device!(0)
display(Flux.device())
include("random_pos_gaussian_source.jl")

function Waves.DesignEncoder(env::WaveEnv, h_size::Int, activation::Function, nfreq::Int, latent_dim::OneDim)
    mlp = Chain(
        Dense(length(vec(env.design)), h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation), 
        Dense(h_size, nfreq),
        SinWaveEmbedder(latent_dim, nfreq),
        c -> 2.0f0 * sigmoid.(c))

    design_encoder = DesignEncoder(env.design_space, mlp, env.integration_steps)
end


function control_pinn()
    ## loading environment and data
    dataset_name = "variable_source_yaxis_x=-10.0"
    DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
    @time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
    @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:10]

    h_size = 256
    activation = leakyrelu
    latent_gs = 100.0f0
    elements = 1024
    horizon = 1
    batchsize = 2 ## shorter horizons can use large batchsize
    train_val_split = 0.90 ## choosing percentage of data for val
    data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)
    ## spliting data
    idx = Int(round(length(data) * train_val_split))
    train_data, val_data = data[1:idx], data[idx+1:end]

    ## preparing DataLoader(s)
    train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
    val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
    println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")
    s, a, t, y = gpu(Flux.batch.(first(train_loader)))

    latent_dim = OneDim(latent_gs, elements)
    W = gpu(WaveEncoder(env, 4, h_size, activation, 500, latent_dim))
    D = gpu(DesignEncoder(env, h_size, activation, 500, latent_dim))
    R = gpu(Chain(
        Conv((2,), 8 => 64, activation, pad = SamePad()),
        Conv((2,), 64 => 64, activation, pad = SamePad()),
        MaxPool((2,)),
        Conv((2,), 64 => 128, activation, pad = SamePad()),
        Conv((2,), 128 => 128, activation, pad = SamePad()),
        MaxPool((2,)),
        Conv((2,), 128 => 128, activation, pad = SamePad()),
        Conv((2,), 128 => 256, activation, pad = SamePad()),
        Conv((2,), 256 => 256),
        GlobalMaxPool(),
        l -> permutedims(l, (2, 1, 3))
        ))

    U = gpu(Chain(
        Dense(256 + 2, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, 2),
    ))

    ## preparing discretization grid
    tspan = build_tspan(env)
    grad_t = gpu(Waves.gradient(tspan))
    grad_x = gpu(Waves.gradient(latent_dim.x))
    t_grid = repeat(reshape(tspan, 1, 1, length(tspan)), 1, size(latent_dim.x, 1), 1) / (env.dt * env.integration_steps)
    x_grid = repeat(reshape(latent_dim.x, 1, elements, 1), 1, 1, length(tspan)) / latent_gs
    pinn_grid = gpu(vcat(x_grid, t_grid))
    pinn_grid_flat = reshape(pinn_grid, 2, :, 1)
    grad_x = gpu(Matrix{Float32}(cpu(grad_x)))
    grad_t = gpu(Matrix{Float32}(cpu(grad_t)))
    dx = get_dx(latent_dim)

    opt_state = Optimisers.setup(Optimisers.Adam(1e-3), (W, D, R, U))

    for i in 1:2000
        loss, back = Flux.pullback(W, D, R, U) do _W, _D, _R, _U
            ## evaluate wave encoder
            x = _W(s)
            z = x[:, 1:4, :]
            ## expand force oscilation over time period
            F = Source(x[:, 5, :], env.source.freq)
            f = cat([F(t[i, :]) for i in axes(t, 1)]..., dims = 3)
            f = permutedims(f, (1, 3, 2))
            ## add dim to pml
            pml = permutedims(x[:, 6, :, :], (1, 3, 2))
            ## unroll wavespeed interpolation
            C = _D(s, a, t)
            c = cat([C(t[i, :]) for i in axes(t, 1)]..., dims = 3)
            c = permutedims(c, (1, 3, 2))
            # ## get wavespeed at beginning and end of time period
            c1 = Flux.unsqueeze(C(t[1, :]), 2)
            c2 = Flux.unsqueeze(C(t[end, :]), 2)
            # ## compress info into latent vector
            l = _R(hcat(x, c1, c2))
            l = repeat(l, 1, size(pinn_grid_flat, 2), 1)
            ## form input to pinn
            input = vcat(l, repeat(pinn_grid_flat, 1, 1, size(l, 3)))
            ## evaluate pinn and reshape solution
            sol = reshape(_U(input), 2, elements, length(tspan), size(l, 3))

            ## extract solution fields
            u = sol[1, :, :, :]
            v = sol[2, :, :, :]

            energy_loss = Flux.mse(
                dropdims(sum(u .^ 2, dims = 1) * dx, dims = 1),
                y[:, 3, :]
                )

            ## compute physics losses
            u_t = batched_transpose(batched_mul(grad_t, batched_transpose(u)))
            N_u = WATER * c .* batched_mul(grad_x, v) .- pml .* u
            v_t = batched_transpose(batched_mul(grad_t, batched_transpose(v)))
            N_v = WATER * c .* batched_mul(grad_x, u .+ f) .- pml .* v
            return Flux.mse(u_t, N_u) + Flux.mse(v_t, N_v) + energy_loss
        end

        println(loss)
        gs = back(one(loss))
        opt_state, (W, D, R, U) = Optimisers.update(opt_state, (W, D, R, U), gs)
    end


    ## evaluate wave encoder
    x = W(s)
    z = x[:, 1:4, :]
    ## expand force oscilation over time period
    F = Source(x[:, 5, :], env.source.freq)
    f = cat([F(t[i, :]) for i in axes(t, 1)]..., dims = 3)
    f = permutedims(f, (1, 3, 2))
    ## add dim to pml
    pml = permutedims(x[:, 6, :, :], (1, 3, 2))
    ## unroll wavespeed interpolation
    C = D(s, a, t)
    c = cat([C(t[i, :]) for i in axes(t, 1)]..., dims = 3)
    c = permutedims(c, (1, 3, 2))
    # ## get wavespeed at beginning and end of time period
    c1 = Flux.unsqueeze(C(t[1, :]), 2)
    c2 = Flux.unsqueeze(C(t[end, :]), 2)
    # ## compress info into latent vector
    l = R(hcat(x, c1, c2))
    l = repeat(l, 1, size(pinn_grid_flat, 2), 1)
    ## form input to pinn
    input = vcat(l, repeat(pinn_grid_flat, 1, 1, size(l, 3)))
    ## evaluate pinn and reshape solution
    sol = reshape(U(input), 2, elements, length(tspan), size(l, 3))

    ## extract solution fields
    u = cpu(sol[1, :, :, :])
    v = cpu(sol[2, :, :, :])


    fig = Figure()
    ax = Axis(fig[1, 1])
    heatmap!(ax, u[:, :, 1], colormap = :ice)
    ax = Axis(fig[1, 2])
    heatmap!(ax, v[:, :, 1], colormap = :ice)
    save("pinn.png", fig)
end

control_pinn()

# function main()
#     activation = leakyrelu
#     h_size = 256
#     in_channels = 4
#     nfreq = 500
#     elements = 1024
#     horizon = 1
#     lr = 1f-3 # 1f-4 for initial stage
#     batchsize = 32 ## shorter horizons can use large batchsize
#     val_every = 20
#     val_batches = val_every
#     epochs = 10
#     latent_gs = 20.0f0 #100.0f0
#     pml_width = 10.0f0
#     pml_scale = 10000.0f0
#     train_val_split = 0.90 ## choosing percentage of data for val
#     data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)

#     # dataset_name = "variable_source_yaxis_x=-10.0"
#     dataset_name = "part2_variable_source_yaxis_x=-10.0"
#     DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"

#     ## loading environment and data
#     @time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
#     @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:5]

#     ## spliting data
#     idx = Int(round(length(data) * train_val_split))
#     train_data, val_data = data[1:idx], data[idx+1:end]

#     ## preparing DataLoader(s)
#     train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
#     val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
#     println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")
#     s, a, t, y = gpu(Flux.batch.(first(train_loader)))

#     latent_dim = OneDim(latent_gs, elements)
#     wave_encoder = gpu(Waves.WaveEncoder(env, in_channels, h_size, activation, nfreq, latent_dim))

#     tspan = build_tspan(env)
#     x = latent_dim.x

#     t_grid = repeat(reshape(tspan, 1, 1, length(tspan)), 1, size(x, 1), 1) / (env.dt * env.integration_steps)
#     x_grid = repeat(reshape(x, 1, elements, 1), 1, 1, length(tspan)) / latent_gs
#     pinn_grid = gpu(vcat(x_grid, t_grid))
#     pinn_grid_flat = reshape(pinn_grid, 2, :, 1)

#     z0 = wave_encoder(s)
#     # f = z0[:, 5, 1]
#     f = gpu(build_normal(latent_dim.x, [0.0f0, 5.0f0], [0.3f0, 0.3f0], [1.0f0, -1.0f0]))

#     grad_t = gpu(Waves.gradient(tspan))
#     grad_x = gpu(Waves.gradient(x))
#     tspan = gpu(tspan)

#     pinn = gpu(Chain(
#         Dense(2, h_size, activation),
#         Dense(h_size, h_size, activation),
#         Dense(h_size, h_size, activation),
#         Dense(h_size, h_size, activation),
#         Dense(h_size, h_size, activation),
#         Dense(h_size, h_size, activation),
#         Dense(h_size, h_size, activation),
#         Dense(h_size, 2),
#     ))

#     u_prev = cpu(reshape(pinn(pinn_grid_flat), 2, 1024, 101))

#     opt_state = Optimisers.setup(Optimisers.Adam(1e-4), pinn)

#     # u = reshape(pinn(pinn_grid_flat), 2, 1024, 101)
#     # u_tot = u[1, :, :]

#     for i in 1:2000

#         loss, back = Flux.pullback(pinn) do _pinn
#             u = reshape(_pinn(pinn_grid_flat), 2, 1024, 101)
#             u_tot = u[1, :, :]
#             v_tot = u[2, :, :]

#             force = f .* sin.(2000.0f0 * 2.0f0 * pi * tspan)'
#             N_u = WATER * grad_x * (v_tot)
#             N_v = WATER * grad_x * (u_tot .+ force)
            
#             physics_loss = Flux.mean((((grad_t * u_tot')' .- N_u) .+ ((grad_t * v_tot')' .- N_v)) .^ 2)
#             energy_loss = Flux.mean((y[:, 1, 1] .- vec(sum(u_tot .^ 2, dims = 1))) .^ 2)
#             boundary_loss = sum(u_tot[[1, end], :] .^ 2)

#             physics_loss + energy_loss + boundary_loss
#         end
        
#         println(loss)
#         gs = back(one(loss))[1]
#         opt_state, pinn = Optimisers.update(opt_state, pinn, gs)
#     end

#     u_opt = cpu(reshape(pinn(pinn_grid_flat), 2, 1024, 101))

#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     heatmap!(ax, u_prev[1, :, :], colormap = :ice)
#     ax = Axis(fig[1, 2])
#     heatmap!(ax, u_prev[2, :, :], colormap = :ice)

#     ax = Axis(fig[2, 1])
#     heatmap!(ax, u_opt[1, :, :], colormap = :ice)
#     ax = Axis(fig[2, 2])
#     heatmap!(ax, u_opt[2, :, :], colormap = :ice)
#     save("pinn.png", fig)
# end

# main()