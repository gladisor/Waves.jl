include("dependencies.jl")

struct NormalizedDense
    dense::Dense
    norm::LayerNorm
    act::Function
end

function NormalizedDense(in_size::Int, out_size::Int, act::Function)
    return NormalizedDense(Dense(in_size, out_size), LayerNorm(out_size), act)
end

Flux.@functor NormalizedDense

function (dense::NormalizedDense)(x)
    return x |> dense.dense |> dense.norm |> dense.act
end

function build_mlp(in_size, h_size, n_h, out_size, act)

    return Chain(
        NormalizedDense(in_size, h_size, act),
        [NormalizedDense(h_size, h_size, act) for _ in 1:n_h]...,
        # Dense(in_size, h_size, act),
        # [Dense(h_size, h_size, act) for _ in 1:n_h]...,
        Dense(h_size, out_size)
        )
end

function build_residual_block(k::Int, in_channels::Int, out_channels::Int, activation::Function)

    main = Chain(
        Conv((k, k), in_channels => out_channels, activation, pad = SamePad()),
        Conv((k, k), out_channels => out_channels, pad = SamePad()))
    
    skip = Chain(
        Conv((1, 1), in_channels => out_channels, pad = SamePad()))

    return Chain(
        Parallel(+, main, skip),
        activation,
        MaxPool((2, 2)),
        )
end

function build_hypernet_wave_encoder(;nfreq::Int, h_size::Int, act::Function, ambient_speed::Float32, dim::OneDim)
    
    ## parameterizes three functions: displacement, velocity, and force
    embedder = build_mlp(2 * nfreq, h_size, 2, 3, act)

    ps, re = destructure(embedder)

    return Chain(
        SingleImageInput(),
        MeanPool((4, 4)),
        # DownBlock(3, 1, 32, act),
        # DownBlock(3, 32, 64, act),
        # DownBlock(2, 64, 128, act),
        # DownBlock(2, 128, 256, act),
        build_residual_block(3, 1, 32, act),
        build_residual_block(3, 32, 64, act),
        build_residual_block(2, 64, 128, act),
        build_residual_block(2, 128, 256, act),
        GlobalMaxPool(),
        flatten,
        NormalizedDense(256, 512, act),
        Dense(512, length(ps), bias = false),
        vec,
        re,
        FrequencyDomain(dim, nfreq),
        # Flux.Scale([1.0f0, 1.0f0/ambient_speed, 1.0f0], true, tanh)
        z -> hcat(tanh.(z[:, 1]), tanh.(z[:, 2]) / ambient_speed, tanh.(z[:, 3]))
    )
end

function build_hypernet_design_encoder(;nfreq, in_size, h_size, n_h, act, dim, speed_activation::Function)
    
    ## parameterizes a single function (wave speed)
    embedder = build_mlp(2 * nfreq, h_size, 2, 1, act)
    ps, re = destructure(embedder)

    encoder = Chain(
        build_mlp(in_size, h_size, n_h, h_size, act),
        LayerNorm(h_size),
        act,
        Dense(h_size, length(ps), bias = false),
        re,
        FrequencyDomain(dim, nfreq),
        vec,
        speed_activation,
        )
end

function build_hypernet_wave_control_model(
        dim::OneDim; 
        design_input_size::Int,
        nfreq::Int,
        h_size::Int,
        n_h::Int, 
        act::Function, 
        speed_activation::Function,
        ambient_speed::Float32,
        freq::Float32,
        pml_width::Float32,
        pml_scale::Float32,
        dt::AbstractFloat,
        steps::Int
        )

    wave_encoder = build_hypernet_wave_encoder(
        nfreq = nfreq,
        h_size = h_size, 
        act = act,
        ambient_speed = ambient_speed,
        dim = dim)

    design_encoder = build_hypernet_design_encoder(
        nfreq = nfreq,
        in_size = design_input_size,
        h_size = h_size,
        n_h = n_h,
        act = act,
        dim = dim,
        speed_activation = speed_activation)

    dynamics = LatentDynamics(dim, 
        ambient_speed = ambient_speed, 
        freq = freq, 
        pml_width = pml_width,
        pml_scale = pml_scale)

    iter = Integrator(runge_kutta, dynamics, 0.0f0, dt, steps)
    mlp = Chain(
        flatten, 
        build_mlp(4 * size(dim, 1), h_size, n_h, 1, act),
        vec)

    return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
end

function compute_gradient(model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, sigma::AbstractMatrix{Float32}, loss_func::Function)
    loss, back = pullback(_model -> loss_func(_model(s, a), sigma), model)
    return (loss, back(one(loss))[1])
end

const SIGNAL_SCALE = 1.0f0

function train_loop(
        model::WaveControlModel;
        loss_func::Function,
        train_steps::Int, ## only really effects validation frequency
        train_loader::DataLoader,
        val_steps::Int,
        val_loader::DataLoader,
        epochs::Int,
        lr::AbstractFloat,
        decay_rate::Float32,
        latent_dim::OneDim,
        evaluation_samples::Int,
        checkpoint_every::Int,
        path::String
        )

    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, model)

    metrics = Dict(
        :train_loss => Vector{Float32}(),
        :val_loss => Vector{Float32}(),
    )

    for i in 1:epochs

        epoch_loss = Vector{Float32}()

        @showprogress for (j, (s, a, tspan, sigma)) in enumerate(train_loader)
            s = gpu(s[1])
            a = gpu(a[1])
            tspan = tspan[1]
            sigma = gpu(sigma[1]) * SIGNAL_SCALE

            loss, gs = compute_gradient(model, s, a, sigma, loss_func)
            opt_state, model = Optimisers.update(opt_state, model, gs)

            push!(epoch_loss, loss)
            if j == train_steps
                break
            end
        end

        opt_state = Optimisers.adjust(opt_state, lr * decay_rate ^ i)

        push!(metrics[:train_loss], sum(epoch_loss) / train_steps)

        epoch_loss = Vector{Float32}()
        epoch_path = mkpath(joinpath(path, "epoch_$i"))

        @showprogress for (j, (s, a, tspan, sigma)) in enumerate(val_loader)
            s = gpu(s[1])
            a = gpu(a[1])
            tspan = tspan[1]
            sigma = gpu(sigma[1]) * SIGNAL_SCALE
            
            loss = loss_func(model(s, a), sigma)
            push!(epoch_loss, loss)

            if j <= evaluation_samples
                latent_path = mkpath(joinpath(epoch_path, "latent_$j"))
                visualize!(model, latent_dim, s, a, tspan, sigma, path = latent_path)
            end

            if j == val_steps
                break
            end
        end

        push!(metrics[:val_loss], sum(epoch_loss) / val_steps)


        fig = Figure()
        ax = Axis(fig[1,1], title = "Loss History", xlabel = "Epoch", ylabel = "Loss Value")
        lines!(ax, metrics[:train_loss], color = :blue, label = "Train")
        lines!(ax, metrics[:val_loss], color = :orange, label = "Val")
        axislegend(ax)
        save(joinpath(epoch_path, "loss.png"), fig)

        train_loss = metrics[:train_loss][end]
        val_loss = metrics[:val_loss][end]
        println("Epoch: $i, Train Loss: $train_loss, Val Loss: $val_loss")

        if i % checkpoint_every == 0 || i == epochs
            checkpoint_path = mkpath(joinpath(epoch_path, "model"))
            println("Checkpointing")
            save(model, checkpoint_path)
        end
    end
end

function overfit!(model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix, dim::OneDim, lr)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, model)

    for i in 1:200
        loss, gs = compute_gradient(model, s, a, sigma, Flux.mae)
        opt_state, model = Optimisers.update(opt_state, model, gs)
        println(loss)
    end

    @time visualize!(model, dim, s, a, tspan, sigma, path = "")
end

########################################################################################

# fig = Figure()

# ax1 = Axis(fig[1, 1], aspect = 1.0f0)
# ax2 = Axis(fig[1, 2], aspect = 1.0f0)
# ax3 = Axis(fig[1, 3], aspect = 1.0f0)

# tot = cpu(s.wave_total)[:, :, 1]
# inc = cpu(s.wave_incident)[:, :, 1]
# sc = tot .- inc

# heatmap!(ax1, dim.x, dim.y, tot, colormap = :ice)
# heatmap!(ax2, dim.x, dim.y, inc, colormap = :ice)
# heatmap!(ax3, dim.x, dim.y, sc, colormap = :ice)
# save("wave.png", fig)

# ########################################################################################

Flux.device!(0)

elu_speed(c) = 1.0f0 .+ elu.(c)
sigmoid_speed(c, scale = 2.0f0) = scale * sigmoid.(c)
exp_speed(c) = exp.(c)

nfreq = 6
n_h = 3
lr = 1e-6
act = relu
speed_activation = sigmoid #exp_speed
pml_scale = 0.0f0
horizon = 1
decay_rate = 0.98f0

data_path = "data/ambient_speed=water/dt=1.0e-5/single_cylinder_cloak"
@time states, actions, tspans, sigmas = prepare_data(EpisodeData(path = joinpath(data_path, "episodes/episode1/episode.bson")), horizon)
idx = 100
s = gpu(states[idx])
a = gpu(actions[idx])
design_input = vcat(vec(s.design), vec(a[1])) |> gpu

grid_size = 15.0f0
dim = OneDim(grid_size, 512)
@time model = build_hypernet_wave_control_model(
    dim,
    design_input_size = length(design_input),
    nfreq = nfreq,
    h_size = 512,
    n_h = n_h,
    act = act,
    speed_activation = speed_activation,
    ambient_speed = WATER,
    freq = 2000.0f0,
    pml_width = 5.0f0,
    pml_scale = pml_scale,
    dt = 1e-5,
    steps = 100) |> gpu

tspan = tspans[idx]
sigma = gpu(sigmas[idx])

# @time visualize!(model, dim, s, a, tspan, sigma, path = "")
@time overfit!(model, s, a, tspan, sigma, dim, lr)

# # ## Loading data into memory
# # println("Load Train Data")
# # @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episodes/episode$i/episode.bson")) for i in 1:45])
# # println("Load Val Data")
# # @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episodes/episode$i/episode.bson")) for i in 46:55])

# # # ## Establishing model path
# # model_path = mkpath(joinpath(data_path, "models/last_set/lr=$(lr)_decay_rate=$(decay_rate)_nfreq=$(nfreq)_act=$(act)_speedact=$(speed_activation)_pml_scale=$(pml_scale)_grid_size=$(grid_size)_horizon=$(horizon)"))

# # train_loop(model,
# #     loss_func = mse,
# #     train_steps = 200,
# #     train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true),
# #     val_steps = 200,
# #     val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true),
# #     epochs = 1000,
# #     lr = lr,
# #     decay_rate = decay_rate,
# #     evaluation_samples = 10,
# #     checkpoint_every = 100,
# #     latent_dim = dim,
# #     path = model_path
# #     )