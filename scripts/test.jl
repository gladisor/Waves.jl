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
        Dense(h_size, out_size)
        )
end

function build_hypernet_wave_encoder(;nfreq::Int, h_size::Int, n_h::Int, act::Function, ambient_speed::Float32, dim::OneDim)

    embedder = build_mlp(2 * nfreq, h_size, n_h, 3, act)

    ps, re = destructure(embedder)

    return Chain(
        SingleImageInput(),
        MaxPool((2, 2)),
        DownBlock(3, 1, 16, act),
        InstanceNorm(16),
        DownBlock(3, 16, 64, act),
        InstanceNorm(64),
        DownBlock(3, 64, 128, act),
        InstanceNorm(128),
        DownBlock(3, 128, 256, act),
        GlobalMeanPool(),
        flatten,
        NormalizedDense(256, 512, act),
        Dense(512, length(ps), bias = false),
        vec,
        re,
        FrequencyDomain(dim, nfreq),
        z -> hcat(tanh.(z[:, 1]), tanh.(z[:, 2]) / ambient_speed, tanh.(z[:, 3]))
    )
end

function build_hypernet_design_encoder(;nfreq, in_size, h_size, n_h, act, dim, speed_activation::Function)

    embedder = build_mlp(2 * nfreq, h_size, n_h, 1, act)

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

function LatentDynamics(dim::OneDim; ambient_speed, freq, pml_width, pml_scale)
    pml = build_pml(dim, pml_width, pml_scale)
    grad = build_gradient(dim)
    bc = dirichlet(dim)
    return LatentDynamics(ambient_speed, freq, pml, grad, bc)
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
        n_h = n_h, 
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
        # act,
        # Dense(h_size, 1)
        vec)

    return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
end

function compute_gradient(model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, sigma::AbstractMatrix{Float32}, loss_func::Function)
    loss, back = pullback(_model -> loss_func(_model(s, a), sigma), model)
    return (loss, back(one(loss))[1])
end

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
            loss, gs = compute_gradient(model, gpu(s[1]), gpu(a[1]), gpu(sigma[1]), loss_func)
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
            sigma = gpu(sigma[1])
            
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

Flux.device!(3)

elu_speed(c) = 1.0f0 .+ elu.(c)
sigmoid_speed(c, scale = 5.0f0) = 5.0f0 * sigmoid.(c)
exp_speed(c) = exp.(c)

nfreq = 2
lr = 1e-5
act = leakyrelu
speed_activation = exp_speed
pml_scale = 5000.0f0
horizon = 5
decay_rate = 0.98f0

data_path = "data/hexagon_large_grid"
@time states, actions, tspans, sigmas = prepare_data(EpisodeData(path = joinpath(data_path, "episode1/episode.bson")), horizon)
idx = 35
s = gpu(states[idx])
a = gpu(actions[idx])
design_input = vcat(vec(s.design), vec(a[1])) |> gpu

grid_size = 20.0f0
dim = OneDim(grid_size, 512)
@time model = build_hypernet_wave_control_model(
    dim,
    design_input_size = length(design_input),
    nfreq = nfreq,
    h_size = 512,
    n_h = nfreq,
    act = act,
    speed_activation = speed_activation,
    ambient_speed = AIR,
    freq = 200.0f0,
    pml_width = 5.0f0,
    pml_scale = pml_scale,
    dt = 5e-5,
    steps = 100) |> gpu

## Loading data into memory
println("Load Train Data")
@time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:150])
println("Load Val Data")
@time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 151:200])

## Establishing model path
model_path = mkpath(joinpath(data_path, "models/freq_model_fixed_long/lr=$(lr)_decay_rate=$(decay_rate)_nfreq=$(nfreq)_act=$(act)_speedact=$(speed_activation)_pml_scale=$(pml_scale)_grid_size=$(grid_size)_horizon=$(horizon)"))

train_loop(model,
    loss_func = mse,
    train_steps = 200,
    train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true),
    val_steps = 200,
    val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true),
    epochs = 1000,
    lr = lr,
    decay_rate = decay_rate,
    evaluation_samples = 10,
    checkpoint_every = 10,
    latent_dim = dim,
    path = model_path
    )