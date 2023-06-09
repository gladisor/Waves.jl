const EPSILON = Float32(1e-3)

abstract type WaveInputLayer end
(input::WaveInputLayer)(s::Vector{WaveEnvState}) = cat(input.(s)..., dims = 4)

struct TotalWaveInput <: WaveInputLayer end
Flux.@functor TotalWaveInput
(input::TotalWaveInput)(s::WaveEnvState) = s.wave_total[:, :, :, :]

struct IncidentWaveInput <: WaveInputLayer end
Flux.@functor IncidentWaveInput
(input::IncidentWaveInput)(s::WaveEnvState) = s.wave_incident[:, :, :, :]

struct ScatteredWaveInput <: WaveInputLayer end
Flux.@functor ScatteredWaveInput
(input::ScatteredWaveInput)(s::WaveEnvState) = s.wave_total[:, :, :, :] .- s.wave_incident[:, :, :, :]

struct ResidualBlock
    main::Chain
    skip::Conv
    activation::Function
    pool::MaxPool
end

Flux.@functor ResidualBlock

function ResidualBlock(k::Tuple{Int, Int}, in_channels::Int, out_channels::Int, activation::Function)
    main = Chain(
        Conv(k, in_channels => out_channels, activation, pad = SamePad()),
        Conv(k, out_channels => out_channels, pad = SamePad())
    )

    skip = Conv((1, 1), in_channels => out_channels, pad = SamePad())

    return ResidualBlock(main, skip, activation, MaxPool((2, 2)))
end

function (block::ResidualBlock)(x::AbstractArray{Float32})
    return (block.main(x) .+ block.skip(x)) |> block.activation |> block.pool
end

struct NormalizedDense
    dense::Dense
    norm::LayerNorm
    act::Function
end

function NormalizedDense(in_size::Int, out_size::Int, act::Function; kwargs...)
    return NormalizedDense(Dense(in_size, out_size; kwargs...), LayerNorm(out_size), act)
end

Flux.@functor NormalizedDense

function (dense::NormalizedDense)(x)
    return x |> dense.dense |> dense.norm |> dense.act
end

struct FrequencyDomain
    domain::AbstractMatrix{Float32}
end

Flux.@functor FrequencyDomain
Flux.trainable(::FrequencyDomain) = (;)

function FrequencyDomain(dim::OneDim, nfreq::Int)
    dim = cpu(dim)
    L = dim.x[end] - dim.x[1]
    frequencies = (Float32.(collect(1:nfreq)) .* dim.x') / L
    domain = vcat(sin.(2.0f0 * pi * frequencies), cos.(2.0f0 * pi * frequencies))
    return FrequencyDomain(domain)
end

function (freq::FrequencyDomain)(m)
    return m(freq.domain)
end

struct Hypernet
    dense::Dense
    re::Optimisers.Restructure
    domain::FrequencyDomain
end

Flux.@functor Hypernet

function Hypernet(in_size::Int, base::Chain, domain::FrequencyDomain)
    ps, re = destructure(base)
    dense = Dense(in_size, length(ps), bias = false)
    return Hypernet(dense, re, domain)
end

function (hypernet::Hypernet)(x::AbstractVector{Float32})
    m = hypernet.re(hypernet.dense(x))
    return hypernet.domain(m)
end

function (hypernet::Hypernet)(x::AbstractMatrix{Float32})
    models = hypernet.re.(eachcol(hypernet.dense(x)))
    return  cat([hypernet.domain(m) for m in models]..., dims = ndims(x) + 1)
end

function build_mlp(in_size::Int, h_size::Int, n_h::Int, activation::Function)

    return Chain(
        NormalizedDense(in_size, h_size, activation),
        [NormalizedDense(h_size, h_size, activation) for _ in 1:n_h]...)
end

function build_hypernet_wave_encoder(;
        latent_dim::OneDim,
        nfreq::Int,
        h_size::Int,
        activation::Function,
        input_layer::WaveInputLayer,
        )

    embedder = Chain(
        Dense(2 * nfreq, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, 3, tanh)) ## unbounded wavespeed doesnt work

    ps, re = destructure(embedder)

    model = Chain(
        input_layer,
        MaxPool((4, 4)),
        ResidualBlock((3, 3), 1, 32, activation),
        ResidualBlock((3, 3), 32, 64, activation),
        ResidualBlock((3, 3), 64, 128, activation),
        GlobalMaxPool(),
        flatten,
        NormalizedDense(128, h_size, activation),
        Hypernet(h_size, embedder, FrequencyDomain(latent_dim, nfreq)),
        Scale([1.0f0, 1.0f0/WATER, 1.0f0], false),
        z -> permutedims(z, (2, 1, 3))
        )

    return model
end

struct HypernetDesignEncoder
    design_space::DesignSpace
    action_space::DesignSpace
    layers::Chain
end

Flux.@functor HypernetDesignEncoder

function HypernetDesignEncoder(
        design_space::DesignSpace,
        action_space::DesignSpace,
        nfreq::Int,
        h_size::Int,
        activation::Function,
        latent_dim::OneDim)

    embedder = Chain(
        Dense(2 * nfreq, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, 1, sigmoid)
        )

    ps, re = destructure(embedder)

    ## static wavespeed
    # in_size = length(vec(design_space.low)) + length(vec(action_space.low))
    
    ## transient wavespeed
    in_size = length(vec(design_space.low))

    layers = Chain(
        NormalizedDense(in_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        Hypernet(h_size, embedder, FrequencyDomain(latent_dim, nfreq)),
        c -> permutedims(c, (2, 1))
        )

    return HypernetDesignEncoder(design_space, action_space, layers)
end

function normalize(design::AbstractDesign, ds::DesignSpace)
    return (vec(design) .- vec(ds.low)) ./ (vec(ds.high) .- vec(ds.low) .+ EPSILON)
end

function (model::HypernetDesignEncoder)(d::AbstractDesign, a::AbstractDesign)
    d1 = normalize(d, model.design_space)
    d2 = normalize(model.design_space(d, a), model.design_space)

    c1 = model.layers(d1)
    c2 = model.layers(d2)

    c1 = gpu(randn(Float32, size(c1)))
    c2 = gpu(randn(Float32, size(c2)))

    dc = (c2 .- c1) / (100/0.00001f0) ## hardcoded for now
    return hcat(c1, dc)
end

struct LatentDynamics <: AbstractDynamics
    C0::Float32
    freq::Float32
    pml::AbstractVector{Float32}
    grad::AbstractMatrix{Float32}
    bc::AbstractVector{Float32}
end

Flux.@functor LatentDynamics
Flux.trainable(::LatentDynamics) = (;)

function LatentDynamics(dim::OneDim; ambient_speed, freq, pml_width, pml_scale)
    pml = build_pml(dim, pml_width, pml_scale)
    grad = build_gradient(dim)
    bc = dirichlet(dim)
    return LatentDynamics(ambient_speed, freq, pml, grad, bc)
end

function (dyn::LatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]
    f = wave[:, 3]
    c = wave[:, 4]
    dc = wave[:, 5]

    force = f * sin(2.0f0 * pi * dyn.freq * t)
    du = dyn.C0 ^ 2 * c .* (dyn.grad * v) .- dyn.pml .* u
    dv = (dyn.grad * (u .+ force)) .- dyn.pml .* v
    df = f * 0.0f0
    ## static wavespeed
    # dc = c * 0.0f0
    ddc = dc * 0.0f0
    return hcat(du .* dyn.bc, dv, df, dc, ddc)
end

struct ScatteredEnergyModel
    wave_encoder::Chain
    design_encoder::HypernetDesignEncoder
    latent_dim::OneDim
    iter::Integrator
    design_space::DesignSpace
    mlp::Chain
end

Flux.@functor ScatteredEnergyModel

function propagate(model::ScatteredEnergyModel, latent_state::AbstractMatrix{Float32}, d::AbstractDesign, a::AbstractDesign)
    c = model.design_encoder(d, a)
    zi = hcat(latent_state, c)
    return (model.iter(zi), model.design_space(d, a))
end

function (model::ScatteredEnergyModel)(h::Tuple{AbstractMatrix{Float32}, AbstractDesign}, a::AbstractDesign)
    latent_state, d = h
    z, d = propagate(model, latent_state, d, a)
    return (z[:, [1, 2, 3], end], d), z
end

function flatten_repeated_last_dim(x::AbstractArray{Float32})

    last_dim = size(x, ndims(x))
    first_component = selectdim(x, ndims(x), 1)
    second_component = selectdim(selectdim(x, ndims(x) - 1, 2:size(x, ndims(x) - 1)), ndims(x), 2:last_dim)
    new_dims = (size(second_component)[1:end-2]..., prod(size(second_component)[end-1:end]))

    return cat(
        first_component,
        reshape(second_component, new_dims),
        dims = ndims(x) - 1)
end

function generate_latent_solution(model::ScatteredEnergyModel, latent_state::AbstractMatrix{Float32}, d::AbstractDesign, a::Vector{<: AbstractDesign})
    recur = Recur(model, (latent_state, d))
    return cat([recur(action) for action in a]..., dims = 4)
end

function (model::ScatteredEnergyModel)(s::WaveEnvState, a::Vector{<:AbstractDesign})
    latent_state = model.wave_encoder(s)[:, :, 1]
    z = flatten_repeated_last_dim(generate_latent_solution(model, latent_state, s.design, a))
    return model.mlp(Flux.unsqueeze(z, dims = ndims(z) + 1))
end

function (model::ScatteredEnergyModel)(states::Vector{WaveEnvState}, a::Vector{<: Vector{<: AbstractDesign}})
    latent_states = Flux.unbatch(model.wave_encoder(states))
    z = cat(flatten_repeated_last_dim.(generate_latent_solution.([model], latent_states, [s.design for s in states], a))..., dims = 4)
    return model.mlp(z)
end

using Statistics: mean
function compute_gradient(model, states, actions, sigmas, loss_func::Function)
    y = hcat(flatten_repeated_last_dim.(sigmas)...)
    loss, back = Flux.pullback(m -> loss_func(m(states, actions), y), model)
    gs = back(one(loss))[1]

    return loss, gs
end

function visualize!(model, s::WaveEnvState, a::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix; path::String)

    tspan = cpu(flatten_repeated_last_dim(tspan))

    latent_state = model.wave_encoder(s)[:, :, 1]
    design = s.design

    z = cpu(flatten_repeated_last_dim(generate_latent_solution(model, latent_state, design, a)))
    dim = cpu(model.latent_dim)
    pred_sigma = cpu(model(s, a))[:, 1]

    fig = Figure(resolution = (1920, 1080), fontsize = 30)
    z_grid = fig[1, 1] = GridLayout()

    ax1, hm1 = heatmap(z_grid[1, 1], dim.x, tspan, z[:, 1, :], colormap = :ice)
    Colorbar(z_grid[1, 2], hm1)
    ax1.title = "Displacement"
    ax1.ylabel = "Time (s)"
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false

    ax2, hm2 = heatmap(z_grid[1, 3], dim.x, tspan, z[:, 2, :], colormap = :ice)
    Colorbar(z_grid[1, 4], hm2)
    ax2.title = "Velocity"
    ax2.xticklabelsvisible = false
    ax2.xticksvisible = false
    ax2.yticklabelsvisible = false
    ax2.yticksvisible = false

    ax3, hm3 = heatmap(z_grid[2, 1], dim.x, tspan, z[:, 3, :], colormap = :ice, colorrange = (-1.0, 1.0))
    Colorbar(z_grid[2, 2], hm3)
    ax3.title = "Force"
    ax3.xlabel = "Distance (m)"
    ax3.ylabel = "Time (s)"

    ax4, hm4 = heatmap(z_grid[2, 3], dim.x, tspan, z[:, 4, :], colormap = :ice, colorrange = (0.0, 1.0))
    Colorbar(z_grid[2, 4], hm4)
    ax4.title = "Wave Speed"
    ax4.xlabel = "Distance (m)"
    ax4.yticklabelsvisible = false
    ax4.yticksvisible = false

    p_grid = fig[1, 2] = GridLayout()
    p_axis = Axis(p_grid[1, 1], title = "Prediction of Scattered Energy Versus Ground Truth", xlabel = "Time (s)", ylabel = "Scattered Energy (σ)")

    sigma = cpu(flatten_repeated_last_dim(sigma))
    lines!(p_axis, tspan, sigma, color = :blue, label = "True")
    lines!(p_axis, tspan, pred_sigma, color = :orange, label = "Predicted")
    axislegend(p_axis, position = :rb)

    save(joinpath(path, "latent.png"), fig)
    return nothing
end

function overfit!(model, states, actions, tspans, sigmas, lr, n)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, model)

    for i in 1:n
        @time loss, gs = compute_gradient(model, states, actions, sigmas, Flux.mse)
        println("Update: $i, Loss: $loss")
        opt_state, model = Optimisers.update(opt_state, model, gs)
    end

    for (i, (s, a, tspan, sigma)) in enumerate(zip(states, actions, tspans, sigmas))
        @time visualize!(model, s, a, tspan, sigma, path = mkpath("$i"))
    end
end

function train_loop(
        model; #::ScatteredEnergyModel;
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

        for (j, batch) in enumerate(train_loader)
            states, actions, tspans, sigmas = gpu(batch)

            loss, gs = compute_gradient(model, states, actions, sigmas, loss_func)
            push!(epoch_loss, loss)

            opt_state, model = Optimisers.update(opt_state, model, gs)

            if j == train_steps
                break
            end
        end

        opt_state = Optimisers.adjust(opt_state, lr * decay_rate ^ i)

        push!(metrics[:train_loss], sum(epoch_loss) / train_steps)

        epoch_loss = Vector{Float32}()
        epoch_path = mkpath(joinpath(path, "epoch_$i"))

        for (j, batch) in enumerate(val_loader)
            states, actions, tspans, sigmas = gpu(batch)

            y = hcat(flatten_repeated_last_dim.(sigmas)...)
            loss = loss_func(model(states, actions), y)
            push!(epoch_loss, loss)

            if j <= evaluation_samples
                latent_path = mkpath(joinpath(epoch_path, "latent_$j"))
                for (i, (s, a, tspan, sigma)) in enumerate(zip(states, actions, tspans, sigmas))
                    try
                        @time visualize!(model, s, a, tspan, sigma, path = mkpath(joinpath(latent_path, "$i")))
                    catch e
                        println(e)
                    end
                end
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
            println("Checkpointing")

            BSON.bson(joinpath(epoch_path, "model.bson"), model = cpu(model))
        end
    end
end