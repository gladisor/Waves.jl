const EPSILON = Float32(1e-3)

"""
WaveInputLayer is an abstract input layer which handles the conversion from a WaveEnvState
or a vector of WaveEnvState(s) to the correct input format to a CNN model.
"""
abstract type WaveInputLayer end
(input::WaveInputLayer)(s::Vector{WaveEnvState}) = cat(input.(s)..., dims = 4)

struct TotalWaveInput <: WaveInputLayer end
Flux.@functor TotalWaveInput
(input::TotalWaveInput)(s::WaveEnvState) = s.wave_total[:, :, :, :] .+ 1f-5

struct ResidualBlock
    main::Chain
    skip::Conv
    activation::Function
    pool::MaxPool
end

Flux.@functor ResidualBlock

function ResidualBlock(k::Tuple{Int, Int}, in_channels::Int, out_channels::Int, activation::Function)
    main = Chain(
        Conv(k, in_channels => out_channels, pad = SamePad()),
        BatchNorm(out_channels),
        activation,
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
    return cat([hypernet.domain(m) for m in models]..., dims = ndims(x) + 1)
end

## forier learning
struct SinWaveEmbedder
    frequencies::AbstractMatrix{Float32}
end

Flux.@functor SinWaveEmbedder
Flux.trainable(::SinWaveEmbedder) = (;)

function SinWaveEmbedder(dim::OneDim, nfreq::Int)
    dim = cpu(dim)
    L = dim.x[end] - dim.x[1]
    C = L / 2.0f0

    n = Float32.(collect(1:nfreq))
    frequencies = (pi * n .* (dim.x' .- C)) / L
    return SinWaveEmbedder(permutedims(sin.(frequencies), (2, 1)))
end

function (embedder::SinWaveEmbedder)(x::AbstractMatrix{Float32})
    x_norm = x ./ sum(abs, x, dims = 1)
    return (embedder.frequencies * x_norm)
end

function (embedder::SinWaveEmbedder)(x::AbstractArray{Float32, 3})
    x_norm = x ./ sum(abs, x, dims = 1)
    return batched_mul(embedder.frequencies, x_norm)
end

struct NonTrainableScale
    x::AbstractVector{Float32}
end

Flux.@functor NonTrainableScale
Flux.trainable(::NonTrainableScale) = (;)

function (nts::NonTrainableScale)(x::AbstractArray{Float32})
    return nts.x .* x
end

struct LatentWaveNormalization
    weight::AbstractArray{Float32}
end

Flux.@functor LatentWaveNormalization
Flux.trainable(::LatentWaveNormalization) = (;)

sigmoid_step(x, l) = sigmoid.(-5.0f0 * (x .- l)) - sigmoid.(-5.0f0 * (x .+ l))

function LatentWaveNormalization(dim::OneDim, C::Float32)

    u_weight = dim.x .^ 0.0f0
    f_weight = sigmoid_step(dim.x, dim.x[end] - 5.0f0)

    return LatentWaveNormalization(
        hcat(
            u_weight, 
            u_weight, 
            u_weight ./ C, 
            u_weight ./ C, 
            f_weight
            )[:, :, :]
        )
end

function (n::LatentWaveNormalization)(x)
    return n.weight .* x
end

function build_split_hypernet_wave_encoder(;
        latent_dim::OneDim,
        nfreq::Int,
        h_size::Int,
        activation::Function,
        input_layer::WaveInputLayer,
        )

    latent_elements = size(latent_dim, 1)

    ## zero out forces in pml
    model = Chain(
        input_layer,
        MeanPool((4, 4)),
        ResidualBlock((3, 3), 3, 32, activation),
        ResidualBlock((3, 3), 32, 64, activation),
        ResidualBlock((3, 3), 64, h_size, activation),
        GlobalMaxPool(),
        flatten,

        # Dense(h_size, 3 * nfreq, tanh), ## output frequencies for all fields (u, v, f)
        # b -> reshape(b, nfreq, 3, :),
        Dense(h_size, 5 * nfreq, tanh),
        b -> reshape(b, nfreq, 5, :),

        SinWaveEmbedder(latent_dim, nfreq), ## embed into sine wave
        LatentWaveNormalization(latent_dim, WATER)
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

    in_size = length(vec(design_space.low))

    layers = Chain(
        x -> x[:, :],
        Dense(in_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, nfreq, tanh),
        SinWaveEmbedder(latent_dim, nfreq),
        c -> sigmoid.(c .+ 0.5f0) .+ 0.20f0
    )

    return HypernetDesignEncoder(design_space, action_space, layers)
end

"""
Normalizes the design parameter vector between -1 and 1
"""
function normalize(design::AbstractDesign, ds::DesignSpace)
    scale = 2.0f0
    return scale * (vec(design) .- vec(ds.low)) ./ (vec(ds.high) .- vec(ds.low) .+ EPSILON) .- (scale / 2.0f0)
end

"""
A single propagation step for the design, increments the design to the next one by applying the
action. Produces the wavespeed representation for the new design.
"""
function (model::HypernetDesignEncoder)(d1::AbstractDesign, a::AbstractDesign)
    d2 = model.design_space(d1, a)
    c = model.layers(normalize(d2, model.design_space))
    return (d2, c)
end

"""
Generates a sequence of wavespeed functions by evaluating the design after applying each
action.
"""
function (model::HypernetDesignEncoder)(d::AbstractDesign, a::DesignSequence)
    recur = Recur(model, d)
    return hcat(
        model.layers(normalize(d, model.design_space)),
        [recur(action) for action in a]...
        )
end

(model::HypernetDesignEncoder)(s::WaveEnvState, a::DesignSequence) = model(s.design, a)

"""
Converts a sequence of wavespeed fields into an array of (elements x steps x 2) where the 
last dimention contains the initial wavespeed field and dwavespeed/dt.
"""
function build_wavespeed_fields(c::AbstractMatrix{Float32})
    dc = diff(c, dims = 2) ./ 1.0f-3 ## hardcoded for now
    return cat(c[:, 1:size(c, ndims(c))-1], dc, dims = 3)
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

# function (dyn::LatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
#     u = wave[:, 1]
#     v = wave[:, 2]
#     f = wave[:, 3]
#     c = wave[:, 4]
#     dc = wave[:, 5]

#     force = f * sin(2.0f0 * pi * dyn.freq * t)
#     du = dyn.C0 ^ 2 * c .* (dyn.grad * v) .- dyn.pml .* u
#     dv = (dyn.grad * (u .+ force)) .- dyn.pml .* v

#     return hcat(
#         du .* dyn.bc, ## remeber to turn bc off when using pml
#         dv,
#         f * 0.0f0,
#         dc, 
#         dc * 0.0f0
#         )
# end

function (dyn::LatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u_inc = wave[:, 1]
    u_tot = wave[:, 2]
    v_inc = wave[:, 3]
    v_tot = wave[:, 4]
    f = wave[:, 5]

    c = wave[:, 6]
    dc = wave[:, 7]

    force = f * sin(2.0f0 * pi * dyn.freq * t)

    du_inc = dyn.C0 ^ 2 * (dyn.grad * v_inc) .- dyn.pml .* u_inc
    du_tot = dyn.C0 ^ 2 * c .* (dyn.grad * v_tot) .- dyn.pml .* u_tot

    dv_inc = (dyn.grad * (u_inc .+ force)) .- dyn.pml .* v_inc
    dv_tot = (dyn.grad * (u_tot .+ force)) .- dyn.pml .* v_tot

    return hcat(
        du_inc .* dyn.bc,
        du_tot .* dyn.bc,
        dv_inc,
        dv_tot,
        f * 0.0f0,

        dc,
        dc * 0.0f0
        )
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

function (model::ScatteredEnergyModel)(latent_state::AbstractMatrix{Float32}, c_dc::AbstractMatrix{Float32})
    zi = hcat(latent_state, c_dc)
    z = model.iter(zi)
    # return (z[:, [1, 2, 3], end], z) ## For latent dynamics
    return (z[:, 1:end-2, end], z)
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

function flatten_repeated_last_dim(x::Vector{<:AbstractMatrix{Float32}})
    return hcat(flatten_repeated_last_dim.(x)...)
end

function generate_latent_solution(model::ScatteredEnergyModel, latent_state::AbstractMatrix{Float32}, c::AbstractMatrix{Float32})
    c_dc = build_wavespeed_fields(c)
    recur = Recur(model, latent_state)
    return cat([recur(c_dc[:, i, :]) for i in axes(c_dc, 2)]..., dims = 4)
end

function generate_latent_solution(model::ScatteredEnergyModel, s::WaveEnvState, a::DesignSequence)
    latent_state = model.wave_encoder(s)[:, :, 1]
    c = model.design_encoder(s.design, a)
    z = flatten_repeated_last_dim(generate_latent_solution(model, latent_state, c))
    return Flux.unsqueeze(z, dims = ndims(z) + 1)
end

function generate_latent_solution(model::ScatteredEnergyModel, s::Vector{WaveEnvState}, a::Vector{<: DesignSequence})
    latent_state = Flux.unbatch(model.wave_encoder(s))
    c = model.design_encoder.(s, a)
    z = cat(flatten_repeated_last_dim.(generate_latent_solution.([model], latent_state, c))..., dims = 4)
    return z
end

function (model::ScatteredEnergyModel)(
        s::Union{WaveEnvState, Vector{WaveEnvState}}, 
        a::Union{
            DesignSequence, 
            Vector{<: DesignSequence}
            }
        )
    
    z = generate_latent_solution(model, s, a)
    return model.mlp(z)
end

"""
Computes the gradient of the model parameters with respect to the loss function on a batch of
data.
"""
function compute_gradient(
        model::ScatteredEnergyModel, 
        states::Vector{WaveEnvState}, 
        actions::Vector{<: DesignSequence},
        sigmas::Vector{<: AbstractArray{Float32}}, 
        loss_func::Function)

    y = hcat(flatten_repeated_last_dim.(sigmas)...)
    loss, back = Flux.pullback(m -> loss_func(m(states, actions), y), model)
    gs = back(one(loss))[1]

    return loss, gs
end

function visualize!(model, s::WaveEnvState, a::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix; path::String)

    tspan = cpu(flatten_repeated_last_dim(tspan))

    latent_state = model.wave_encoder(s)[:, :, 1]
    design = s.design

    z = cpu(flatten_repeated_last_dim(generate_latent_solution(model, s, a)))
    dim = cpu(model.latent_dim)
    pred_sigma = cpu(model(s, a))[:, 1]

    fig = Figure(resolution = (1920, 1080), fontsize = 30)
    z_grid = fig[1, 1] = GridLayout()

    # ax1, hm1 = heatmap(z_grid[1, 1], dim.x, tspan, z[:, 1, :], colormap = :ice)
    ax1, hm1 = heatmap(z_grid[1, 1], dim.x, tspan, z[:, 2, :] .- z[:, 1, :], colormap = :ice)
    Colorbar(z_grid[1, 2], hm1)
    ax1.title = "Displacement"
    ax1.ylabel = "Time (s)"
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false

    # ax2, hm2 = heatmap(z_grid[1, 3], dim.x, tspan, z[:, 2, :], colormap = :ice)
    ax2, hm2 = heatmap(z_grid[1, 3], dim.x, tspan, z[:, 4, :] .- z[:, 3, :], colormap = :ice)
    Colorbar(z_grid[1, 4], hm2)
    ax2.title = "Velocity"
    ax2.xticklabelsvisible = false
    ax2.xticksvisible = false
    ax2.yticklabelsvisible = false
    ax2.yticksvisible = false

    # ax3, hm3 = heatmap(z_grid[2, 1], dim.x, tspan, z[:, 3, :], colormap = :ice, colorrange = (-1.0, 1.0))
    ax3, hm3 = heatmap(z_grid[2, 1], dim.x, tspan, z[:, 5, :], colormap = :ice, colorrange = (-1.0, 1.0))
    Colorbar(z_grid[2, 2], hm3)
    ax3.title = "Force"
    ax3.xlabel = "Distance (m)"
    ax3.ylabel = "Time (s)"

    ax4, hm4 = heatmap(
        z_grid[2, 3], 
        dim.x, 
        tspan, 
        # z[:, 4, :], 
        z[:, 6, :],
        colormap = :ice, 
        colorrange = (0.0, 1.2)
        )

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

    z_sc = z[:, 2, :] .- z[:, 1, :]
    # z_sc = z[:, 1, :]
    render!(dim, cpu(z_sc), path = joinpath(path, "vid.mp4"))
    return nothing
end

function visualize!(model::ScatteredEnergyModel, states::Vector{WaveEnvState}, actions::Vector{<: DesignSequence}, tspans::Vector{<: AbstractMatrix{Float32}}, sigmas::Vector{<: AbstractMatrix{Float32}}; path::String)
    
    for i in axes(states, 1)
        visualize!(model, states[i], actions[i], tspans[i], sigmas[i], path = mkpath(joinpath(path, "$i")))
    end

    return nothing
end

function train_loop(
        model::ScatteredEnergyModel;
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
        path::String,
        opt,
        )

    # lg = WandbLogger(;project = "Waves.jl")
    # global_logger(lg)

    opt_state = Optimisers.setup(opt, model)
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}())

    for i in 1:epochs
        epoch_loss = Vector{Float32}()

        trainmode!(model)
        for (j, batch) in enumerate(train_loader)
            print(i)
            print(" ")
            states, actions, tspans, sigmas = gpu(batch)
            
            # loss, back = Flux.pullback(m -> loss_func(m, states, actions, sigmas), model)
            # gs = back(one(loss))[1]

            y = flatten_repeated_last_dim(sigmas)
            loss, back = Flux.pullback(m -> loss_func(m(states, actions), y), model)
            gs = back(one(loss))[1]

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

        testmode!(model)
        for (j, batch) in enumerate(val_loader)
            states, actions, tspans, sigmas = gpu(batch)

            # loss = loss_func(model, states, actions, sigmas)
            y = flatten_repeated_last_dim(sigmas)
            loss = loss_func(model(states, actions), y)

            push!(epoch_loss, loss)

            if j <= evaluation_samples
                latent_path = mkpath(joinpath(epoch_path, "latent_$j"))

                for (i, (s, a, tspan, sigma)) in enumerate(zip(states, actions, tspans, sigmas))
                    try
                        ## change this back s versus s[1]
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

        # Wandb.log(lg, Dict("Epoch" => i, "Train Loss" => train_loss, "Val Loss" => val_loss))

        if i % checkpoint_every == 0 || i == epochs
            println("Checkpointing")

            BSON.bson(joinpath(epoch_path, "model.bson"), model = cpu(model))
        end
    end

    # close(lg)
end

function compute_manual_gradient(model::ScatteredEnergyModel, s::WaveEnvState, a::DesignSequence, sigma::AbstractMatrix{Float32})

    y = flatten_repeated_last_dim(sigma)

    uvf, back_wave_enc = Flux.pullback(m -> m(s)[:, :, 1], model.wave_encoder)
    c, back_design_enc = Flux.pullback(m -> m(s, a), model.design_encoder)
    
    z, back_latent_dynamics = Flux.pullback(
        (m, _uvf, _c) -> flatten_repeated_last_dim(generate_latent_solution(m, _uvf, _c))[:, :, :, :],
        model, uvf, c)
    
    y_hat, back_mlp = Flux.pullback((m, _z) -> m(_z), model.mlp, z)
    
    loss, back_loss = Flux.pullback(_y_hat -> Flux.mse(_y_hat, y), y_hat)

    loss_adj = back_loss(one(loss))[1]
    mlp_adj = back_mlp(loss_adj)

    latent_dynamics_adj = back_latent_dynamics(mlp_adj[2] ./ (model.iter.steps * length(a)))

    gs = (
        mlp = mlp_adj[1],
        iter = latent_dynamics_adj[1].iter,
        latent_dim = nothing,
        design_space = nothing,
        design_encoder = back_design_enc(latent_dynamics_adj[3])[1],
        wave_encoder = back_wave_enc(latent_dynamics_adj[2])[1]
        )

    return loss, gs
end

function overfit(
        model::ScatteredEnergyModel, 
        s::Union{WaveEnvState, Vector{WaveEnvState}},
        a::Union{DesignSequence, Vector{<: DesignSequence}},
        t::Union{AbstractMatrix{Float32}, Vector{ <: AbstractMatrix{Float32}}}, 
        sigma::Union{AbstractMatrix{Float32}, Vector{ <: AbstractMatrix{Float32}}},
        n::Int; 
        opt,
        path::String = "")

    y = flatten_repeated_last_dim(sigma)

    opt_state = Optimisers.setup(opt, model)

    trainmode!(model)

    for i in 1:n

        loss, back = Flux.pullback(m -> Flux.mse(m(s, a), y), model)
        gs = back(one(loss))[1]

        # loss, gs = compute_manual_gradient(model, s[1], a[1], sigma[1])
        println("Update: $i, Loss: $loss")
        opt_state, model = Optimisers.update(opt_state, model, gs)
    end

    testmode!(model)

    visualize!(model, s, a, t, sigma, path = path)
    return model
end

function test_design_encoder(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign})
    
    d1 = s.design
    d2 = model.design_space(d1, a[1])
    
    c1 = model.design_encoder.layers(normalize(d1, model.design_space))
    c2 = model.design_encoder.layers(normalize(d2, model.design_space))
    
    uvf = model.wave_encoder(s)[:, :, 1]
    c = model.design_encoder(s.design, a[1])
    z = model.iter(hcat(uvf, c))
    
    display(c1 ≈ c2)
    display(z[:, 4, 1] ≈ z[:, 4, end])
    display(c1 ≈ z[:, 4, 1])
    display(c2 ≈ z[:, 4, end])
end

# function reshape_latent_solution(z::AbstractArray{Float32})
#     z = z[:, [1, 2, 3, 4], :, :]
#     # z = z[:, 1:end-1, :, :]
#     return reshape(z, (size(z, 1) * size(z, 2), :, size(z, ndims(z))))
# end

function build_mlp_decoder(latent_elements::Int, h_size::Int, activation::Function)
    return Chain(
        Dense(4 * latent_elements, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, 1),
        flatten
    )
end

function build_scattered_wave_decoder(latent_elements::Int, h_size::Int, k_size::Int, activation::Function)
    return Chain(

        x -> vcat(
            x[:, 2, :, :] .- x[:, 1, :, :], 
            x[:, 4, :, :] .- x[:, 3, :, :],
            x[:, end-1, :, :]),
            
        x -> permutedims(x, (2, 1, 3)),

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), 3 * latent_elements => h_size, activation),

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size, activation),

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size, activation),

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size, activation),

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size, activation),

        Conv((1,), h_size => 1),
        flatten
    )
end

function build_full_cnn_decoder(latent_elements::Int, h_size::Int, k_size::Int, activation::Function)
    return Chain(
        x -> permutedims(x, (2, 1, 3)),
        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), 4 * latent_elements => h_size),
        activation,

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size),
        activation,

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size),
        activation,

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size),
        activation,

        x -> pad_reflect(x, (k_size - 1, 0)),
        Conv((k_size,), h_size => h_size),
        activation,

        Conv((1,), h_size => 1),
        flatten
    )
end