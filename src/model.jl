export build_split_hypernet_wave_encoder, HypernetDesignEncoder, LatentDynamics, build_scattered_wave_decoder, ScatteredEnergyModel, train_loop, visualize!, plot_latent_simulation_and_scattered_energy!, overfit, generate_latent_solution, flatten_repeated_last_dim

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
    damper = sigmoid_step(dim.x, dim.x[end] - 5.0f0)
    return LatentWaveNormalization(hcat(damper, damper, damper ./ C, damper ./ C, damper)[:, :, :])
end

function (n::LatentWaveNormalization)(x)
    return n.weight .* x
end

function build_split_hypernet_wave_encoder(;
        latent_dim::OneDim,
        nfreq::Int,
        h_size::Int,
        activation::Function)

    latent_elements = size(latent_dim, 1)

    ## zero out forces in pml
    model = Chain(
        TotalWaveInput(),
        MeanPool((4, 4)),
        ResidualBlock((3, 3), 3, 32, activation),
        ResidualBlock((3, 3), 32, 64, activation),
        ResidualBlock((3, 3), 64, h_size, activation),
        GlobalMaxPool(),
        flatten,
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
        Dense(in_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, nfreq, tanh),
        SinWaveEmbedder(latent_dim, nfreq),
        sigmoid
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
    return (d2, d2)
end

function (model::HypernetDesignEncoder)(d1::Vector{<: AbstractDesign}, a::Vector{<: AbstractDesign})
    d2 = model.design_space.(d1, a)
    return (d2, d2)
end

"""
Generates a sequence of wavespeed functions by evaluating the design after applying each
action.
"""
function (model::HypernetDesignEncoder)(d::AbstractDesign, a::DesignSequence)
    recur = Recur(model, d)
    design_sequence = vcat(d, [recur(action) for action in a])
    x = hcat(normalize.(design_sequence, [model.design_space])...)[:, :, :]
    return model.layers(x)
end

(model::HypernetDesignEncoder)(s::WaveEnvState, a::DesignSequence) = model(s.design, a)
(model::HypernetDesignEncoder)(states::Vector{WaveEnvState}, actions::Vector{<: DesignSequence}) = model(
    [s.design for s in states], 
    hcat(actions...)
    )

function (model::HypernetDesignEncoder)(d::Vector{<: AbstractDesign}, a::Matrix{<: AbstractDesign})
    recur = Recur(model, d)

    design_sequences = hcat(d, [recur(a[i, :]) for i in axes(a, 1)]...)
    x = normalize.(design_sequences, [model.design_space])
    x_batch = cat([hcat(x[i, :]...) for i in axes(x, 1)]..., dims = 3)
    return model.layers(x_batch)
end

"""
Converts a sequence of wavespeed fields into an array of (elements x steps x 2) where the 
last dimention contains the initial wavespeed field and dwavespeed/dt.
"""
function build_wavespeed_fields(c::AbstractMatrix{Float32})
    dc = diff(c, dims = 2) ./ 1.0f-3 ## hardcoded for now
    return cat(c[:, 1:size(c, ndims(c))-1], dc, dims = 3)
end

"""
Batchwise implementation
"""
function build_wavespeed_fields(c::AbstractArray{Float32, 3})

    ## currently the 1.0f-3 constant is computed by (steps - 1) * dt
    dc = diff(c, dims = 2) ./ 1.0f-3 ## hardcoded for now

    return cat(
        Flux.unsqueeze(c[:, 1:end-1, :], dims = 3),
        Flux.unsqueeze(dc, dims = 3),
        dims = 3
        )
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

## batch implementation
function (dyn::LatentDynamics)(wave::AbstractArray{Float32, 3}, t::AbstractVector{Float32})
    u_inc = wave[:, 1, :]
    u_tot = wave[:, 2, :]
    v_inc = wave[:, 3, :]
    v_tot = wave[:, 4, :]
    f = wave[:, 5, :]
    c = wave[:, 6, :]
    dc = wave[:, 7, :]

    force = f .* sin.(2.0f0 * pi * dyn.freq * permutedims(t))
    du_inc = dyn.C0 ^ 2 * (dyn.grad * v_inc) .- dyn.pml .* u_inc
    du_tot = dyn.C0 ^ 2 * c .* (dyn.grad * v_tot) .- dyn.pml .* u_tot

    dv_inc = (dyn.grad * (u_inc .+ force)) .- dyn.pml .* v_inc
    dv_tot = (dyn.grad * (u_tot .+ force)) .- dyn.pml .* v_tot

    return hcat(
        Flux.unsqueeze(du_inc .* dyn.bc, dims = 2),
        Flux.unsqueeze(du_tot .* dyn.bc, dims = 2),
        Flux.unsqueeze(dv_inc, dims = 2),
        Flux.unsqueeze(dv_tot, dims = 2),
        Flux.unsqueeze(f * 0.0f0, dims = 2),
        Flux.unsqueeze(dc, dims = 2),
        Flux.unsqueeze(dc * 0.0f0, dims = 2)
        )
end

function build_embedder(::LatentDynamics)

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

mutable struct CustomRecur{T,S}
    cell::T
    state::S
end

function (m::CustomRecur)(x...)
    m.state, y = m.cell(m.state, x...)
    return y
end

Flux.@functor CustomRecur
Flux.trainable(a::CustomRecur) = (; cell = a.cell)

"""
Inputs to this function are the latent state containing displacement, velocity and force fields
and also the c_dc which is the current design state and the rate at which the design changes.
"""
function (model::ScatteredEnergyModel)(latent_state::AbstractArray{Float32, 3}, c_dc::AbstractArray{Float32, 3}, t::AbstractMatrix{Float32})
    zi = hcat(latent_state, c_dc)
    z = model.iter(zi, t)

    ## last two fields are always wavespeed fields. c and dc
    next_latent_state = z[:, 1:end-2, :, end] ## (features x fields x batch x sequence)
    return (next_latent_state, z)
end

function preprocess_times(t::AbstractMatrix{Float32})
    return t[:, :, :]
end

function preprocess_times(t::Vector{<: AbstractMatrix{Float32}})
    return cat(t..., dims = 3)
end

"""
Propagates the latent solutions batchwise. Outputs a tensor: (elements x fields x sequence x batch)
"""
function generate_latent_solution(model::ScatteredEnergyModel, s, a, t)

    ## uvf is a tensor of shape (elements x fields x batch)
    uvf = model.wave_encoder(s)

    ## c_dc is a tensor of shape (elements x sequence x 2 x batch)
    ## the first of the third dimention is the starting wavespeed field
    ## the second of the third dimention is the time derivative of the starting wavespeed field
    c_dc = build_wavespeed_fields(model.design_encoder(s, a))

    ## times are a stack of (timesteps x sequence x batch)
    t = preprocess_times(t)

    recur = CustomRecur(model, uvf)

    z = flatten_repeated_last_dim(
        cat(
            [recur(c_dc[:, i, :, :], t[:, i, :]) for i in axes(c_dc, 2)]..., 
            dims = 5)
            )

    return permutedims(z, (1, 2, 4, 3))
end

function (model::ScatteredEnergyModel)(
        s::Union{WaveEnvState, Vector{WaveEnvState}}, 
        a::Union{
            DesignSequence, 
            Vector{<: DesignSequence}},
        t::Union{
            <: AbstractMatrix{Float32}, 
            Vector{<: AbstractMatrix{Float32}}})

    z = generate_latent_solution(model, s, a, t)
    return model.mlp(z)
end

function plot_latent_simulation_and_scattered_energy!(model::ScatteredEnergyModel, tspan::Vector, z::Array{Float32, 3}, sigma_pred::Vector, sigma_true::Vector; path::String)

    dim = cpu(model.latent_dim)

    fig = Figure(resolution = (1920, 1080), fontsize = 30)
    z_grid = fig[1, 1] = GridLayout()

    ax1, hm1 = heatmap(z_grid[1, 1], dim.x, tspan, z[:, 2, :] .- z[:, 1, :], colormap = :ice)
    Colorbar(z_grid[1, 2], hm1)
    ax1.title = "Displacement"
    ax1.ylabel = "Time (s)"
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false

    ax2, hm2 = heatmap(z_grid[1, 3], dim.x, tspan, z[:, 4, :] .- z[:, 3, :], colormap = :ice)
    Colorbar(z_grid[1, 4], hm2)
    ax2.title = "Velocity"
    ax2.xticklabelsvisible = false
    ax2.xticksvisible = false
    ax2.yticklabelsvisible = false
    ax2.yticksvisible = false

    ax3, hm3 = heatmap(z_grid[2, 1], dim.x, tspan, z[:, 5, :], colormap = :ice, colorrange = (-1.0, 1.0))
    Colorbar(z_grid[2, 2], hm3)
    ax3.title = "Force"
    ax3.xlabel = "Distance (m)"
    ax3.ylabel = "Time (s)"

    ax4, hm4 = heatmap(z_grid[2, 3], dim.x, tspan, z[:, 6, :], colormap = :ice, colorrange = (0.0, 1.2))
    Colorbar(z_grid[2, 4], hm4)
    ax4.title = "Wave Speed"
    ax4.xlabel = "Distance (m)"
    ax4.yticklabelsvisible = false
    ax4.yticksvisible = false

    p_grid = fig[1, 2] = GridLayout()
    p_axis = Axis(p_grid[1, 1], title = "Prediction of Scattered Energy Versus Ground Truth", xlabel = "Time (s)", ylabel = "Scattered Energy (σ)")
    lines!(p_axis, tspan, sigma_true, color = :blue, label = "True")
    lines!(p_axis, tspan, sigma_pred, color = :orange, label = "Predicted")
    axislegend(p_axis, position = :rb)
    save(path, fig)
end

function visualize!(model, s::WaveEnvState, a::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix; path::String, render::Bool = false)
    z = generate_latent_solution(model, s, a, tspan)
    
    tspan = cpu(flatten_repeated_last_dim(tspan))
    sigma_pred = vec(cpu(model.mlp(z[:, :, :, [1]])))
    sigma_true = cpu(flatten_repeated_last_dim(sigma))

    plot_latent_simulation_and_scattered_energy!(model, tspan, cpu(z[:, :, :, 1]), sigma_pred, sigma_true, path = joinpath(path, "latent.png"))

    z_sc = cpu(z[:, 2, :] .- z[:, 1, :])
    
    if render
        render!(cpu(model.latent_dim), z_sc, path = joinpath(path, "vid.mp4"))
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
        evaluation_samples::Int,
        checkpoint_every::Int,
        path::String,
        opt)

    opt_state = Optimisers.setup(opt, model)
    metrics = Dict(
        :train_loss => Vector{Float32}(), 
        :val_loss => Vector{Float32}())

    for i in 1:epochs
        epoch_loss = Vector{Float32}()

        trainmode!(model)
        for (j, batch) in enumerate(train_loader)
            print("#")
            states, actions, tspans, sigmas = gpu(batch)

            y = flatten_repeated_last_dim(sigmas)
            loss, back = Flux.pullback(m -> loss_func(m(states, actions, tspans), y), model)
            gs = back(one(loss))[1]

            push!(epoch_loss, loss)
            opt_state, model = Optimisers.update(opt_state, model, gs)

            if j == train_steps
                break
            end
        end

        println("")

        opt_state = Optimisers.adjust(opt_state, lr * decay_rate ^ i)

        push!(metrics[:train_loss], sum(epoch_loss) / train_steps)
        epoch_loss = Vector{Float32}()
        epoch_path = mkpath(joinpath(path, "epoch_$i"))

        testmode!(model)

        k = 1

        for (j, batch) in enumerate(val_loader)
            states, actions, tspans, sigmas = gpu(batch)

            y = flatten_repeated_last_dim(sigmas)
            loss = loss_func(model(states, actions, tspans), y)

            push!(epoch_loss, loss)

            eval_path = mkpath(joinpath(epoch_path, "eval"))

            if k <= evaluation_samples
                for (i, (s, a, tspan, sigma)) in enumerate(zip(states, actions, tspans, sigmas))

                    @time visualize!(model, s, a, tspan, sigma, path = mkpath(joinpath(eval_path, "$k")))

                    k += 1
                    if k >= evaluation_samples
                        break
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
            BSON.bson(joinpath(epoch_path, "opt_state.bson"), opt_state = cpu(opt_state))
        end
    end
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
        println("Update: $i, Loss: $loss")
        opt_state, model = Optimisers.update(opt_state, model, gs)
    end

    testmode!(model)

    visualize!(model, s[1], a[1], t[1], sigma[1], path = mkpath("1"))
    visualize!(model, s[2], a[2], t[2], sigma[2], path = mkpath("2"))
    visualize!(model, s[3], a[3], t[3], sigma[3], path = mkpath("3"))
    return model
end

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