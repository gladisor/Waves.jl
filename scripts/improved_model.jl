const EPSILON = Float32(1e-3)

abstract type WaveInputLayer end

struct TotalWaveInput <: WaveInputLayer end
Flux.@functor TotalWaveInput
(input::TotalWaveInput)(s::WaveEnvState) = s.wave_total[:, :, :, :]

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
        # InstanceNorm(out_channels),
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

function NormalizedDense(in_size::Int, out_size::Int, act::Function)
    return NormalizedDense(Dense(in_size, out_size), LayerNorm(out_size), act)
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
        Dense(h_size, 3, tanh)
        )

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
        Dense(h_size, length(ps), bias = false),
        vec,
        re,
        FrequencyDomain(latent_dim, nfreq),
        Scale([1.0f0, 1.0f0/WATER, 1.0f0], false),
        z -> permutedims(z, (2, 1))
        )

    return model
end

struct HypernetDesignEncoder
    design_space::DesignSpace
    action_space::DesignSpace
    layers::Chain
end

Flux.@functor HypernetDesignEncoder
Flux.trainable(model::HypernetDesignEncoder) = (;model.layers)

function HypernetDesignEncoder(
        design_space::DesignSpace,
        action_space::DesignSpace,
        nfreq::Int,
        h_size::Int,
        n_h::Int,
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

    in_size = length(vec(design_space.low)) + length(vec(action_space.low))

    layers = Chain(
        # Dense(in_size, h_size, activation),
        # LayerNorm(h_size),
        # Dense(h_size, h_size, activation),
        # LayerNorm(h_size),
        # Dense(h_size, h_size, activation),
        # LayerNorm(h_size),
        # Dense(h_size, h_size, activation),
        # LayerNorm(h_size),
        # Dense(h_size, h_size, activation),
        # LayerNorm(h_size),

        NormalizedDense(in_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        Dense(h_size, length(ps), bias = false),
        re,
        FrequencyDomain(latent_dim, nfreq),
        vec,
        )

    return HypernetDesignEncoder(design_space, action_space, layers)
end

function (model::HypernetDesignEncoder)(d::AbstractDesign, a::AbstractDesign)
    d = (d - model.design_space.low) / (model.design_space.high - model.design_space.low + EPSILON)
    a = (a - model.action_space.low) / (model.action_space.high - model.action_space.low + EPSILON)
    x = vcat(vec(d), vec(a))
    return model.layers(x)
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

    force = f * sin(2.0f0 * pi * dyn.freq * t)

    du = dyn.C0 ^ 2 * c .* (dyn.grad * v) .- dyn.pml .* u
    dv = (dyn.grad * (u .+ force)) .- dyn.pml .* v
    df = f * 0.0f0
    dc = c * 0.0f0

    return hcat(du .* dyn.bc, dv, df, dc)
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
Flux.trainable(model::ScatteredEnergyModel) = (;model.wave_encoder, model.design_encoder, model.mlp)

function propagate(model::ScatteredEnergyModel, latent_state::AbstractMatrix{Float32}, d::AbstractDesign, a::AbstractDesign)
    c = model.design_encoder(d, a)
    zi = hcat(latent_state, c)
    return (model.iter(zi), model.design_space(d, a))
end

function (model::ScatteredEnergyModel)(h::Tuple{AbstractMatrix{Float32}, AbstractDesign}, a::AbstractDesign)
    latent_state, d = h
    z, d = propagate(model, latent_state, d, a)
    return (z[:, [1, 2, 3], end], d), model.mlp(z)
end

function (model::ScatteredEnergyModel)(s::WaveEnvState, a::Vector{<:AbstractDesign})
    latent_state = model.wave_encoder(s)
    recur = Recur(model, (latent_state, s.design))
    return hcat([recur(action) for action in a]...)
end

function compute_gradient(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, sigma::AbstractMatrix{Float32}, loss_func::Function)
    loss, back = Flux.pullback(_model -> loss_func(_model(s, a), sigma), model)
    return (loss, back(one(loss))[1])
end

function visualize!(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix; path::String)

    tspan = cpu(tspan)
    tspan_flat = vcat(tspan[1], vec(tspan[2:end, :]))

    latent_state = model.wave_encoder(s)
    design = s.design

    zs = []
    for (i, action) in enumerate(a)
        z, design = propagate(model, latent_state, design, action)
        latent_state = z[:, [1, 2, 3], end]

        if i == 1
            push!(zs, cpu(z))
        else
            push!(zs, cpu(z[:, :, 2:end]))
        end
    end

    dim = cpu(model.latent_dim)

    z = cat(zs..., dims = ndims(zs[1]))
    pred_sigma = cpu(model(s, a))

    fig = Figure(resolution = (1920, 1080), fontsize = 30)
    z_grid = fig[1, 1] = GridLayout()

    ax1, hm1 = heatmap(z_grid[1, 1], dim.x, tspan_flat, z[:, 1, :], colormap = :ice)
    Colorbar(z_grid[1, 2], hm1)
    ax1.title = "Displacement"
    ax1.ylabel = "Time (s)"
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false

    ax2, hm2 = heatmap(z_grid[1, 3], dim.x, tspan_flat, z[:, 2, :], colormap = :ice)
    Colorbar(z_grid[1, 4], hm2)
    ax2.title = "Velocity"
    ax2.xticklabelsvisible = false
    ax2.xticksvisible = false
    ax2.yticklabelsvisible = false
    ax2.yticksvisible = false

    ax3, hm3 = heatmap(z_grid[2, 1], dim.x, tspan_flat, z[:, 3, :], colormap = :ice)
    Colorbar(z_grid[2, 2], hm3)
    ax3.title = "Force"
    ax3.xlabel = "Distance (m)"
    ax3.ylabel = "Time (s)"

    ax4, hm4 = heatmap(z_grid[2, 3], dim.x, tspan_flat, z[:, 4, :], colormap = :ice)
    Colorbar(z_grid[2, 4], hm4)
    ax4.title = "Wave Speed"
    ax4.xlabel = "Distance (m)"
    ax4.yticklabelsvisible = false
    ax4.yticksvisible = false

    p_grid = fig[1, 2] = GridLayout()
    p_axis = Axis(p_grid[1, 1], title = "Prediction of Scattered Energy Versus Ground Truth", xlabel = "Time (s)", ylabel = "Scattered Energy (σ)")

    for i in axes(tspan, 2)
        lines!(p_axis, tspan[:, i], cpu(sigma[:, i]), color = :blue, label = "True", linewidth = 3)
        lines!(p_axis, tspan[:, i], pred_sigma[:, i], color = :orange, label = "Predicted", linewidth = 3)
    end

    save(joinpath(path, "latent.png"), fig)
    render!(dim, z, path = joinpath(path, "latent.mp4"))
    return nothing
end