struct LocalizationLayer
    coords::AbstractArray{Float32, 4}
end

Flux.@functor LocalizationLayer
Flux.trainable(::LocalizationLayer) = (;)

function LocalizationLayer(dim::TwoDim, resolution::Tuple{Int, Int})
    x = imresize(build_grid(dim), resolution) ./ maximum(dim.x)
    return LocalizationLayer(x[:, :, :, :])
end

function (layer::LocalizationLayer)(x)
    return cat(
        x,
        repeat(layer.coords, 1, 1, 1, size(x, 4)),
        dims = 3
    )
end

function Waves.build_wave_encoder(;
        latent_dim::OneDim,
        h_size::Int,
        nfreq::Int,
        c0::Float32,
        k::Tuple{Int, Int} = (3, 3),
        in_channels::Int = 3,
        activation::Function = leakyrelu)

    nfields = 6

    return Chain(
        Waves.TotalWaveInput(),
        LocalizationLayer(env.dim, env.resolution),
        Waves.ResidualBlock(k, 2 + in_channels, 32, activation),
        Waves.ResidualBlock(k, 32, 64, activation),
        Waves.ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        Flux.flatten,
        Parallel(
            vcat,
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
        ),
        b -> reshape(b, nfreq, nfields, :),
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(
            x[:, [1], :],       # u_tot
            x[:, [2], :] ./ c0, # v_tot
            x[:, [3], :],       # u_inc
            x[:, [4], :] ./ c0, # v_inc
            x[:, [5], :],       # f
            x[:, [6], :] .^ 2   # pml
            )
        )
end

function Waves.AcousticEnergyModel(;
        env::WaveEnv, 
        latent_dim::OneDim,
        h_size::Int, 
        nfreq::Int, 
        pml_width::Float32,
        pml_scale::Float32)

    wave_encoder = Waves.build_wave_encoder(;
        latent_dim, 
        h_size, 
        nfreq,
        c0 = env.iter.dynamics.c0)

    mlp = Chain(
        Dense(length(vec(env.design)), h_size, leakyrelu), 
        Dense(h_size, h_size, leakyrelu), 
        Dense(h_size, h_size, leakyrelu),
        Dense(h_size, h_size, leakyrelu), 
        Dense(h_size, nfreq),
        SinWaveEmbedder(latent_dim, nfreq),
        c -> 2.0f0 * sigmoid.(c))

    design_encoder = Waves.DesignEncoder(env.design_space, mlp, env.integration_steps)
    F = Waves.SinusoidalSource(latent_dim, nfreq, env.source.freq)
    dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, pml_width, pml_scale)
    iter = Integrator(runge_kutta, dyn, env.dt)
    return AcousticEnergyModel(wave_encoder, design_encoder, F, iter, get_dx(latent_dim))
end

# function (dyn::AcousticDynamics{OneDim})(x::AbstractArray, t::AbstractVector{Float32}, θ)
#     C, F, PML = θ
#     pml_scale = dyn.pml[[1]]
#     σ = pml_scale .* PML
#     ∇ = dyn.grad

#     U_tot = x[:, 1, :]
#     V_tot = x[:, 2, :]
#     U_inc = x[:, 3, :]
#     V_inc = x[:, 4, :]

#     c = C(t)
#     f = F(t)

#     dU_tot = (dyn.c0 ^ 2 * c) .* (∇ * V_tot) .- σ .* U_tot
#     dV_tot = ∇ * (U_tot .+ f) .- σ .* V_tot

#     dU_inc = (dyn.c0 ^ 2) * (∇ * V_inc) .- σ .* U_inc
#     dV_inc = ∇ * (U_inc .+ f) .- σ .* V_inc

#     return hcat(
#         Flux.unsqueeze(dU_tot, 2) .* dyn.bc,
#         Flux.unsqueeze(dV_tot, 2),
#         Flux.unsqueeze(dU_inc, 2) .* dyn.bc,
#         Flux.unsqueeze(dV_inc, 2),
#         )
# end

function get_parameters_and_initial_condition(model::AcousticEnergyModel, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    x = model.wave_encoder(s)
    z0 = x[:, 1:4, :]
    F = Source(x[:, 5, :], model.F.freq)
    PML = x[:, 6, :]
    C = model.design_encoder(s, a, t)
    θ = [C, F, PML]
    return z0, θ
end

function Waves.generate_latent_solution(model::AcousticEnergyModel, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z0, θ = get_parameters_and_initial_condition(model, s, a, t)
    return model.iter(z0, t, θ)
end

# function Waves.add_gradients(g1::AbstractArray{Float32}, g2::AbstractArray{Float32})
#     return g1 .+ g2
# end

# function Waves.add_gradients(g1::Vector, g2::Vector)
#     return Waves.add_gradients.(g1, g2)
# end
