using Waves
using Flux
using Optimisers
using CairoMakie
using ChainRulesCore
using BSON

function build_wave_encoder(;
            k::Tuple{Int, Int},
            in_channels::Int,
            activation::Function,
            h_size::Int,
            nfields::Int,
            nfreq::Int,
            latent_dim::OneDim)

    return Chain(
        Waves.TotalWaveInput(),
        Waves.ResidualBlock(k, in_channels, 32, activation),
        Waves.ResidualBlock(k, 32, 64, activation),
        Waves.ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        Flux.flatten,
        Dense(h_size, nfields * nfreq),
        b -> reshape(b, nfreq, nfields, :),
        Waves.SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(x[:, [1], :], x[:, [2], :] ./ WATER, x[:, [3], :], x[:, [4], :] ./ WATER)
    )
end

struct LearnableSourceLatentWaveDynamics <: AbstractDynamics
    ## general dynamics parameters
    grad::AbstractMatrix{Float32}
    C::Float32
    pml::AbstractVector{Float32}
    bc::AbstractVector{Float32}

    ## source parameters
    x::AbstractVector{Float32}

    mu::AbstractVector{Float32}
    mu_low::Float32
    mu_high::Float32

    sig::AbstractVector{Float32}
    sig_low::Float32
    sig_high::Float32

    a::AbstractVector{Float32}
    a_low::Float32
    a_high::Float32

    freq::Float32
end

Flux.@functor LearnableSourceLatentWaveDynamics
Flux.trainable(dyn::LearnableSourceLatentWaveDynamics) = (;dyn.mu, dyn.sig, dyn.a)

function LearnableSourceLatentWaveDynamics(
        dim::OneDim;
        C::Float32,
        pml_width::Float32,
        pml_scale::Float32,
        n::Int,
        mu_low::Float32, 
        mu_high::Float32,
        sig_low::Float32,
        sig_high::Float32,
        a_low::Float32,
        a_high::Float32,
        freq::Float32)

    grad = build_gradient(dim)
    pml = build_pml(dim, pml_width, pml_scale)
    bc = dirichlet(dim)

    mu = randn(Float32, n)
    sig = randn(Float32, n)
    a = randn(Float32, n)

    return LearnableSourceLatentWaveDynamics(
        grad, C, pml, bc, 
        dim.x,
        mu, mu_low, mu_high, 
        sig, sig_low, sig_high,
        a, a_low, a_high,
        freq)
end

function get_mu(dyn::LearnableSourceLatentWaveDynamics)
    return sigmoid(dyn.mu) * (dyn.mu_high - dyn.mu_low) .+ dyn.mu_low
end

function get_sig(dyn::LearnableSourceLatentWaveDynamics)
    return sigmoid(dyn.sig) * (dyn.sig_high - dyn.sig_low) .+ dyn.sig_low
end

function get_a(dyn::LearnableSourceLatentWaveDynamics)
    return sigmoid(dyn.a) * (dyn.a_high - dyn.a_low) .+ dyn.a_low
end

function get_force(dyn::LearnableSourceLatentWaveDynamics, t::AbstractVector{Float32})
    mu = permutedims(get_mu(dyn))
    sig = permutedims(get_sig(dyn))
    a = permutedims(get_a(dyn))
    x = dyn.x
    
    f = dropdims(
            sum(
                a .* exp.(-(x .- mu) .^ 2 ./ (2.0f0 * sig .^ 2)), 
                dims = 2),
            dims = 2)

    return f .* permutedims(sin.(2.0f0 * pi * dyn.freq * t))
end

function (dyn::LearnableSourceLatentWaveDynamics)(x::AbstractArray{Float32, 3}, t::AbstractVector{Float32})
    u_inc = x[:, 1, :]
    v_inc = x[:, 2, :]

    u_tot = x[:, 3, :]
    v_tot = x[:, 4, :]

    c = x[:, 5, :]
    dc = x[:, 6, :]

    f = get_force(dyn, t)

    du_inc = dyn.C ^ 2 * (dyn.grad * v_inc) .- dyn.pml .* u_inc
    du_tot = dyn.C ^ 2 * c .* (dyn.grad * v_tot) .- dyn.pml .* u_tot

    dv_inc = (dyn.grad * (u_inc .+ f)) .- dyn.pml .* v_inc
    dv_tot = (dyn.grad * (u_tot .+ f)) .- dyn.pml .* v_tot

    return hcat(
        Flux.unsqueeze(du_inc .* dyn.bc, dims = 2),
        Flux.unsqueeze(dv_inc, dims = 2),
        Flux.unsqueeze(du_tot .* dyn.bc, dims = 2),
        Flux.unsqueeze(dv_tot, dims = 2),
        Flux.unsqueeze(dc, dims = 2),
        Flux.unsqueeze(dc * 0.0f0, dims = 2))
end

struct DesignEncoder
    design_space::DesignSpace
    layers::Chain
end

Flux.@functor DesignEncoder
Flux.trainable(enc::DesignEncoder) = (;enc.layers)

function DesignEncoder(design_space::DesignSpace, h_size::Int, activation::Function, nfreq::Int, dim::OneDim)

    in_size = length(vec(design_space.low))

    layers = Chain(
        Dense(in_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, nfreq),
        Waves.SinWaveEmbedder(dim, nfreq),
        sigmoid
    )

    return DesignEncoder(design_space, layers)
end

"""
A single propagation step for the design, increments the design to the next one by applying the
action.
"""
function (enc::DesignEncoder)(d1::AbstractDesign, a::AbstractDesign)
    d2 = enc.design_space(d1, a)
    return (d2, d2)
end

function (enc::DesignEncoder)(d1::Vector{<: AbstractDesign}, a::Vector{<: AbstractDesign})
    d2 = model.design_space.(d1, a)
    return (d2, d2)
end

"""
Generates a sequence of wavespeed functions by evaluating the design after applying each
action.
"""
function (enc::DesignEncoder)(d::AbstractDesign, a::DesignSequence)
    recur = Flux.Recur(enc, d)
    design_sequence = vcat(d, [recur(action) for action in a])
    x = hcat(Waves.normalize.(design_sequence, [enc.design_space])...)[:, :, :]
    return enc.layers(x)
end

function (enc::DesignEncoder)(d::Vector{<: AbstractDesign}, a::Matrix{<: AbstractDesign})
    recur = Flux.Recur(enc, d)

    design_sequences = hcat(d, [recur(a[i, :]) for i in axes(a, 1)]...)
    x = Waves.normalize.(design_sequences, [enc.design_space])
    x_batch = cat([hcat(x[i, :]...) for i in axes(x, 1)]..., dims = 3)
    return enc.layers(x_batch)
end

(enc::DesignEncoder)(s::WaveEnvState, a::DesignSequence) = enc(s.design, a)
(enc::DesignEncoder)(states::Vector{WaveEnvState}, actions::Vector{<: DesignSequence}) = enc([s.design for s in states], hcat(actions...))


Flux.device!(0)
main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_pulse_intensity=10.0_freq=1000.0/"
data_path = joinpath(main_path, "episodes")

train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:2])
train_loader = Flux.DataLoader(prepare_data(train_data, 10), shuffle = true, batchsize = -1, partial = false)

env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
s, a, t, y = gpu(first(train_loader))

latent_dim = OneDim(15.0f0, 512)
wave_encoder = build_wave_encoder(;
    k = (3, 3), 
    in_channels = 3, 
    activation = leakyrelu, 
    h_size = 256,
    nfields = 4,
    nfreq = 50,
    latent_dim = latent_dim
    ) |> gpu

dyn = gpu(LearnableSourceLatentWaveDynamics(
    latent_dim,
    C = WATER,
    pml_width = 5.0f0,
    pml_scale = 10000.0f0,
    n = 1,
    mu_low = -10.0f0,
    mu_high = 10.0f0,
    sig_low = 0.2f0,
    sig_high = 1.0f0,
    a_low = -1.0f0,
    a_high = 1.0f0,
    freq = 1000.0f0))

iter = gpu(Integrator(runge_kutta, dyn, 0.0f0, 1f-5, 500))
tspan = gpu(collect(build_tspan(iter)))[:, :]
uv = wave_encoder(s)

design_encoder = gpu(DesignEncoder(env.design_space, 128, leakyrelu, 50, latent_dim))
c = design_encoder(s, a)

# c = gpu(ones(Float32, size(latent_dim)...))
# dc = gpu(zeros(Float32, size(latent_dim)...))
# x = hcat(uv, c, dc)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, latent_dim.x, cpu(c[:, 1]))
save("c.png", fig)



# opt_state = Optimisers.setup(Optimisers.Descent(1e-4), iter)

# ## rendering original source
# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)
# z = cpu(iter(x, tspan))
# record(fig, "unoptimized_latent.mp4", axes(tspan, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 3, 1, i], color = :blue)
#     lines!(ax, latent_dim.x, vec(cpu(get_force(iter.dynamics, tspan[i, :]))), color = :orange)
# end

# ## optimizing 
# for i in 1:20

#     cost, back = Flux.pullback(iter) do _iter
#         z = _iter(x, tspan)
#         total_energy = z[:, 3, 1, :] .^ 2
#         return -sum(total_energy)
#     end

#     println(cost)
#     gs = back(one(cost))[1]
#     opt_state, iter = Optimisers.update(opt_state, iter, gs)
# end

# ## plotting optimized source
# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)
# z = cpu(iter(x, tspan))
# record(fig, "optimized_latent.mp4", axes(tspan, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 3, 1, i], color = :blue)
#     lines!(ax, latent_dim.x, vec(cpu(get_force(iter.dynamics, tspan[i, :]))), color = :orange)
# end
