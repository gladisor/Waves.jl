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
    mlp = Chain(flatten, build_mlp(4 * size(dim, 1), h_size, n_h, 1, act), vec)

    return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
end

elu_speed(c) = 1.0f0 .+ elu.(c)
sigmoid_speed(c, scale = 5.0f0) = 5.0f0 * sigmoid.(c)

data_path = "data/hexagon_large_grid"
@time states, actions, tspans, sigmas = prepare_data(
    EpisodeData(path = joinpath(data_path, "episode1/episode.bson")), 3)
;

idx = 35
s = gpu(states[idx])
a = gpu(actions[idx])
tspan = gpu(tspans[idx])
sigma = gpu(sigmas[idx])
design_input = vcat(vec(s.design), vec(a[1])) |> gpu

# dim = OneDim(15.0f0, 512)
# @time model = build_hypernet_wave_control_model(
#     dim,
#     design_input_size = length(design_input),
#     nfreq = 2,
#     h_size = 512,
#     n_h = 2,
#     act = leakyrelu,
#     speed_activation = softplus,
#     ambient_speed = AIR,
#     freq = 200.0f0,
#     pml_width = 5.0f0,
#     pml_scale = 5000.0f0,
#     dt = 5e-5,
#     steps = 100) |> gpu

@time visualize!(model, dim, s, a, tspan, sigma, path = "latent")
# function train_loop(;
#         model::WaveControlModel;
#         train_steps::Int, ## only really effects validation
#         val_steps::Int,
#         epochs::Int
#         decay_rate::Float32,
#         decay_steps::Float32,
#         opt,
#         lr::Float32,
#         )     
# end