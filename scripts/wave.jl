using Waves
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie
using ReinforcementLearning
using LinearAlgebra

struct Episode{S, Y}
    s::Vector{S}
    a::Vector{<: AbstractDesign}
    t::Vector{Vector{Float32}}
    y::Vector{Y}
end

Base.length(ep::Episode) = length(ep.s)

function generate_episode!(policy::AbstractPolicy, env::WaveEnv)
    s = WaveEnvState[]
    a = AbstractDesign[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]

    reset!(env)
    while !is_terminated(env)
        push!(s, state(env))
        action = policy(env)
        push!(a, cpu(action))
        push!(t, build_tspan(env))
        env(action)
        push!(y, cpu(env.signal))
        println(env.time_step)
    end

    return Episode(s, a, t, y)
end

function prepare_data(ep::Episode{S, Matrix{Float32}}, horizon::Int) where S
    s = S[]
    a = Vector{<: AbstractDesign}[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]
    
    n = horizon - 1
    for i in 1:(length(ep) - n)
        boundary = i + n
        push!(s, ep.s[i])
        push!(a, ep.a[i:boundary])
        push!(t, flatten_repeated_last_dim(hcat(ep.t[i:boundary]...)))

        signal = cat(ep.y[i:boundary]..., dims = 3)
        signal = permutedims(flatten_repeated_last_dim(permutedims(signal, (2, 1, 3))))
        push!(y, signal)
    end

    return s, a, t, y
end

function prepare_data(eps::Vector{Episode{S, Y}}, horizon::Int) where {S, Y}
    return vcat.(prepare_data.(eps, horizon)...)
end


"""
mask:   (sequence x sequence)
x:      (sequence x batch)
y:      (features x sequence x batch)
"""
struct PolynomialInterpolation
    mask::AbstractMatrix
    x::AbstractMatrix
    y::AbstractArray
end

Flux.@functor PolynomialInterpolation
Flux.trainable(interp::PolynomialInterpolation) = (interp.x, interp.y)

function PolynomialInterpolation(x::AbstractArray, y::AbstractArray)
    mask = I(size(x, 1))
    return PolynomialInterpolation(mask, x, y)
end

function (interp::PolynomialInterpolation)(t::AbstractVector{Float32})

    scale = maximum(interp.x, dims = 1)

    n = interp.mask .+ .!interp.mask .* Flux.unsqueeze((interp.x .- t'), 2)
    numer = prod(n, dims = 1)

    T = Flux.unsqueeze(interp.x, 2) .- Flux.unsqueeze(interp.x, 1)
    d = T .+ interp.mask
    denom = prod(d, dims = 1)
    coef = numer ./ denom

    return dropdims(sum(interp.y .* coef, dims = 2), dims = 2)
end

function (dyn::AcousticDynamics{OneDim})(x::AbstractArray, t::AbstractVector{Float32}, θ)
    C, F = θ
    U = x[:, 1, :]
    V = x[:, 2, :]
    ∇ = dyn.grad

    b = dyn.c0 .^ 2 * C(t)

    dU = b .* (∇ * V) .- dyn.pml .* U
    dV = ∇ * (U .+ F(t)) .- dyn.pml .* V
    return hcat(dU .* dyn.bc, dV)
end

# struct GaussianSource <: AbstractSource

# end

dim = TwoDim(15.0f0, 700)
n = 10
μ = zeros(Float32, n, 2)
μ[:, 1] .= -10.0f0
μ[:, 2] .= range(-2.0f0, 2.0f0, n)

σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 0.3f0
pulse = build_normal(build_grid(dim), μ, σ, a)
source = Source(pulse, 1000.0f0)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = source,
    integration_steps = 100,
    actions = 5))

policy = RandomDesignPolicy(action_space(env))
# render!(policy, env, path = "vid.mp4")
# ep = generate_episode!(policy, env)

data = Flux.DataLoader(prepare_data([ep, ep], 3), batchsize = 6, shuffle = true, partial = false)
s, a, t, y = Flux.batch.(first(data))
t′ = t[1:env.integration_steps:end, :]

latent_dim = OneDim(15.0f0, 512)
nfreq = 50
emb = Waves.SinWaveEmbedder(latent_dim, nfreq)
w = emb(randn(Float32, nfreq, size(t′)...) * 5.0f0)

interp = PolynomialInterpolation(t′, w)
C = Chain(interp, sigmoid)

# dyn = AcousticDynamics(latent_dim, WATER, 2.0f0, 10000.0f0)
# iter = Integrator(runge_kutta, dyn, env.dt)

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, 0.0f0, 1.0f0)

record(fig, "interp.mp4", axes(t, 1)) do i
    empty!(ax)

    w = C(t[i, :])

    lines!(ax, latent_dim.x, w[:, 1], color = :blue)
    lines!(ax, latent_dim.x, w[:, 2], color = :orange)
    lines!(ax, latent_dim.x, w[:, 3], color = :red)
end