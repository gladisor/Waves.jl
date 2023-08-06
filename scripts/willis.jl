using Waves
using Waves: build_tspan, update_design
using Flux
using ReinforcementLearning
using DataStructures: CircularBuffer
using CairoMakie
using Interpolations: linear_interpolation
using Images: imresize
Flux.CUDA.allowscalar(false)
include("improved_model.jl")

function normal(grid::AbstractArray{Float32, 3}, x::Float32, y::Float32, σx::Float32, σy::Float32)
    μ = reshape([x, y], 1, 1, 2)
    σ = reshape([σx, σy], 1, 1, 2)

    exponent = dropdims(sum(
            ((grid .- μ) .^ 2) ./ (2.0f0 * σ .^ 2),
        dims = 3), dims = 3)

    f = exp.(- exponent) / (2.0f0 * π * prod(σ))

    return f
end

struct WillisDynamics <: AbstractDynamics
    ambient_speed::Float32
    design::DesignInterpolator
    source¹::AbstractSource
    source²::AbstractSource
    grid::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor WillisDynamics
Flux.trainable(::WillisDynamics) = (;)

function WillisDynamics(dim::AbstractDim; 
        ambient_speed::Float32,
        pml_width::Float32,
        pml_scale::Float32,
        design::AbstractDesign = NoDesign(),
        source¹::AbstractSource = NoSource(),
        source²::AbstractSource = NoSource()
        )

    design = DesignInterpolator(design)
    grid = build_grid(dim)
    grad = build_gradient(dim)
    pml = build_pml(dim, pml_width, pml_scale)
    bc = dirichlet(dim)

    return WillisDynamics(ambient_speed, design, source¹, source², grid, grad, pml, bc)
end

function pml_wave_dynamics(
        wave::AbstractArray{Float32, 3}, 
        b::Union{Float32, AbstractMatrix{Float32}}, 
        ∇::AbstractMatrix{Float32}, 
        σx::AbstractMatrix{Float32}, 
        σy::AbstractMatrix{Float32}, 
        f::AbstractMatrix{Float32}, 
        bc::AbstractMatrix{Float32})

    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    Vxx = ∂x(∇, Vx)
    Vyy = ∂y(∇, Vy)
    Ux = ∂x(∇, U .+ f)
    Uy = ∂y(∇, U .+ f)

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U
    return cat(bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function (dyn::WillisDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)
    wave¹_inc = wave[:, :, 1:6]
    wave¹_tot = wave[:, :, 7:12]
    wave²_inc = wave[:, :, 13:18]
    wave²_tot = wave[:, :, 19:24]

    C = speed(dyn.design(t), dyn.grid, dyn.ambient_speed)
    b = C .^ 2
    b0 = dyn.ambient_speed ^ 2

    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'
    f¹ = dyn.source¹(t)
    f² = dyn.source²(t)

    dwave¹_inc = pml_wave_dynamics(wave¹_inc, b0, ∇, σx, σy, f¹, dyn.bc)
    dwave¹_tot = pml_wave_dynamics(wave¹_tot, b, ∇, σx, σy, f¹, dyn.bc)
    dwave²_inc = pml_wave_dynamics(wave²_inc, b0, ∇, σx, σy, f², dyn.bc)
    dwave²_tot = pml_wave_dynamics(wave²_tot, b, ∇, σx, σy, f², dyn.bc)

    return cat(dwave¹_inc, dwave¹_tot, dwave²_inc, dwave²_tot, dims = 3)
end

function Waves.update_design(dyn::WillisDynamics, interp::DesignInterpolator)
    return WillisDynamics(dyn.ambient_speed, interp, dyn.source¹, dyn.source², dyn.grid, dyn.grad, dyn.pml, dyn.bc)
end

# struct LearnableGaussianSource <: AbstractSource
#     mu::AbstractArray{Float32}
#     logsigma::AbstractArray{Float32}
#     grid::AbstractArray

#     mu_max::Float32
#     sigma_max::Float32
#     freq::Float32
# end

# function (source::LearnableGaussianSource)(t)
#     mu = tanh.(source.mu) * source.mu_max
#     sigma = clamp.(exp.(source.logsigma), 0.1f0, source.sigma_max)
#     f = exp.(-0.5f0 * ((source.grid .- mu) ./ sigma) .^ 2)# ./ (sigma * sqrt(2.0f0 * pi))
#     return f .* sin.(2.0f0 * pi * source.freq * t')
# end

# Flux.@functor LearnableGaussianSource
# Flux.trainable(source::LearnableGaussianSource) = (;source.mu, source.logsigma)

struct LatentWillisDynamics <: AbstractDynamics
    dyn::LatentDynamics
end

Flux.@functor LatentWillisDynamics
# Flux.trainable(::LatentWillisDynamics) = (;)

LatentWillisDynamics(args...; kwargs...) = LatentWillisDynamics(LatentDynamics(args...; kwargs...))

function pml_wave_dynamics(
        wave::AbstractArray{Float32, 3},
        f::AbstractMatrix{Float32},
        c,
        C0::Float32,
        ∇::AbstractMatrix{Float32},
        σx::AbstractVector{Float32},
        bc::AbstractVector{Float32})

    u_inc = wave[:, 1, :]
    u_tot = wave[:, 2, :]
    v_inc = wave[:, 3, :]
    v_tot = wave[:, 4, :]

    du_inc = C0 ^ 2 * (∇ * v_inc) .- σx .* u_inc
    du_tot = C0 ^ 2 * c .* (∇ * v_tot) .- σx .* u_tot
    dv_inc = (∇ * (u_inc .+ f)) .- σx .* v_inc
    dv_tot = (∇ * (u_tot .+ f)) .- σx .* v_tot

    return hcat(
        Flux.unsqueeze(du_inc .* bc, dims = 2),
        Flux.unsqueeze(du_tot .* bc, dims = 2),
        Flux.unsqueeze(dv_inc, dims = 2),
        Flux.unsqueeze(dv_tot, dims = 2),
        )
end

function (dyn::LatentWillisDynamics)(wave::AbstractArray{Float32, 3}, t::AbstractVector{Float32})

    u_left = hcat(
        wave[:, 1:5, :],
        wave[:, end-1:end, :]
        )

    u_right = hcat(
        wave[:, 6:10, :],
        wave[:, end-1:end, :]
    )

    du_left = dyn.dyn(u_left, t)
    du_right = dyn.dyn(u_right, t)

    return hcat(
        du_left[:, 1:end-2, :],
        du_right
    )

    # u_left = wave[:, 1:4, :]
    # f1 = wave[:, 5, :]

    # u_right = wave[:, 6:9, :]
    # f2 = wave[:, 10, :]

    # c = wave[:, 11, :]
    # dc = wave[:, 12, :]

    # force1 = f1 .* sin.(2.0f0 * pi * dyn.dyn.freq * permutedims(t))
    # force2 = f2 .* sin.(2.0f0 * pi * dyn.dyn.freq * permutedims(t))

    # # C0 = dyn.dyn.C0
    # # ∇ = dyn.dyn.grad
    # # σx = dyn.dyn.pml
    # # bc = dyn.dyn.bc
    
    # du_left = pml_wave_dynamics(u_left, force1, c, dyn.dyn.C0, dyn.dyn.grad, dyn.dyn.pml, dyn.dyn.bc)
    # du_right = pml_wave_dynamics(u_right, force2, c, dyn.dyn.C0, dyn.dyn.grad, dyn.dyn.pml, dyn.dyn.bc)

    # return hcat(
    #     du_left,
    #     Flux.unsqueeze(f1 * 0.0f0, dims = 2),

    #     du_right,
    #     Flux.unsqueeze(f2 * 0.0f0, dims = 2),

    #     Flux.unsqueeze(dc, dims = 2),
    #     Flux.unsqueeze(dc * 0.0f0, dims = 2)
    #     )
end

function encode_wave(model::ScatteredEnergyModel{LatentWillisDynamics}, s::WaveEnvState)
    wave_left = s.wave_total[:, :, 1:3, :]
    wave_right = s.wave_total[:, :, 4:6, :]
    return hcat(model.wave_encoder(wave_left), model.wave_encoder(wave_right))
end

function encode_wave(model::ScatteredEnergyModel{LatentWillisDynamics}, states::Vector{WaveEnvState})
    wave_left = cat([s.wave_total[:, :, 1:3] for s in states]..., dims = 4)
    wave_right = cat([s.wave_total[:, :, 4:6] for s in states]..., dims = 4)
    return hcat(model.wave_encoder(wave_left), model.wave_encoder(wave_right))
end

function decode_signal(model::ScatteredEnergyModel{LatentWillisDynamics}, z::AbstractArray{Float32, 4})
    z_left = hcat(z[:, 1:5, :, :], z[:, end-1:end, :, :])
    z_right = hcat(z[:, 6:10, :, :], z[:, end-1:end, :, :])
    return hcat(model.mlp(z_left), model.mlp(z_right))
end

mutable struct WillisEnv <: AbstractEnv
    dim::TwoDim
    design_space::DesignSpace
    action_speed::Float32

    dynamics::WillisDynamics
    actions::Int
    integration_steps::Int
    dt::Float32
    image_resolution::Tuple{Int, Int}
    focal::AbstractArray{Float32, 3}

    wave::CircularBuffer{AbstractArray{Float32, 3}}
    signal::AbstractMatrix{Float32}
    time_step::Int
end

Flux.@functor WillisEnv
Flux.trainable(::WillisEnv) = (;)

function WillisEnv(dim::TwoDim; design_space::DesignSpace, action_speed::Float32, dynamics::WillisDynamics, actions::Int, integration_steps::Int, dt::Float32, image_resolution::Tuple{Int, Int} = (128, 128))
    wave = CircularBuffer{AbstractArray{Float32, 3}}(3)
    fill!(wave, gpu(build_wave(dim, fields = 24)))

    grid = build_grid(dim)
    f1 = normal(grid, 10.0f0, 0.0f0, 1.0f0, 1.0f0)
    f2 = normal(grid, -10.0f0, 0.0f0, 1.0f0, 1.0f0)
    focal = cat(f1, f2, dims = 3)

    signal = zeros(Float32, 2, 1 + integration_steps * actions)
    return WillisEnv(dim, design_space, action_speed, dynamics, actions, integration_steps, dt, image_resolution, focal, wave, signal, 0)
end

function Base.time(env::WillisEnv)
    return env.time_step * env.dt
end

function Waves.build_tspan(env::WillisEnv)
    return build_tspan(time(env), env.dt, env.integration_steps)
end

function RLBase.is_terminated(env::WillisEnv)
    return env.time_step >= env.actions * env.integration_steps
end

function RLBase.state(env::WillisEnv)

    x = cpu(cat(
        convert(Vector{AbstractArray{Float32, 3}}, env.wave)...,
        dims = 4))

    u_tot = cat(
        x[:, :, 7, :],
        x[:, :, 19, :],
        dims = 3
    )

    return WaveEnvState(
        cpu(env.dim),
        build_tspan(env), ## forward looking tspan
        imresize(u_tot, env.image_resolution),
        cpu(env.dynamics.design(time(env)))
        )
end

function RLBase.state_space(env::WillisEnv)
    return state(env)
end

function RLBase.action_space(env::WillisEnv)
    return build_action_space(rand(env.design_space), env.action_speed * env.dt * env.integration_steps)
end

function RLBase.reset!(env::WillisEnv)
    env.time_step = 0
    z = gpu(zeros(Float32, size(env.wave[end])))
    empty!(env.wave)
    fill!(env.wave, z)

    design = DesignInterpolator(rand(env.design_space))
    env.dynamics = update_design(env.dynamics, design)
    env.signal *= 0.0f0
    return nothing
end

function RLBase.reward(env::WillisEnv)
    return sum(env.signal)
end

function (policy::RandomDesignPolicy)(::WillisEnv)
    return rand(policy.a_space)
end

function (env::WillisEnv)(action::AbstractDesign)
    ti = time(env)
    tspan = Waves.build_tspan(env)

    design = env.dynamics.design(ti)
    interp = DesignInterpolator(design, env.design_space(design, gpu(action)), ti, tspan[end])
    env.dynamics = update_design(env.dynamics, interp)

    iter = Integrator(runge_kutta, env.dynamics, ti, env.dt, env.integration_steps)

    u = iter(env.wave[end])
    push!(env.wave, u[:, :, :, end])

    u_inc = u[:, :, [1, 13], :] ## incident fields
    u_tot = u[:, :, [7, 19], :] ## total fields
    u_sc = (u_tot .- u_inc) .^ 2

    dim = cpu(env.dim)
    dx = Flux.mean(diff(dim.x))
    dy = Flux.mean(diff(dim.y))

    env.signal = dropdims(sum(u_sc .* env.focal[:, :, :, :], dims = (1, 2)), dims = (1, 2)) * dx * dy
    env.time_step += env.integration_steps
    return tspan, imresize(cpu(u_tot), env.image_resolution)
end

struct WillisData
    states::Vector{WaveEnvState}
    actions::Vector{<: AbstractDesign}
    tspans::Vector{Vector{Float32}}
    signals::Vector{AbstractMatrix{Float32}}
end

Flux.@functor WillisData
Base.length(episode::WillisData) = length(episode.states)

function WillisData(policy::AbstractPolicy, env::WillisEnv)
    states = WaveEnvState[]
    actions = AbstractDesign[]
    tspans = Vector{Float32}[]
    signals = Matrix{Float32}[]

    reset!(env)
    while !is_terminated(env)
        action = policy(env)
        
        push!(states, state(env))
        push!(actions, action)
        push!(tspans, build_tspan(env))
        @time env(action)
        push!(signals, cpu(env.signal))
    end

    return WillisData(states, actions, tspans, signals)
end

function render!(policy::AbstractPolicy, env::WillisEnv; path::String)

    reset!(env)
    t = Vector{Float32}[]
    u = AbstractArray{Float32, 4}[]
    tau = Float32[time(env)]
    design = AbstractDesign[cpu(env.dynamics.design(time(env)))]

    while !is_terminated(env)
        @time t_i, u_i = env(policy(env))
        push!(t, t_i)
        push!(u, u_i)
        push!(tau, time(env))
        push!(design, cpu(env.dynamics.design(time(env))))
    end

    t = flatten_repeated_last_dim(hcat(t...))
    u = flatten_repeated_last_dim(cat(u..., dims = 5))
    z_extreme = maximum(abs.(u))
    u = linear_interpolation(t, Flux.unbatch(u))
    design = linear_interpolation(tau, design)

    timesteps = 1:10:length(t)

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0f0, xlabel = "Space (m)", ylabel = "Space (m)")
    ax2 = Axis(fig[1, 2], aspect = 1.0f0, xlabel = "Space (m)", ylabel = "Space (m)")

    dim = cpu(env.dim)
    kwargs = Dict(:colormap => :ice, :colorrange => (-1.5, 1.5))

    CairoMakie.record(fig, path, timesteps) do i
        u_i = u(t[i])
        d_i = design(t[i])

        display(size(u_i))
        empty!(ax1)
        empty!(ax2)
        heatmap!(ax1, dim.x, dim.y, u_i[:, :, 1]; kwargs...)
        mesh!(ax1, d_i)
        heatmap!(ax2, dim.x, dim.y, u_i[:, :, 2]; kwargs...)
        mesh!(ax2, d_i)
    end
end

Flux.device!(0)
dim = TwoDim(20.0f0, 512)
grid = build_grid(dim)
pulse¹ = exp.(- 2.0f0 * (grid[:, :, 1] .- -15.0f0) .^ 2)
pulse² = exp.(- 2.0f0 * (grid[:, :, 1] .- 15.0f0) .^ 2)

source¹ = Source(pulse¹, freq = 500.0f0)
source² = Source(pulse², freq = 500.0f0)

rot = Float32.(Waves.build_2d_rotation_matrix(30))
pos = vcat(
        [0.0f0, 0.0f0]',
        Waves.hexagon_ring(3.5f0),
        Waves.hexagon_ring(4.75f0) * rot,
        Waves.hexagon_ring(6.0f0))

DESIGN_SPEED = AIR * 3
r_low = fill(0.2f0, size(pos, 1))
r_high = fill(1.0f0, size(pos, 1))
c = fill(DESIGN_SPEED, size(pos, 1))

design_low = AdjustableRadiiScatterers(Cylinders(pos, r_low, c))
design_high = AdjustableRadiiScatterers(Cylinders(pos, r_high, c))
ds = DesignSpace(design_low, design_high)

dynamics = WillisDynamics(
    dim, 
    ambient_speed = WATER,
    pml_width = 5.0f0, 
    pml_scale = 10000.0f0,
    source¹ = source¹,
    source² = source²,
    design = rand(ds)
    )

env = gpu(WillisEnv(
    dim,
    design_space = ds,
    action_speed = 200.0f0,
    dynamics = dynamics,
    actions = 20,
    integration_steps = 100,
    dt = 1.0f-5,
    image_resolution = (128, 128))
    )

reset!(env)
policy = RandomDesignPolicy(action_space(env))
# episode = WillisData(policy, env)

latent_elements = 512
latent_dim = OneDim(20.0f0, latent_elements)
pml_width = 5.0f0
pml_scale = 10000.0f0
activation = leakyrelu
h_size = 256
nfreq = 50 #200
k_size = 2

wave_encoder = build_split_hypernet_wave_encoder(latent_dim = latent_dim, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dyn = LatentWillisDynamics(latent_dim, ambient_speed = env.dynamics.ambient_speed, freq = env.dynamics.source¹.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dyn, 0.0f0, env.dt, env.integration_steps)
mlp = build_scattered_wave_decoder(latent_elements, h_size, k_size, activation)
println("Constructing Model")
model = gpu(ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp))

s = gpu(episode.states[end-2])
a = gpu(episode.actions[end-2:end])
t = gpu(hcat(episode.tspans[end-2:end]...))
signal = gpu(cat(episode.signals[end-2:end]..., dims = 3))
signal = permutedims(flatten_repeated_last_dim(signal))[:, :, :]
# pred_signal = model(s, a, t)

opt = Optimisers.Adam(1e-3)
opt_state = Optimisers.setup(opt, model)

loss, back = Flux.pullback(m -> Flux.mse(m(s, a, t), signal), model)
gs = back(one(loss))[1]
opt_state, model = Optimisers.update(opt_state, model, gs)

# z = cpu(generate_latent_solution(model, s, a, t))
# z_sc_left = z[:, 2, :, :] .- z[:, 1, :, :]
# z_sc_right = z[:, 7, :, :] .- z[:, 6, :, :]

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)

# CairoMakie.record(fig, "latent.mp4", axes(z, 3)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z_sc_left[:, i, 1], color = :blue)
#     lines!(ax, latent_dim.x, z_sc_right[:, i, 1], color = :red)
# end
