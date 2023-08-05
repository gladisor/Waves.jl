using Waves
using Waves: build_tspan, update_design
using Flux
using ReinforcementLearning
using DataStructures: CircularBuffer
using CairoMakie
using Interpolations: linear_interpolation

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

function pml_wave_dynamics(wave, b, ∇, σx, σy, f, bc)

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

mutable struct WillisEnv <: AbstractEnv
    dim::TwoDim
    design_space::DesignSpace
    action_speed::Float32

    wave::CircularBuffer{AbstractArray{Float32, 3}}
    dynamics::WillisDynamics

    image_resolution::Tuple{Int, Int}

    dt::Float32
    signal::Vector{Float32}
    time_step::Int
    integration_steps::Int
    actions::Int
end

Flux.@functor WillisEnv
Flux.trainable(::WillisEnv) = (;)

function WillisEnv(dim::TwoDim; design_space::DesignSpace, action_speed::Float32, dynamics::WillisDynamics, actions::Int, integration_steps::Int, dt::Float32, image_resolution::Tuple{Int, Int} = (128, 128))
    wave = CircularBuffer{AbstractArray{Float32, 3}}(3)
    fill!(wave, gpu(build_wave(dim, fields = 24)))
    signal = zeros(Float32, 1 + integration_steps * actions)
    return WillisEnv(dim, design_space, action_speed, wave, dynamics, image_resolution, dt, signal, 0, integration_steps, actions)
end

function Base.time(env::WillisEnv)
    return env.time_step * env.dt
end

function RLBase.is_terminated(env::WillisEnv)
    return env.time_step >= env.actions * env.integration_steps
end

function RLBase.state(env::WillisEnv)
    return nothing
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

    env.dynamics = update_design(env.dynamics, DesignInterpolator(rand(env.design_space)))
    # env.dynamics = WillisDynamics(
    #     env.dynamics.ambient_speed,
    #     DesignInterpolator(rand(env.design_space)),
    #     env.dynamics.source¹,
    #     env.dynamics.source²,
    #     env.dynamics.grid,
    #     env.dynamics.grad,
    #     env.dynamics.pml,
    #     env.dynamics.bc)
    env.signal = zeros(Float32, env.integration_steps + 1)
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
    tspan = Waves.build_tspan(ti, env.dt, env.integration_steps)

    design = env.dynamics.design(ti)
    interp = DesignInterpolator(design, env.design_space(design, gpu(action)), ti, tspan[end])
    env.dynamics = update_design(env.dynamics, interp)

    iter = Integrator(runge_kutta, env.dynamics, ti, env.dt, env.integration_steps)

    u = iter(env.wave[end])
    push!(env.wave, u[:, :, :, end])
    # env.σ = sum.(energy.(displacement.(u_scattered))) / 64.0f0
    env.time_step += env.integration_steps
    return tspan, u
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
        Waves.hexagon_ring(6.0f0)
    )

DESIGN_SPEED = 3 * AIR
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
    design = rand(ds))

env = gpu(WillisEnv(
    dim,
    dynamics = dynamics,
    design_space = ds,
    action_speed = 200.0f0,
    actions = 10,
    integration_steps = 100,
    dt = 1.0f-5))

reset!(env)
policy = RandomDesignPolicy(action_space(env))

# @time tspan1, u1 = cpu(env(policy(env)))
# @time tspan2, u2 = cpu(env(policy(env)))
# @time tspan3, u3 = cpu(env(policy(env)))
# @time tspan4, u4 = cpu(env(policy(env)))
# @time tspan5, u5 = cpu(env(policy(env)))
# @time tspan6, u6 = cpu(env(policy(env)))
# @time tspan7, u7 = cpu(env(policy(env)))
# @time tspan8, u8 = cpu(env(policy(env)))
# @time tspan9, u9 = cpu(env(policy(env)))
# @time tspan10, u10 = cpu(env(policy(env)))

# @time tspan11, u11 = cpu(env(policy(env)))
# @time tspan12, u12 = cpu(env(policy(env)))
# @time tspan13, u13 = cpu(env(policy(env)))
# @time tspan14, u14 = cpu(env(policy(env)))
# @time tspan15, u15 = cpu(env(policy(env)))
# @time tspan16, u16 = cpu(env(policy(env)))
# @time tspan17, u17 = cpu(env(policy(env)))
# @time tspan18, u18 = cpu(env(policy(env)))
# @time tspan19, u19 = cpu(env(policy(env)))
# @time tspan20, u20 = cpu(env(policy(env)))

# tspan = hcat(
#     tspan1, 
#     tspan2, 
#     tspan3, 
#     tspan4, 
#     tspan5, 
#     tspan6,
#     tspan7,
#     tspan8,
#     tspan9,
#     tspan10,
#     tspan11,
#     tspan12,
#     tspan13,
#     tspan14,
#     tspan15,
#     tspan16,
#     tspan17,
#     tspan18,
#     tspan19,
#     tspan20
#     )

# u = cat(
#     u1, 
#     u2, 
#     u3, 
#     u4, 
#     u5, 
#     u6,
#     u7,
#     u8,
#     u9,
#     u10,
#     u11, 
#     u12, 
#     u13, 
#     u14, 
#     u15, 
#     u16,
#     u17,
#     u18,
#     u19,
#     u20,
#     dims = 5)

# t = flatten_repeated_last_dim(tspan)

# u_inc_l = flatten_repeated_last_dim(u[:, :, 1, :, :])
# u_tot_l = flatten_repeated_last_dim(u[:, :, 7, :, :])
# u_sc_l = u_tot_l .- u_inc_l

# u_inc_r = flatten_repeated_last_dim(u[:, :, 13, :, :])
# u_tot_r = flatten_repeated_last_dim(u[:, :, 19, :, :])
# u_sc_r = u_tot_r .- u_inc_r

u_tot_l_reduced = imresize(u_tot_l, env.image_resolution)
u_tot_r_reduced = imresize(u_tot_r, env.image_resolution)

u_interp_l = linear_interpolation(t, Flux.unbatch(u_tot_l_reduced))
u_interp_r = linear_interpolation(t, Flux.unbatch(u_tot_r_reduced))

f1 = normal(grid, 10.0f0, 0.0f0, 1.0f0, 1.0f0)[:, :, :]
f2 = normal(grid, -10.0f0, 0.0f0, 1.0f0, 1.0f0)[:, :, :]

signal1 = vec(sum(u_sc_l .^ 2 .* f1, dims = (1, 2)))
signal2 = vec(sum(u_sc_r .^ 2 .* f2, dims = (1, 2)))

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, signal1)
lines!(ax, signal2)
save("signal.png", fig)

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)
heatmap!(ax, dim.x, dim.y, f1[:, :, 1] .+ f2[:, :, 1], colormap = :ice)
save("normal.png", fig)

fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0)
ax2 = Axis(fig[1, 2], aspect = 1.0)

CairoMakie.record(fig, "test.mp4", 1:3:length(t)) do i
    empty!(ax1)
    heatmap!(ax1, dim.x, dim.y, u_interp_l(t[i]), colormap = :ice, colorrange = (-2.0f0, 2.0f0))

    empty!(ax2)
    heatmap!(ax2, dim.x, dim.y, u_interp_r(t[i]), colormap = :ice, colorrange = (-2.0f0, 2.0f0))
end
