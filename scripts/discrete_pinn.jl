using Waves, CairoMakie, Flux, Optimisers, BSON

struct SimpleWave <: AbstractDynamics
    grad::AbstractMatrix{Float32}
    c0::Float32
    bc::AbstractArray{Float32}
end

Flux.@functor SimpleWave
Flux.trainable(::SimpleWave) = (;)

function (dyn::SimpleWave)(x::AbstractArray{Float32, 3}, t::AbstractVector{Float32}, theta)

    F = theta
    f = F(t)

    u = x[:, 1, :]
    v = x[:, 2, :]
    u_t = WATER ^ 2 * dyn.grad * v
    v_t = dyn.grad * (u .+ f)

    return hcat(
        Flux.unsqueeze(u_t, dims = 2) .* dyn.bc,
        Flux.unsqueeze(v_t, dims = 2)
        )
end

function runge_kutta_stages(f::AbstractDynamics, u::AbstractArray{Float32}, t, θ, dt::Float32)
    k1 = f(u,                    t,               θ)
    k2 = f(u .+ 0.5f0 * dt * k1, t .+ 0.5f0 * dt, θ)
    k3 = f(u .+ 0.5f0 * dt * k2, t .+ 0.5f0 * dt, θ)
    k4 = f(u .+ dt * k3,         t .+ dt,         θ)
    
    return k1
end

Flux.device!(0)

dt = 1f-5
integration_steps = 200
latent_gs = 10.0f0
elements = 1024
latent_dim = OneDim(latent_gs, elements)
dx = get_dx(latent_dim)

dyn = gpu(SimpleWave(
    build_gradient(latent_dim), 
    WATER, 
    build_dirichlet(latent_dim)))

iter = gpu(Integrator(runge_kutta, dyn, dt))
t = build_tspan(0.0f0, dt, integration_steps)

wave = gpu(zeros(Float32, elements, 2, 1))
t = gpu(t[:, :])

F = gpu(Source(
    build_normal(latent_dim.x, [0.0f0], [0.3f0], [1.0f0]),
    1000.0f0))

z = iter(wave, t, F)

# z_batch = reshape(z, (elements, 2, length(t)))
# next_z = runge_kutta_stages(dyn, z_batch, vec(t), F, dt)
# Flux.mse(z_batch[:, :, 2:end], next_z[:, :, 1:end-1])

u = z[:, 1, 1, :]
# v = z[:, 2, 1, :]
fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)
record(fig, "vid.mp4", axes(u, 2)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, cpu(u[:, i]), color = :blue)
end