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

function build_pinn_grid(latent_dim::OneDim, t::Vector{Float32})
    latent_gs = maximum(latent_dim.x)
    elements = length(latent_dim.x)
    dt = Flux.mean(diff(vec(t)))
    integration_steps = length(t)

    t_grid = repeat(reshape(t, 1, 1, integration_steps), 1, size(latent_dim.x, 1), 1) / (dt * integration_steps)
    x_grid = gpu(repeat(reshape(latent_dim.x, 1, elements, 1), 1, 1, integration_steps) / latent_gs)
    pinn_grid = gpu(vcat(x_grid, t_grid))
    return reshape(pinn_grid, 2, :, 1)
end

"""
Expects input of size: (2, elements, steps, batch)
"""
function compute_wave_physics_loss(z::AbstractArray{Float32, 4}, ∇ₓ, ∇ₜ, f, bc)
    u = z[1, :, :, 1]
    v = z[2, :, :, 1]

    u_t = (∇ₜ * u')' .* bc
    N_u = WATER ^ 2 * ∇ₓ * v .* bc
    v_t = (∇ₜ * v')'
    N_v = ∇ₓ * (u .+ f)

    return Flux.mse(u_t, N_u) + Flux.mse(v_t, N_v)
end

Flux.device!(0)

dt = 1f-6
integration_steps = 1000
latent_gs = 5.0f0
elements = 100
latent_dim = OneDim(latent_gs, elements)
dx = get_dx(latent_dim)

dyn = gpu(SimpleWave(
    build_gradient(latent_dim), 
    WATER, 
    build_dirichlet(latent_dim)))

iter = gpu(Integrator(runge_kutta, dyn, dt))
t = build_tspan(0.0f0, dt, integration_steps)

pinn_grid = build_pinn_grid(latent_dim, t)

wave = zeros(Float32, elements, 2, 1)
wave = gpu(wave)

grad_x = gpu(Waves.gradient(latent_dim.x))
grad_t = gpu(Waves.gradient(t))

t = gpu(t[:, :])

F = gpu(Source(
    build_normal(latent_dim.x, [0.0f0], [0.3f0], [1.0f0]),
    1000.0f0))
f = hcat([F(t[i, :]) for i in axes(t, 1)]...)

z = iter(wave, t, F)
u = z[:, 1, 1, :]
# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)
# record(fig, "vid.mp4", axes(u, 2)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, cpu(u[:, i]), color = :blue)
# end


energy = dropdims(sum(u .^ 2, dims = 1) * dx, dims = 1)

h_size = 512
activation = leakyrelu
# scale = gpu([1.0f0, WATER])
U = gpu(Chain(
        Dense(2, h_size, activation),  
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation),
        Dense(h_size, 2),
        # uv -> uv ./ scale
        ))


opt_state = Optimisers.setup(Optimisers.Adam(5f-4), U)

for i in 1:300

    loss, back = Flux.pullback(U) do _U
        z_pinn = reshape(_U(pinn_grid), 2, elements, integration_steps + 1, :)
        z_pinn = permutedims(z_pinn, (2, 1, 4, 3))[:, :, 1, :]
        z_pinn_next = runge_kutta(dyn, z_pinn, vec(t), F, dt)
        u_pinn = z_pinn[:, 1, :]
        energy_pinn = dropdims(sum(u_pinn .^ 2, dims = 1) * dx, dims = 1)

        physics_loss = Flux.mse(z_pinn[:, :, 2:end], z_pinn_next[:, :, 1:end-1])
        boundary_loss = Flux.mean(u_pinn[1, :] .^ 2) + Flux.mean(u_pinn[end, :] .^ 2)
        energy_loss = Flux.mse(energy, energy_pinn)
        initial_condition_loss = Flux.mse(z_pinn[:, :, 1], z[:, :, 1, 1])
        return physics_loss + boundary_loss + initial_condition_loss + energy_loss
    end

    println("Loss: $loss")
    gs = back(one(loss))[1]
    opt_state, U = Optimisers.update(opt_state, U, gs)
end







z_pinn = reshape(U(pinn_grid), 2, elements, integration_steps + 1, :)
z_pinn = permutedims(z_pinn, (2, 1, 4, 3))

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, cpu(z_pinn[:, 1, 1, :]))
ax = Axis(fig[1, 2])
heatmap!(ax, cpu(z[:, 1, 1, :]))

ax = Axis(fig[2, 1])
heatmap!(ax, cpu(z_pinn[:, 2, 1, :]))
ax = Axis(fig[2, 2])
heatmap!(ax, cpu(z[:, 2, 1, :]))
save("z_pinn.png", fig)

u_pinn = z_pinn[:, 1, 1, :]
energy_pinn = dropdims(sum(u_pinn .^ 2, dims = 1) * dx, dims = 1)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, cpu(energy))
lines!(ax, cpu(energy_pinn))
save("energy.png", fig)

u_pinn = cpu(u_pinn)
fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)

record(fig, "vid.mp4", axes(u_pinn, 2)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, u_pinn[:, i], color = :blue)
end