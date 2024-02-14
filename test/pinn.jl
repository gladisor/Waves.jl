using Waves, CairoMakie, Flux, Optimisers, BSON

Flux.device!(1)

latent_gs = 100.0f0
elements = 1024
latent_dim = OneDim(latent_gs, elements)
dx = get_dx(latent_dim)
c = sin.(5.0f0 * 2.0f0 * pi / 10.0f0 * latent_dim.x) ./ 2.0f0 .+ 1.0f0
c = gpu(c[:, :])
pml = gpu(build_pml(latent_dim, 3.0f0, 10000.0f0))[:, :]

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
    u_t = WATER * c .* dyn.grad * v .- pml .* u
    v_t = WATER * c .* dyn.grad * (u .+ f) .- pml .* v

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

function build_pinn_grid(latent_dim::OneDim, t::Vector{Float32})
    latent_gs = maximum(latent_dim.x)
    elements = length(latent_dim.x)
    dt = Flux.mean(diff(vec(t)))
    integration_steps = length(t)

    t_grid = repeat(reshape(t, 1, 1, integration_steps), 1, size(latent_dim.x, 1), 1) / (dt * integration_steps)
    x_grid = repeat(reshape(latent_dim.x, 1, elements, 1), 1, 1, integration_steps) / latent_gs
    pinn_grid = vcat(x_grid, t_grid)
    return reshape(pinn_grid, 2, :, 1)
end

dt = 1f-5
integration_steps = 300

dyn = gpu(SimpleWave(
    build_gradient(latent_dim), 
    WATER, 
    build_dirichlet(latent_dim)))

iter = gpu(Integrator(runge_kutta, dyn, dt))
t = build_tspan(0.0f0, dt, integration_steps)
pinn_grid = gpu(build_pinn_grid(latent_dim, t))

grad_x = gpu(Waves.gradient(latent_dim.x))
grad_t = gpu(Waves.gradient(t))

wave = gpu(zeros(Float32, elements, 2, 1))
t = gpu(t[:, :])

F = gpu(
    Source(
        build_normal(latent_dim.x, [-2.0f0, 2.0f0, 3.0f0], [0.3f0, 0.3f0, 0.4f0], [1.0f0, -1.0f0, 1.0f0]),
        1000.0f0)
        )
f = hcat([F(t[i, :]) for i in axes(t, 1)]...)

z = iter(wave, t, F)
u = z[:, 1, 1, :]
v = z[:, 2, 1, :]
energy = vec(sum(u .^ 2, dims = 1) * dx)

h_size = 256
activation = leakyrelu

function main()
    U = gpu(
        Chain(
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
            Dense(h_size, h_size, activation), 
            Dense(h_size, h_size, activation),  
            Dense(h_size, 2)))
    
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), U)

    for i in 1:5000

        U_LOSS = nothing
        V_LOSS = nothing
        BOUNDARY_LOSS = nothing
        IC_LOSS = nothing
        ENERGY_LOSS = nothing

        loss, back = Flux.pullback(U) do _U
            z_pinn = reshape(_U(pinn_grid), 2, elements, integration_steps + 1, :)
            z_pinn = permutedims(z_pinn, (2, 1, 4, 3))[:, :, 1, :] # (space, fields, time)
            u_pinn = z_pinn[:, 1, :]
            v_pinn = z_pinn[:, 2, :]

            u_t = (grad_t * u_pinn')'
            N_u = (WATER * c .* (grad_x * v_pinn) .- pml .* u_pinn) .* dyn.bc

            v_t = (grad_t * v_pinn')'
            N_v = WATER * c .* (grad_x * (u_pinn .+ f)) .- pml .* v_pinn

            energy_pinn = vec(sum(u_pinn .^ 2, dims = 1) * dx)

            u_loss = Flux.mse(u_t, N_u) / WATER
            v_loss = Flux.mse(v_t, N_v) / WATER
            boundary_loss = Flux.mean(u_pinn[1, :] .^ 2) + Flux.mean(u_pinn[end, :] .^ 2)
            ic_loss = Flux.mse(z_pinn[:, :, 1], z[:, :, 1, 1])
            energy_loss = Flux.mse(energy_pinn, energy)

            Flux.ignore() do
                U_LOSS = u_loss
                V_LOSS = v_loss
                BOUNDARY_LOSS = boundary_loss
                IC_LOSS = ic_loss
                ENERGY_LOSS = energy_loss
            end

            return u_loss + v_loss + 100.0f0 * WATER * (boundary_loss + ic_loss) + energy_loss
        end

        println("u: $U_LOSS, v: $V_LOSS, b: $BOUNDARY_LOSS, ic: $IC_LOSS, energy: $ENERGY_LOSS")
        gs = back(one(loss))[1]
        opt_state, U = Optimisers.update(opt_state, U, gs)
    end

    return U
end

U = main()

z_pinn = reshape(U(pinn_grid), 2, elements, integration_steps + 1, :)
z_pinn = permutedims(z_pinn, (2, 1, 4, 3))[:, :, 1, :] # (space, fields, time)
u_pinn = z_pinn[:, 1, :]
energy_pinn = vec(sum(u_pinn .^ 2, dims = 1) * dx)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, cpu(energy), label = "Ground Truth")
lines!(ax, cpu(energy_pinn), label = "PINN")
axislegend(ax, position = :lt)
save("test/energy.png", fig)

fig = Figure()
ax = Axis(fig[1, 1], title = "Ground Truth", xlabel = "Space (m)", ylabel = "Time (s)")
heatmap!(ax, latent_dim.x, cpu(t[:, 1]), cpu(u))
ax = Axis(fig[1, 2], title = "PINN", xlabel = "Space(m)")
heatmap!(ax, latent_dim.x, cpu(t[:, 1]), cpu(u_pinn))
save("test/sol.png", fig)

fig = Figure()
ax = Axis(fig[1, 1], title = "Comparison of PINN vs GT Solutions", xlabel = "Space (m)", ylabel = "Displacement (m)")
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)
record(fig, "test/vid.mp4", axes(u, 2)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, cpu(u[:, i]), color = :blue)
    lines!(ax, latent_dim.x, cpu(u_pinn[:, i]), color = :orange)
end

