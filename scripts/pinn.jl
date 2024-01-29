using Waves, Flux, CairoMakie, BSON
using Optimisers

Flux.device!(0)
Flux.CUDA.allowscalar(false)
display(Flux.device())

include("wave_control_pinn.jl")

"""
Computes the time derivative of a batch of one dimentional scalar fields. The dimentionality
of the field should be (space x time x batch).
"""
function batched_time_derivative(grad_t::AbstractMatrix, field::AbstractArray{Float32, 3})
    return batched_transpose(batched_mul(grad_t, batched_transpose(field)))
end

function evaluate_over_time(f, t::AbstractMatrix{Float32})
    hcat([Flux.unsqueeze(f(t[i, :]), 2) for i in axes(t, 1)]...)
end

function compute_acoustic_wave_physics_loss(
        grad_x::AbstractMatrix{Float32},
        grad_t::AbstractMatrix{Float32},
        sol::AbstractArray{Float32, 4}, # (fields, space, batch, time)
        c0::Float32,
        C::LinearInterpolation,
        F::Source,
        pml::AbstractMatrix{Float32},
        bc::AbstractMatrix{Float32},
        t::AbstractMatrix{Float32})

    sol = permutedims(sol, (1, 2, 4, 3))

    ## unpack fields from solution
    u_tot = sol[:, 1, :, :] ## (space, time, batch)
    v_tot = sol[:, 2, :, :]
    u_inc = sol[:, 3, :, :]
    v_inc = sol[:, 4, :, :]

    ## compute derivatives
    ## u_tot
    u_tot_t = batched_time_derivative(grad_t, u_tot)
    ## v_tot
    v_tot_t = batched_time_derivative(grad_t, v_tot)
    ## u_inc
    u_inc_t = batched_time_derivative(grad_t, u_inc)
    ## v_inc
    v_inc_t = batched_time_derivative(grad_t, v_inc)

    c = evaluate_over_time(C, t) ## design encoder
    f = evaluate_over_time(F, t) ## wave encoder 
    pml = Flux.unsqueeze(pml, 2) ## wave encoder
    bc = Flux.unsqueeze(bc, 2)

    pml_scale = 10000.0f0

    N_u_tot = (c0 * c .* batched_mul(grad_x, v_tot) .- pml_scale * pml .* u_tot) .* bc
    N_v_tot = (c0 * c .* batched_mul(grad_x, u_tot .+ f) .- pml_scale * pml .* v_tot)

    N_u_inc = (c0 * batched_mul(grad_x, v_inc) .- pml_scale * pml .* u_inc) .* bc
    N_v_inc = (c0 * batched_mul(grad_x, u_inc .+ f) .- pml_scale * pml .* v_inc)

    return (
        Flux.mse(u_tot_t, N_u_tot) + 
        Flux.mse(v_tot_t, N_v_tot) + 
        Flux.mse(u_inc_t, N_u_inc) + 
        Flux.mse(v_inc_t, N_v_inc))
end

function call_acoustic_wave_pinn(U::Chain, grid::AbstractArray{Float32, 3})
    return permutedims(reshape(U(grid), 4, 1024, 101, :), (2, 1, 4, 3))
end

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
## declaring hyperparameters
activation = relu
h_size = 256
in_channels = 4
nfreq = 500
elements = 1024
horizon = 1
lr = 1f-4
batchsize = 4 #32 ## shorter horizons can use large batchsize
val_every = 20
val_batches = val_every
epochs = 10
latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0
train_val_split = 0.90 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)
latent_dim = OneDim(latent_gs, elements)
dx = get_dx(latent_dim)

## loading environment and data
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:5]

## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]

## preparing DataLoader(s)
train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

## contstruct
base = build_cnn_base(env, in_channels, activation, h_size)
head = build_pinn_wave_encoder_head(h_size, activation, nfreq, latent_dim)
W = gpu(WaveEncoder(base, head))
D = gpu(DesignEncoder(env, h_size, activation, nfreq, latent_dim))
dyn = AcousticDynamics(latent_dim, WATER, 1.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

U = gpu(
        Chain(
            Dense(64 + 2, h_size, activation),  
            Dense(h_size, h_size, activation), 
            Dense(h_size, h_size, activation), 
            Dense(h_size, h_size, activation), 
            Dense(h_size, h_size, activation), 
            Dense(h_size, h_size, activation), 
            Dense(h_size, h_size, activation),
            Dense(h_size, h_size, activation), 
            Parallel(
                vcat,
                Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, 1)),
                Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, 1)),
                Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, 1)),
                Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, 1))
            )
        )
    )

R = gpu(build_compressor(8, h_size, activation, 64))
s, a, t, y = gpu(Flux.batch.(first(train_loader)))

tspan = build_tspan(0.0f0, env.dt, env.integration_steps)
grid = gpu(build_pinn_grid(latent_dim, tspan))
grad_x = gpu(Matrix{Float32}(Waves.gradient(latent_dim.x)))
grad_t = gpu(Matrix{Float32}(Waves.gradient(tspan)))
c0 = env.iter.dynamics.c0
bc = gpu(build_dirichlet(latent_dim))[:, :]

opt_state = Optimisers.setup(Optimisers.Adam(5f-4), (U, R, W, D))

for i in 1:2000

    ENERGY_LOSS = nothing
    PHYSICS_LOSS = nothing

    loss, back = Flux.pullback((U, R, W, D)) do (_U, _R, _W, _D)
        z = _W(s) ## initial conditions, force, pml
        C = _D(s, a, t) ## interpolation of material properties
        l = _R(hcat(z, Flux.unsqueeze(C(t[1, :]), 2), Flux.unsqueeze(C(t[end, :]), 2)))
        pinn_input = vcat(repeat(Flux.unsqueeze(l, 2), 1, size(grid, 2), 1), repeat(grid, 1, 1, size(l, 2)))
        pinn_sol = call_acoustic_wave_pinn(_U, pinn_input)

        F = Source(z[:, 5, :], env.source.freq)
        pml = z[:, 6, :]

        physics_loss = compute_acoustic_wave_physics_loss(grad_x, grad_t, pinn_sol, c0, C, F, pml, bc, t)
        ic_loss = Flux.mse(pinn_sol[:, :, :, 1], z[:, 1:4, :])
        bc_loss = Flux.mean(pinn_sol[[1, end], [1, 3], :, :] .^ 2)

        y_hat = compute_latent_energy(pinn_sol, dx)
        energy_loss = Flux.mse(y_hat, y)
        L_physics = 100.0f0 * c0 * (ic_loss + bc_loss) + physics_loss / c0

        Flux.ignore() do
            ENERGY_LOSS = energy_loss
            PHYSICS_LOSS = L_physics
        end

        # return energy_loss + 0.001f0 * L_physics
        return energy_loss + 0.01f0 * L_physics
    end
    
    println("$i: Energy Loss: $ENERGY_LOSS, Physics Loss: $PHYSICS_LOSS")
    gs = back(one(loss))[1]
    opt_state, (U, R, W, D) = Optimisers.update(opt_state, (U, R, W, D), gs)
end

z = W(s)
C = D(s, a, t)
l = R(hcat(z, Flux.unsqueeze(C(t[1, :]), 2), Flux.unsqueeze(C(t[end, :]), 2)))
pinn_input = vcat(repeat(Flux.unsqueeze(l, 2), 1, size(grid, 2), 1), repeat(grid, 1, 1, size(l, 2)))
pinn_sol = call_acoustic_wave_pinn(U, pinn_input)
y_hat = compute_latent_energy(pinn_sol, dx)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, cpu(z[:, 5, 1]), label = "Force")
ax = Axis(fig[2, 1])
lines!(ax, cpu(z[:, 6, 1]), label = "PML")
save("parameters.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, cpu(y[:, 1, 1]))
lines!(ax, cpu(y_hat[:, 1, 1]))
ax = Axis(fig[2, 1])
lines!(ax, cpu(y[:, 2, 1]))
lines!(ax, cpu(y_hat[:, 2, 1]))
ax = Axis(fig[3, 1])
lines!(ax, cpu(y[:, 3, 1]))
lines!(ax, cpu(y_hat[:, 3, 1]))
save("energy.png", fig)

pinn_tot = pinn_sol[:, 1, 1, :]
pinn_inc = pinn_sol[:, 3, 1, :]
pinn_sc = pinn_tot .- pinn_inc

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)
record(fig, "vid.mp4", axes(tspan, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, cpu(pinn_tot[:, i]), color = :blue)
    lines!(ax, latent_dim.x, cpu(pinn_inc[:, i]), color = :orange)
    lines!(ax, latent_dim.x, cpu(pinn_sc[:, i]), color = :green)
end
