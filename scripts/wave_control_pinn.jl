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

function build_compressor(in_channels::Int, h_size::Int, activation::Function, out_size::Int)
    return Chain(
        Conv((2,), in_channels => h_size, activation, pad = SamePad()),
        Conv((2,), h_size => h_size, activation, pad = SamePad()),
        MaxPool((2,)),
        Conv((2,), h_size => h_size, activation, pad = SamePad()),
        Conv((2,), h_size => h_size, activation, pad = SamePad()),
        MaxPool((2,)),
        Conv((2,), h_size => h_size, activation, pad = SamePad()),
        Conv((2,), h_size => h_size, activation, pad = SamePad()),
        Conv((2,), h_size => out_size),
        GlobalMaxPool(),
        l -> dropdims(l, dims = 1))
end

function build_wave_pinn(in_size::Int, h_size::Int, activation::Function)
    return Chain(
        Dense(in_size, h_size, activation),  
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
end

struct WaveControlPINN
    W::WaveEncoder
    D::DesignEncoder
    R::Chain
    U::Chain
    grid::AbstractArray{Float32, 3}

    latent_dim::OneDim
    time_steps::Int
end

Flux.@functor WaveControlPINN
Flux.trainable(model::WaveControlPINN) = (;model.W, model.D, model.R, model.U)

function compress(model::WaveControlPINN, z::AbstractArray{Float32, 3}, C::LinearInterpolation, t::AbstractMatrix{Float32})
    return model.R(
        hcat(
            z, 
            Flux.unsqueeze(C(t[1, :]), 2), 
            Flux.unsqueeze(C(t[end, :]), 2))
        )
end

function build_pinn_input(model::WaveControlPINN, l::AbstractMatrix{Float32})
    return vcat(repeat(Flux.unsqueeze(l, 2), 1, size(model.grid, 2), 1), repeat(model.grid, 1, 1, size(l, 2)))
end

function compute_pinn_sol(model::WaveControlPINN, pinn_input::AbstractArray{Float32, 3})
    return permutedims(
        reshape(
            model.U(pinn_input), 
            4, 
            length(model.latent_dim.x), 
            model.time_steps, 
            :),
            (2, 1, 4, 3)
        )
end

function Waves.generate_latent_solution(model::WaveControlPINN, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = model.W(s)
    C = model.D(s, a, t)
    l = compress(model, z, C, t)
    pinn_input = build_pinn_input(model, l)
    pinn_sol = compute_pinn_sol(model, pinn_input)
    return compute_latent_energy(pinn_sol, get_dx(model.latent_dim))
end

struct WaveControlPINNLoss
    c0::Float32
    ω::Float32
    grad_x::AbstractMatrix{Float32}
    grad_t::AbstractMatrix{Float32}
    bc::AbstractMatrix{Float32}
end

Flux.@functor WaveControlPINNLoss
Flux.trainable(loss_func::WaveControlPINNLoss) = (;)

function WaveControlPINNLoss(env::WaveEnv, latent_dim::OneDim)
    grad_x = Matrix{Float32}(Waves.gradient(latent_dim.x))
    tspan = build_tspan(0.0f0, env.dt, env.integration_steps)
    grad_t = Matrix{Float32}(Waves.gradient(tspan))
    bc = build_dirichlet(latent_dim)[:, :]

    return WaveControlPINNLoss(
        env.iter.dynamics.c0,
        env.source.freq,
        grad_x,
        grad_t,
        bc)
end

# function (loss_func::WaveControlPINNLoss)(model::WaveControlPINN, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32}, y::AbstractArray{Float32, 3})
# end

# function make_plots(
#         model::WaveControlPINN, 
#         batch; path::String, samples::Int = 1)

#     s, a, t, y = batch

#     z = cpu(Waves.generate_latent_solution(model, s, a, t))

#     y_hat = cpu(model(s, a, t))
#     y = cpu(y)
#     for i in 1:min(length(s), samples)
#         tspan = cpu(t[:, i])
#         Waves.plot_predicted_energy(tspan, y[:, 1, i], y_hat[:, 1, i], title = "Total Energy", path = joinpath(path, "tot$i.png"))
#         Waves.plot_predicted_energy(tspan, y[:, 2, i], y_hat[:, 2, i], title = "Incident Energy", path = joinpath(path, "inc$i.png"))
#         Waves.plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
#     end
    
#     return nothing
# end