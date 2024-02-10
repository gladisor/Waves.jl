export build_pinn_grid, WaveControlPINN, WaveControlPINNLoss

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

# """
# Computes the time derivative of a batch of one dimentional scalar fields. The dimentionality
# of the field should be (space x time x batch).
# """
# function batched_time_derivative(grad_t::AbstractMatrix, field::AbstractArray{Float32, 3})
#     return batched_transpose(batched_mul(grad_t, batched_transpose(field)))
# end

# function evaluate_over_time(f, t::AbstractMatrix{Float32})
#     hcat([Flux.unsqueeze(f(t[i, :]), 2) for i in axes(t, 1)]...)
# end

# function compute_acoustic_wave_physics_loss(
#         grad_x::AbstractMatrix{Float32},
#         grad_t::AbstractMatrix{Float32},
#         sol::AbstractArray{Float32, 4}, # (fields, space, batch, time)
#         c0::Float32,
#         C::LinearInterpolation,
#         F::Source,
#         pml::AbstractMatrix{Float32},
#         bc::AbstractMatrix{Float32},
#         t::AbstractMatrix{Float32})

#     sol = permutedims(sol, (1, 2, 4, 3))

#     ## unpack fields from solution
#     u_tot = sol[:, 1, :, :] ## (space, time, batch)
#     v_tot = sol[:, 2, :, :]
#     u_inc = sol[:, 3, :, :]
#     v_inc = sol[:, 4, :, :]

#     ## compute derivatives
#     ## u_tot
#     u_tot_t = batched_time_derivative(grad_t, u_tot)
#     ## v_tot
#     v_tot_t = batched_time_derivative(grad_t, v_tot)
#     ## u_inc
#     u_inc_t = batched_time_derivative(grad_t, u_inc)
#     ## v_inc
#     v_inc_t = batched_time_derivative(grad_t, v_inc)

#     c = evaluate_over_time(C, t) ## design encoder
#     f = evaluate_over_time(F, t) ## wave encoder 
#     pml = Flux.unsqueeze(pml, 2) ## wave encoder
#     bc = Flux.unsqueeze(bc, 2)

#     pml_scale = 10000.0f0

#     N_u_tot = (c0 * c .* batched_mul(grad_x, v_tot) .- pml_scale * pml .* u_tot) .* bc
#     N_v_tot = (c0 * c .* batched_mul(grad_x, u_tot .+ f) .- pml_scale * pml .* v_tot)

#     N_u_inc = (c0 * batched_mul(grad_x, v_inc) .- pml_scale * pml .* u_inc) .* bc
#     N_v_inc = (c0 * batched_mul(grad_x, u_inc .+ f) .- pml_scale * pml .* v_inc)

#     return (
#         Flux.mse(u_tot_t, N_u_tot) + 
#         Flux.mse(v_tot_t, N_v_tot) + 
#         Flux.mse(u_inc_t, N_u_inc) + 
#         Flux.mse(v_inc_t, N_v_inc))
# end

# function call_acoustic_wave_pinn(U::Chain, grid::AbstractArray{Float32, 3})
#     return permutedims(reshape(U(grid), 4, 1024, 101, :), (2, 1, 4, 3))
# end

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

function WaveControlPINN(;
        env::WaveEnv, 
        activation::Function, 
        h_size::Int, 
        nfreq::Int, 
        latent_dim::OneDim, 
        l_size::Int = 64)

    W = WaveEncoder(env, 4, h_size, activation, nfreq, latent_dim)
    D = DesignEncoder(env, h_size, activation, nfreq, latent_dim)
    R = build_compressor(8, h_size, activation, l_size)
    U = build_wave_pinn(l_size + 2, h_size, activation)
    tspan = build_tspan(0.0f0, env.dt, env.integration_steps)
    grid = build_pinn_grid(latent_dim, tspan)
    return WaveControlPINN(W, D, R, U, grid, latent_dim, env.integration_steps + 1)
end

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

# function generate_latent_solution(model::WaveControlPINN, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
#     z = model.W(s)
#     C = model.D(s, a, t)
#     l = compress(model, z, C, t)
#     pinn_input = build_pinn_input(model, l)
#     return compute_pinn_sol(model, pinn_input)
# end

mutable struct CustomRecur{T,S}
    cell::T
    state::S
  end

function (m::CustomRecur)(x)
    m.state, y = m.cell(m.state, x)
    return y
end

Flux.@functor CustomRecur
Flux.trainable(a::CustomRecur) = (; cell = a.cell)
Base.show(io::IO, m::CustomRecur) = print(io, "CustomRecur(", m.cell, ")")
Flux.reset!(m::CustomRecur) = (m.state = m.cell.state0)

function Waves.generate_latent_solution(model::WaveControlPINN, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = model.W(s)
    x = z[:, 1:4, :]
    f = z[:, [5], :]
    pml = z[:, [6], :]
    C = model.D(s, a, t)

    timepoints = t[1:model.D.integration_steps:end, :]
    c = Waves.evaluate_over_time(C, timepoints)

    recur = CustomRecur(x) do _x, _ci
        l = model.R(hcat(_x, f, pml, _ci))
        pinn_input = Waves.build_pinn_input(model, l)
        sol = Waves.compute_pinn_sol(model, pinn_input)
        return sol[:, :, :, end], sol
    end

    return flatten_repeated_last_dim(Flux.batch([recur(c[:, i:i+1, :]) for i in 1:(size(c, 2) - 1)]))
end

function (model::WaveControlPINN)(s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    return compute_latent_energy(generate_latent_solution(model, s, a, t), get_dx(model.latent_dim))
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

function (loss_func::WaveControlPINNLoss)(model::WaveControlPINN, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32}, y::AbstractArray{Float32, 3})
    ## call encoders
    z = model.W(s)
    C = model.D(s, a, t)
    F = Source(z[:, 5, :], loss_func.ω)
    pml = z[:, 6, :]
    ## compute the latent solution from pinn
    l = compress(model, z, C, t)
    pinn_input = build_pinn_input(model, l)
    pinn_sol = compute_pinn_sol(model, pinn_input)
    ## compute physics_loss
    f_loss = compute_acoustic_wave_physics_loss(loss_func.grad_x, loss_func.grad_t, pinn_sol, loss_func.c0, C, F, pml, loss_func.bc, t)
    ic_loss = Flux.mse(pinn_sol[:, :, :, 1], z[:, 1:4, :])
    bc_loss = Flux.mean(pinn_sol[[1, end], [1, 3], :, :] .^ 2)
    physics_loss = 100.0f0 * loss_func.c0 * (ic_loss + bc_loss) + f_loss / loss_func.c0
    ## compute energy loss
    y_hat = compute_latent_energy(pinn_sol, get_dx(model.latent_dim))
    energy_loss = Flux.mse(y_hat, y)
    ## sum and return losses
    return energy_loss + 0.01f0 * physics_loss
end

function Waves.make_plots(
        model::WaveControlPINN, 
        batch; path::String, 
        samples::Int = 1)

    s, a, t, y = batch
    z = cpu(generate_latent_solution(model, s, a, t))
    latent_dim = cpu(model.latent_dim)
    render_latent_solution!(latent_dim, cpu(t[:, 1]), z[:, :, 1, :], path = path)

    x = model.W(s)
    z0 = x[:, 1:4, :]
    f = x[:, 5, :]
    pml = x[:, 6, :]
    C = model.D(s, a, t)

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, latent_dim.x, cpu(pml[:, 1]))
    save(joinpath(path, "pml.png"), fig)

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, latent_dim.x, cpu(f[:, 1]))
    save(joinpath(path, "force.png"), fig)

    y_hat = cpu(model(s, a, t))
    y = cpu(y)
    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        plot_predicted_energy(tspan, y[:, 1, i], y_hat[:, 1, i], title = "Total Energy", path = joinpath(path, "tot$i.png"))
        plot_predicted_energy(tspan, y[:, 2, i], y_hat[:, 2, i], title = "Incident Energy", path = joinpath(path, "inc$i.png"))
        plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end

    return nothing
end