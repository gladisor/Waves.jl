# function build_pinn_grid(latent_dim::OneDim, t::Vector{Float32})
#     latent_gs = maximum(latent_dim.x)
#     elements = length(latent_dim.x)
#     dt = Flux.mean(diff(vec(t)))
#     integration_steps = length(t)

#     t_grid = repeat(reshape(t, 1, 1, integration_steps), 1, size(latent_dim.x, 1), 1) / (dt * integration_steps)
#     x_grid = repeat(reshape(latent_dim.x, 1, elements, 1), 1, 1, integration_steps) / latent_gs
#     pinn_grid = vcat(x_grid, t_grid)
#     return reshape(pinn_grid, 2, :, 1)
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

struct WaveControlPINN
    W::WaveEncoder
    D::DesignEncoder
    R::Chain
    U::Chain

    pinn_grid::AbstractArray{Float32, 3}
    ω::Float32
    dx::Float32
end

Flux.@functor WaveControlPINN
Flux.trainable(model::WaveControlPINN) = (;model.W, model.D, model.R, model.U)

function Waves.generate_latent_solution(model::WaveControlPINN, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    x = model.W(s)
    z = x[:, 1:4, :]
    C = model.D(s, a, t)

    l = model.R(
        hcat(
            x, 
            Flux.unsqueeze(C(t[1, :]), 2),
            Flux.unsqueeze(C(t[end, :]), 2)
        )
    )

    num_collocation_points = size(model.pinn_grid, 2)
    batchsize = size(l, 2)

    pinn_input = vcat(
        repeat(Flux.unsqueeze(l, 2), 1, num_collocation_points, 1),
        repeat(model.pinn_grid, 1, 1, batchsize))

    sol = model.U(pinn_input)

    sol = reshape(
        sol, 
        size(sol, 1), 
        size(z, 1), 
        :,
        batchsize)

    return permutedims(sol, (2, 1, 4, 3))
end

function (model::WaveControlPINN)(s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = generate_latent_solution(model, s, a, t)
    return compute_latent_energy(z, model.dx)
end

struct WaveControlPINNLoss
    c0::Float32
    ω::Float32
    grad_x::AbstractMatrix{Float32}
    grad_t::AbstractMatrix{Float32}
    bc::AbstractVector{Float32}
end

Flux.@functor WaveControlPINNLoss
Flux.trainable(loss_func::WaveControlPINNLoss) = (;)

function (loss_func::WaveControlPINNLoss)(model::WaveControlPINN, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32}, y::AbstractArray{Float32, 3})
    ## call to encoders
    x = model.W(s)
    z = x[:, 1:4, :]
    F = Source(x[:, 5, :], model.ω)
    pml = x[:, [6], :]
    C = model.D(s, a, t)
    ## unroll force and speed of sound functions over timespan
    f = hcat([Flux.unsqueeze(F(t[i, :]), 2) for i in axes(t, 1)]...)
    c = hcat([Flux.unsqueeze(C(t[i, :]), 2) for i in axes(t, 1)]...)
    l = model.R(hcat(x, c[:, [1, end], :]))
    ## broacast compressed vector to pinn_grid
    pinn_input = vcat(
        repeat(Flux.unsqueeze(l, 2), 1, size(model.pinn_grid, 2), 1),
        repeat(model.pinn_grid, 1, 1, size(l, 2)))
    
    ## solve pde
    sol = model.U(pinn_input)
    sol = reshape(
        sol, 
        size(sol, 1), 
        size(loss_func.grad_x, 1), 
        size(loss_func.grad_t, 1), 
        size(l, 2))

    ## extract fields from solution
    u_tot = sol[1, :, :, :]
    v_tot = sol[2, :, :, :]
    u_inc = sol[3, :, :, :]
    v_inc = sol[4, :, :, :]
    ## compute scattered field
    u_sc = u_tot .- u_inc

    pml_scale = 10000.0f0
    ## u_tot
    u_tot_t = batched_transpose(batched_mul(loss_func.grad_t, batched_transpose(u_tot)))
    N_u_tot = (loss_func.c0 * c .* batched_mul(loss_func.grad_x, v_tot) .- pml_scale * pml .* u_tot) .* loss_func.bc
    ## v_tot
    v_tot_t = batched_transpose(batched_mul(loss_func.grad_t, batched_transpose(v_tot)))
    N_v_tot = loss_func.c0 * c .* batched_mul(loss_func.grad_x, u_tot .+ f) .- pml_scale * pml .* v_tot
    ## u_inc
    u_inc_t = batched_transpose(batched_mul(loss_func.grad_t, batched_transpose(u_inc)))
    N_u_inc = (loss_func.c0 * batched_mul(loss_func.grad_x, v_inc) .- pml_scale * pml .* u_inc) .* loss_func.bc
    ## v_inc
    v_inc_t = batched_transpose(batched_mul(loss_func.grad_t, batched_transpose(v_inc)))
    N_v_inc = loss_func.c0 * batched_mul(loss_func.grad_x, u_inc .+ f) .- pml_scale * pml .* v_inc
    ## compute physics loss
    u_tot_loss = Flux.mse(u_tot_t, N_u_tot) / loss_func.c0
    v_tot_loss = Flux.mse(v_tot_t, N_v_tot) / loss_func.c0
    u_inc_loss = Flux.mse(u_inc_t, N_u_inc) / loss_func.c0
    v_inc_loss = Flux.mse(v_inc_t, N_v_inc) / loss_func.c0
    physics_loss = u_tot_loss + v_tot_loss# + u_inc_loss + v_inc_loss
    
    ## compute initial condition loss
    sol_0 = permutedims(sol[:, :, 1, :], (2, 1, 3))
    ic_loss = Flux.mse(sol_0, z)

    ## compute boundary condition loss
    tot_bc_loss = Flux.mean(u_tot[1, :, :] .^ 2  .+ u_tot[end, :, :] .^ 2)
    inc_bc_loss = Flux.mean(u_inc[1, :, :] .^ 2  .+ u_inc[end, :, :] .^ 2)
    sc_bc_loss = Flux.mean(u_sc[1, :, :] .^ 2  .+ u_sc[end, :, :] .^ 2)
    # bc_loss = tot_bc_loss + inc_bc_loss + sc_bc_loss
    bc_loss = tot_bc_loss

    ## compute energies
    tot_energy = dropdims(sum(u_tot .^ 2, dims = 1) * model.dx, dims = 1)
    # inc_energy = dropdims(sum(u_inc .^ 2, dims = 1) * model.dx, dims = 1)
    # sc_energy = dropdims(sum(u_sc .^ 2, dims = 1) * model.dx, dims = 1)
    
    # y_hat = hcat(
    #     Flux.unsqueeze(tot_energy, 2),
    #     Flux.unsqueeze(inc_energy, 2),
    #     Flux.unsqueeze(sc_energy, 2)
    #     )
    
    # energy_loss = Flux.mse(y_hat, y)

    energy_loss = Flux.mse(tot_energy, y[:, 1, :])
    # energy_loss = Flux.mse(tot_energy)
    return energy_loss + 100.0f0 * loss_func.c0 * (ic_loss + bc_loss) + physics_loss
end

function make_plots(
        model::WaveControlPINN, 
        batch; path::String, samples::Int = 1)

    s, a, t, y = batch

    z = cpu(Waves.generate_latent_solution(model, s, a, t))

    y_hat = cpu(model(s, a, t))
    y = cpu(y)
    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        Waves.plot_predicted_energy(tspan, y[:, 1, i], y_hat[:, 1, i], title = "Total Energy", path = joinpath(path, "tot$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 2, i], y_hat[:, 2, i], title = "Incident Energy", path = joinpath(path, "inc$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end
    
    return nothing
end