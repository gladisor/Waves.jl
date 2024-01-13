using Waves, CairoMakie, Flux, Optimisers, BSON
using ForwardDiff, ReverseDiff, FiniteDiff
using FiniteDifferences

Flux.CUDA.allowscalar(false)
Flux.device!(0)
display(Flux.device())

struct WaveControlPINN
    W::WaveEncoder
    D::DesignEncoder
    R::Chain
    U::Chain

    pinn_grid::AbstractArray{Float32, 3}
    ∇ₓ::AbstractMatrix{Float32}
    ∇ₜ::AbstractMatrix{Float32}
    ω::Float32
    dx::Float32
end

Flux.@functor WaveControlPINN
Flux.trainable(model::WaveControlPINN) = (;model.W, model.D, model.R, model.U)

function (model::WaveControlPINN)(s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32})
    x = model.W(s)
    z = x[:, 1:4, :]
    ## expand force oscilation over time period
    F = Source(x[:, 5, :], model.ω)
    f = cat([F(t[i, :]) for i in axes(t, 1)]..., dims = 3)
    f = permutedims(f, (1, 3, 2))
    # ## add dim to pml
    pml = permutedims(x[:, 6, :, :], (1, 3, 2))
    # ## unroll wavespeed interpolation
    C = model.D(s, a, t)
    c = cat([C(t[i, :]) for i in axes(t, 1)]..., dims = 3)
    c = permutedims(c, (1, 3, 2))
    ## get wavespeed at beginning and end of time period
    c1 = Flux.unsqueeze(C(t[1, :]), 2)
    c2 = Flux.unsqueeze(C(t[end, :]), 2)
    ## compress info into latent vector
    l = model.R(hcat(x, c1, c2))
    l = repeat(l, 1, size(model.pinn_grid, 2), 1)
    ## form input to pinn
    input = vcat(l, repeat(model.pinn_grid, 1, 1, size(l, 3)))
    ## evaluate pinn and reshape solution
    sol = reshape(model.U(input), 2, size(x, 1), size(t, 1), size(x, 3))
    return sol, z, f, pml, c
end


function main()

    ## loading environment and data
    dataset_name = "variable_source_yaxis_x=-10.0"
    DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
    @time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
    @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:5]

    h_size = 128
    activation = leakyrelu
    latent_gs = 10.0f0
    elements = 1024
    horizon = 1
    batchsize = 1
    train_val_split = 0.90 ## choosing percentage of data for val
    data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)
    ## spliting data
    idx = Int(round(length(data) * train_val_split))
    train_data, val_data = data[1:idx], data[idx+1:end]
    # preparing DataLoader(s)
    train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
    val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
    println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")
    s, a, t, y = gpu(Flux.batch.(first(train_loader)))

    latent_dim = OneDim(latent_gs, elements)
    W = WaveEncoder(env, 4, h_size, activation, 500, latent_dim)
    D = DesignEncoder(env, h_size, activation, 500, latent_dim)
    R = Chain(
        Conv((2,), 8 => 64, activation, pad = SamePad()),
        Conv((2,), 64 => 64, activation, pad = SamePad()),
        MaxPool((2,)),
        Conv((2,), 64 => 64, activation, pad = SamePad()),
        Conv((2,), 64 => 64, activation, pad = SamePad()),
        MaxPool((2,)),
        Conv((2,), 64 => 64, activation, pad = SamePad()),
        Conv((2,), 64 => 64, activation, pad = SamePad()),
        Conv((2,), 64 => 64),
        GlobalMaxPool(),
        l -> permutedims(l, (2, 1, 3)))

    wave_normalization = gpu([1.0f0, WATER])[:, :, :]

    U = Chain(
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
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation),
        Dense(h_size, 2),
        uv -> uv ./ wave_normalization
        )

    ## preparing discretization grid
    tspan = build_tspan(env)
    grad_t = Waves.gradient(tspan)
    grad_x = Waves.gradient(latent_dim.x)
    t_grid = repeat(reshape(tspan, 1, 1, length(tspan)), 1, size(latent_dim.x, 1), 1) / (env.dt * env.integration_steps)
    x_grid = repeat(reshape(latent_dim.x, 1, elements, 1), 1, 1, length(tspan)) / latent_gs
    pinn_grid = gpu(vcat(x_grid, t_grid))
    pinn_grid = reshape(pinn_grid, 2, :, 1)
    grad_x = Matrix{Float32}(grad_x)
    grad_t = Matrix{Float32}(grad_t)
    dx = get_dx(latent_dim)

    model = gpu(WaveControlPINN(W, D, R, U, pinn_grid, grad_x, grad_t, env.source.freq, dx))
    opt_state = Optimisers.setup(Optimisers.Adam(5f-4), model)
    
    force = gpu(build_normal(latent_dim.x, [-5.0f0, 5.0f0], [0.3f0, 0.3f0], [1.0f0, 1.0f0]))
    F = Source(force, model.ω)
    f = hcat([Flux.unsqueeze(F(t[i, :]), dims = 2) for i in axes(t, 1)]...)
    # x = model.W(s)
    # pml = x[:, 6, :]

    energy_scale = 1.0f0

    for i in 1:1000

        p_loss = nothing
        p_u = nothing
        p_v = nothing
        e_loss = nothing

        loss, back = Flux.pullback(model) do m

            ## solve
            sol = reshape(m.U(m.pinn_grid), 2, elements, 101, 1)
            u = sol[1, :, :, :] # (1024 x 101 x 1)
            v = sol[2, :, :, :] # (1024 x 101 x 1)

            ## parameters of solution
            # x = m.W(s)
            # F = Source(x[:, 5, :], m.ω)
            # f = hcat([Flux.unsqueeze(F(t[i, :]), dims = 2) for i in axes(t, 1)]...)
            # f = (1024 x 101 x 1)
            # C = m.D(s, a, t)
            # c = cat([C(t[i, :]) for i in axes(t, 1)]..., dims = 3)
            # c = permutedims(c, (1, 3, 2))

            ## compute pinn loss fields
            # u_t = batched_transpose(batched_mul(m.∇ₜ, batched_transpose(u))) # (1024 x 101 x 1)
            # N_u = WATER ^ 2 * batched_mul(m.∇ₓ, v)
            
            # v_t = batched_transpose(batched_mul(m.∇ₜ, batched_transpose(v)))
            # N_v = batched_mul(m.∇ₓ, u .+ f)

            u_tt = batched_transpose(batched_mul(m.∇ₜ * m.∇ₜ, batched_transpose(u))) # (1024 x 101 x 1)
            N_u = WATER ^ 2 * batched_mul(m.∇ₓ * m.∇ₓ, u) .+ f

            ## compute energy of u
            y_hat = dropdims(sum(u .^ 2, dims = 1) * m.dx, dims = 1) # (101 x 1)
            
            physics_loss_u = Flux.mse(u_tt, N_u) #Flux.mse(u_t, N_u)
            # physics_loss_v = Flux.mse(v_t, N_v)
            energy_loss = Flux.mse(y_hat, y[:, 2, :] * energy_scale) # y = (101 x 3 x 1)

            Flux.ignore() do
                p_u = physics_loss_u
                # p_v = physics_loss_v
                e_loss = energy_loss
            end

            return energy_loss + physics_loss_u# + physics_loss_v
        end

        # println("Iteration $i, U: $p_u, V: $p_v, E: $e_loss")
        println("Iteration $i, U: $p_u, E: $e_loss")


        gs = back(one(loss))[1]
        opt_state, model = Optimisers.update(opt_state, model, gs)
    end

    sol = cpu(reshape(model.U(model.pinn_grid), 2, elements, 101, 1))
    u = sol[1, :, :, :]
    v = sol[2, :, :, :]
    y_hat = cpu(dropdims(sum(u .^ 2, dims = 1) * model.dx, dims = 1))

    t = cpu(t)

    fig = Figure()
    ax1 = Axis(fig[1, 1])
    heatmap!(ax1, latent_dim.x, t[:, 1], u[:, :, 1], colormap = :ice)
    ax2 = Axis(fig[1, 2])
    heatmap!(ax2, latent_dim.x, t[:, 1], v[:, :, 1], colormap = :ice)
    save("pinn.png", fig)

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, cpu(y[:, 2, 1]) * energy_scale)
    lines!(ax, y_hat[:, 1])
    save("sigma.png", fig)
end

main()