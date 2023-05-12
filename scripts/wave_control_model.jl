function visualize_predictions!(model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0, title = "Prediction of Scattered Energy Versus Ground Truth")

    sigma_pred = model(s, a)

    for i in axes(tspan, 2)
        lines!(ax, tspan[:, i], cpu(sigma[:, i]), color = :blue, label = "True")
        lines!(ax, tspan[:, i], cpu(sigma_pred[:, i]), color = :orange, label = "Predicted")
    end

    save(path, fig)
end

function visualize_latent_wave!(model::WaveControlModel, latent_dim::OneDim, s::WaveEnvState, actions::Vector{<: AbstractDesign}, tspan::AbstractMatrix; path::String)

    path = mkpath(path)

    tspan = cpu(tspan)
    tspan = vcat(tspan[1], vec(tspan[2:end, :]))

    z_wave = model.wave_encoder(s.wave_total)
    h = (z_wave, s.design)

    zs = []

    for (i, a) in enumerate(actions)
        z_wave, design = h

        z_design = encode_design(model, design, a)
        z = model.iter(hcat(z_wave, z_design))
        h = (z[:, [1, 2, 3], end], design + a)

        if i == 1
            push!(zs, z)
        else
            push!(zs, z[:, :, 2:end])
        end
    end

    z = cpu(cat(zs..., dims = ndims(zs[1])))

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0f0, ylabel = "Time (t)", title = "Displacement")

    if !(maximum(z[:, 1, :]) > minimum(z[:, 1, :]))
        ylims!(ax1, -1.0f0, 1.0f0)
    end

    heatmap!(ax1, latent_dim.x, tspan, z[:, 1, :], colormap = :ice)

    ax2 = Axis(fig[1, 2], aspect = 1.0f0, title = "Velocity")
    heatmap!(ax2, latent_dim.x, tspan, z[:, 2, :], colormap = :ice)

    if !(maximum(z[:, 2, :]) > minimum(z[:, 2, :]))
        ylims!(ax2, -1.0f0, 1.0f0)
    end

    ax3 = Axis(fig[2, 1], aspect = 1.0f0, xlabel = "Space (m)", ylabel = "Time (t)", title = "Force")
    heatmap!(ax3, latent_dim.x, tspan, z[:, 3, :], colormap = :ice)

    ax4 = Axis(fig[2, 2], aspect = 1.0f0, xlabel = "Space (m)", title = "Wave Speed")
    heatmap!(ax4, latent_dim.x, tspan, z[:, 4, :], colormap = :ice)
    save(joinpath(path, "latent.png"), fig)

    render!(latent_dim, z, path = joinpath(path, "latent.mp4"))
    return nothing
end

function train(model::WaveControlModel, train_loader::DataLoader, val_loader::DataLoader, epochs::Int, lr; latent_dim::OneDim, path::String, checkpoint_every::Int = 3, evaluation_samples::Int = 3, decay::Float32 = 1.0f0)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), model)

    train_loss_history = Float32[]
    val_loss_history = Float32[]

    for i in 1:epochs

        println("Training")
        train_loss = 0.0f0
        @showprogress for (s, a, _, σ) in train_loader
            s, a, σ = gpu(s[1]), gpu(a[1]), gpu(σ[1])

            loss, back = pullback(_model -> mse(_model(s, a), σ), model)
            gs = back(one(loss))[1]

            opt_state, model = Optimisers.update(opt_state, model, gs)
            train_loss += loss
        end

        opt_state = Optimisers.adjust(opt_state, lr * decay ^ i)

        println("Validating")
        val_loss = 0.0f0
        @showprogress for (s, a, _, σ) in val_loader
            s, a, σ = gpu(s[1]), gpu(a[1]), gpu(σ[1])
            val_loss += mse(model(s, a), σ)
        end

        train_loss = train_loss / length(train_loader)
        val_loss = val_loss / length(val_loader)
        push!(train_loss_history, train_loss)
        push!(val_loss_history, val_loss)
        println("Epoch: $i, Train Loss: $train_loss, Val Loss: $val_loss")

        println("Plotting Loss")
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Epoch", ylabel = "Loss")
        lines!(ax, train_loss_history, color = :blue, label = "Train")
        lines!(ax, val_loss_history, color = :orange, label = "Val")
        axislegend(ax)
        save(joinpath(path, "loss.png"), fig)

        if i % checkpoint_every == 0
            checkpoint_path = mkpath(joinpath(path, "model$i"))
            println("Checkpointing")
            save(model, checkpoint_path)

            println("Rendering Images")
            @showprogress for (j, (s, a, t, σ)) in enumerate(val_loader)
                s, a, t, σ = gpu(s[1]), gpu(a[1]), t[1], gpu(σ[1])
                visualize_predictions!(t, σ, model, s, a, path = joinpath(checkpoint_path, "predictions$j.png"))
                visualize_latent_wave!(latent_dim, model, s, a, t, path = joinpath(checkpoint_path, "latent$j"))
                j == evaluation_samples && break
            end
        end
    end

    return model
end

struct LatentWaveActivation
    C::Float32
end

Flux.@functor LatentWaveActivation
Flux.trainable(::LatentWaveActivation) = (;)

function (act::LatentWaveActivation)(wave::AbstractMatrix{Float32})
    u = tanh.(wave[1, :])
    v = tanh.(wave[2, :]) / act.C
    f = tanh.(wave[3, :])
    return hcat(u, v, f)
end

struct LatentDesignActivation end
Flux.@functor LatentDesignActivation

function (act::LatentDesignActivation)(latent_design::AbstractMatrix{Float32})
    f = tanh.(latent_design[1, :])
    c = sigmoid.(latent_design[2, :])
    return hcat(f, c)
end

struct Field
    dim::OneDim
end

Flux.@functor Field
Flux.trainable(field::Field) = (;)

function (field::Field)(m)
    return permutedims(m(field.dim.x'), (2, 1))
end

struct SingleImageInput end
Flux.@functor SingleImageInput
(input::SingleImageInput)(img::AbstractArray{Float32, 3}) = img[:, :, :, :]


struct LatentDynamics <: AbstractDynamics
    C0::Float32
    freq::Float32
    pml::AbstractVector{Float32}
    grad::AbstractMatrix{Float32}
    bc::AbstractVector{Float32}
end

Flux.@functor LatentDynamics
Flux.trainable(::LatentDynamics) = (;)

function (dyn::LatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]
    f = wave[:, 3]
    c = wave[:, 4]

    force = f * sin(2.0f0 * pi * dyn.freq * t)

    du = dyn.C0 ^ 2 * c .* (dyn.grad * v) .- dyn.pml .* u
    dv = (dyn.grad * (u .+ force)) .- dyn.pml .* v
    df = f * 0.0f0
    dc = c * 0.0f0

    return hcat(du .* dyn.bc, dv, df, dc)
end

# function build_wave_control_model(;
#         in_channels,
#         h_channels,
#         design_size,
#         action_size,
#         h_size, 
#         latent_grid_size,
#         latent_elements,
#         latent_pml_width, 
#         latent_pml_scale, 
#         ambient_speed,
#         dt,
#         steps,
#         n_mlp_layers,
#         )

#     wave_encoder = Chain(
#         WaveEncoder(in_channels, h_channels, 2, tanh),
#         Dense(1024, latent_elements, tanh),
#         z -> hcat(z[:, 1], z[:, 2] / ambient_speed)
#         )

#     design_encoder = Chain(
#         Dense(design_size + action_size, h_size, relu),
#         Dense(h_size, 2 * latent_elements),
#         z -> reshape(z, latent_elements, :),
#         z -> hcat(tanh.(z[:, 1]), sigmoid.(z[:, 2]))
#         )

#     latent_dim = OneDim(latent_grid_size, latent_elements)
#     grad = build_gradient(latent_dim)
#     pml = build_pml(latent_dim, latent_pml_width, latent_pml_scale)
#     bc = dirichlet(latent_dim)

#     latent_dynamics = ForceLatentDynamics(ambient_speed, 1.0f0, grad, pml, bc)
#     iter = Integrator(runge_kutta, latent_dynamics, 0.0f0, dt, steps)

#     mlp = Chain(
#         flatten,
#         Dense(latent_elements * 4, h_size, relu), 
#         [Dense(h_size, h_size, relu) for _ in 1:n_mlp_layers]...,
#         Dense(h_size, 1), 
#         vec)

#     model = WaveControlModel(wave_encoder, design_encoder, iter, mlp)
#     return model
# end

# function build_hypernet_model(;
#         grid_size,
#         elements,
#         pml_width,
#         pml_scale,
#         ambient_speed,
#         dt,
#         steps,
#         h_channels,
#         design_input_size,
#         h_size,
#         act
#         )

#     ## Defining integrator
#     latent_dim = OneDim(grid_size, elements)
#     grad = build_gradient(latent_dim)
#     pml = build_pml(latent_dim, pml_width, pml_scale)
#     bc = dirichlet(latent_dim)
#     latent_dynamics = ForceLatentDynamics(ambient_speed, 1.0f0, grad, pml, bc)
#     iter = Integrator(runge_kutta, latent_dynamics, 0.0f0, dt, steps)

#     ## Model for producing fields over the latent dim
#     wave_emb = Chain(
#         Dense(1, h_size, act), 
#         Dense(h_size, h_size, act), 
#         Dense(h_size, h_size, act), 
#         Dense(h_size, h_size, act),
#         Dense(h_size, 2, identity))

#     ps, re = destructure(wave_emb)

#     wave_encoder = Chain(
#         WaveEncoder(1, h_channels, 1, tanh),
#         Dense(1024, h_size, act),
#         Dense(h_size, length(ps), bias = false), ## sometimes the weight matrix is too big and bson fails
#         vec, 
#         re,
#         Field(latent_dim),
#         LatentWaveActivation(ambient_speed))
        
#     design_encoder = Chain(
#         Dense(design_input_size, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, length(ps), bias = false),
#         vec,
#         re,
#         Field(latent_dim),
#         LatentDesignActivation())

#     mlp = Chain(
#         flatten,
#         Dense(4 * elements, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, 1),
#         vec
#         )

#     return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
# end


# function build_hypernet_control_model(;
#         h_size::Int, 
#         act::Function, 
#         latent_dim::OneDim, 
#         ambient_speed::Float32,
#         design_action_size::Int,
#         dt::Float32,
#         steps::Int,
#         pml_width::Float32,
#         pml_scale::Float32
#         )

#     wave_embedder = Chain(
#         Dense(1, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, 3),
#         LatentWaveActivation(ambient_speed)
#     )

#     wave_embedder_ps, wave_embedder_restructure = destructure(wave_embedder)

#     wave_embedder_restructure(wave_embedder_ps)

#     wave_encoder = Chain(
#         SingleImageInput(),
#         MeanPool((2, 2)), ## Initial dimentionality reduction
#         DownBlock(2, 1, 16, act),
#         InstanceNorm(16),
#         DownBlock(2, 16, 32, act),
#         InstanceNorm(32),
#         DownBlock(2, 32, 64, act),
#         InstanceNorm(64),
#         DownBlock(2, 64, 128, act),
#         GlobalMeanPool(),
#         flatten,
#         Dense(128, 256, act),
#         Dense(256, length(wave_embedder_ps), bias = false),
#         vec,
#         wave_embedder_restructure,
#         Field(latent_dim))

#     design_embedder = Chain(
#         Dense(1, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, 2),
#         LatentDesignActivation()
#     )

#     design_embedder_ps, design_embedder_restructure = destructure(design_embedder)

#     design_encoder = Chain(
#         Dense(design_action_size, h_size, act),
#         LayerNorm(h_size),
#         Dense(h_size, h_size, act),
#         LayerNorm(h_size),
#         Dense(h_size, length(design_embedder_ps), bias = false),
#         vec,
#         design_embedder_restructure,
#         Field(latent_dim)
#     )

#     grad = build_gradient(latent_dim)
#     pml = build_pml(latent_dim, pml_width, 1.0f0)
#     bc = dirichlet(latent_dim)
#     dynamics = ForceLatentDynamics(AIR, pml_scale, grad, pml, bc)
#     iter = Integrator(runge_kutta, dynamics, 0.0f0, dt, steps)

#     mlp = Chain(
#         flatten,
#         Dense(4 * size(latent_dim, 1), h_size, gelu),
#         Dense(h_size, h_size, gelu),
#         Dense(h_size, 1),
#         vec
#     )

#     return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
# end


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################

# function build_hypernet_speed_model(;
#         h_size::Int, 
#         act::Function, 
#         latent_dim::OneDim, 
#         ambient_speed::Float32,
#         freq::Float32,
#         design_action_size::Int,
#         dt::Float32,
#         steps::Int,
#         pml_width::Float32,
#         pml_scale::Float32
#         )

#     wave_embedder = Chain(
#         Dense(1, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, 3),
#         LatentWaveActivation(ambient_speed)
#     )

#     wave_embedder_ps, wave_embedder_restructure = destructure(wave_embedder)

#     wave_embedder_restructure(wave_embedder_ps)

#     wave_encoder = Chain(
#         SingleImageInput(),
#         MeanPool((2, 2)), ## Initial dimentionality reduction
#         DownBlock(2, 1, 16, act),
#         InstanceNorm(16),
#         DownBlock(2, 16, 32, act),
#         InstanceNorm(32),
#         DownBlock(2, 32, 64, act),
#         InstanceNorm(64),
#         DownBlock(2, 64, 128, act),
#         GlobalMeanPool(),
#         flatten,
#         Dense(128, 512, act),
#         Dense(512, length(wave_embedder_ps), bias = false),
#         vec,
#         wave_embedder_restructure,
#         Field(latent_dim))

#     design_embedder = Chain(
#         Dense(1, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, 1),
#         vec,
#         # c -> 0.5 .+ softplus.(c)
#         # c -> 1.0f0 .+ elu.(c)
#         c -> 5.0f0 * sigmoid.(c)
#     )

#     design_embedder_ps, design_embedder_restructure = destructure(design_embedder)

#     design_encoder = Chain(
#         Dense(design_action_size, h_size, act),
#         LayerNorm(h_size),
#         Dense(h_size, h_size, act),
#         LayerNorm(h_size),
#         Dense(h_size, length(design_embedder_ps), bias = false),
#         vec,
#         design_embedder_restructure,
#         Field(latent_dim)
#     )

#     pml = build_pml(latent_dim, pml_width, pml_scale)
#     grad = build_gradient(latent_dim)
#     bc = dirichlet(latent_dim)
#     dynamics = LatentDynamics(ambient_speed, freq, pml, grad, bc)

#     iter = Integrator(runge_kutta, dynamics, 0.0f0, dt, steps)

#     mlp = Chain(
#         flatten,
#         Dense(4 * size(latent_dim, 1), h_size, act),
#         Dense(h_size, h_size, act),
#         Dense(h_size, 1),
#         vec
#     )

#     return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
# end