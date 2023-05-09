function visualize_latent!(dim::OneDim, tspan::AbstractMatrix, model, s::WaveEnvState, a::Vector{<: AbstractDesign}; path::String)
    zi = encode(model, s, a[1])
    @time z = model.iter(zi)

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0f0, title = "Displacement", ylabel = "Time (s)", xlabel = "Space (m)")
    heatmap!(ax1, dim.x, tspan[:, 1], cpu(z[:, 1, :]), colormap = :ice)

    ax2 = Axis(fig[1, 2], aspect = 1.0f0, title = "Velocity", xlabel = "Space (m)")
    heatmap!(ax2, dim.x, tspan[:, 1], cpu(z[:, 2, :]), colormap = :ice)
    save(path, fig)
end

function visualize_predictions!(tspan::AbstractMatrix, sigma::AbstractMatrix, model, s::WaveEnvState, a::Vector{<: AbstractDesign}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0, title = "Prediction of Scattered Energy Versus Ground Truth")

    sigma_pred = model(s, a)

    for i in axes(tspan, 2)
        lines!(ax, tspan[:, i], cpu(sigma[:, i]), color = :blue, label = "True")
        lines!(ax, tspan[:, i], cpu(sigma_pred[:, i]), color = :orange, label = "Predicted")
    end

    save(path, fig)
end

function train(model::WaveControlModel, train_loader::DataLoader, val_loader::DataLoader, epochs::Int, lr; path::String, checkpoint_every::Int = 3, evaluation_samples = 3)
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
        save(joinpath(path, "loss.png"), fig)

        if i % checkpoint_every == 0 || (i == 1 || i == epochs)
            checkpoint_path = mkpath(joinpath(path, "model$i"))
            println("Checkpointing")
            save(model, checkpoint_path)

            println("Rendering Images")
            @showprogress for (j, (s, a, t, σ)) in enumerate(val_loader)
                s, a, t, σ = gpu(s[1]), gpu(a[1]), t[1], gpu(σ[1])
                visualize_latent!(latent_dim, t, model, s, a, path = joinpath(checkpoint_path, "latent$j.png"))
                visualize_predictions!(t, σ, model, s, a, path = joinpath(checkpoint_path, "predictions$j.png"))
                j == evaluation_samples && break
            end
        end
    end

    return model
end

function build_wave_control_model(;
        in_channels,
        h_channels,
        design_size,
        action_size,
        h_size, 
        latent_grid_size,
        latent_elements,
        latent_pml_width, 
        latent_pml_scale, 
        ambient_speed,
        dt,
        steps,
        n_mlp_layers,
        )

    wave_encoder = Chain(
        WaveEncoder(in_channels, h_channels, 2, tanh),
        Dense(1024, latent_elements, tanh),
        z -> hcat(z[:, 1], z[:, 2] / ambient_speed)
        )

    design_encoder = Chain(
        Dense(design_size + action_size, h_size, relu),
        Dense(h_size, 2 * latent_elements),
        z -> reshape(z, latent_elements, :),
        z -> hcat(tanh.(z[:, 1]), sigmoid.(z[:, 2]))
        )

    latent_dim = OneDim(latent_grid_size, latent_elements)
    grad = build_gradient(latent_dim)
    pml = build_pml(latent_dim, latent_pml_width, latent_pml_scale)
    bc = dirichlet(latent_dim)

    latent_dynamics = ForceLatentDynamics(ambient_speed, 1.0f0, grad, pml, bc)
    iter = Integrator(runge_kutta, latent_dynamics, 0.0f0, dt, steps)

    mlp = Chain(
        flatten,
        Dense(latent_elements * 4, h_size, relu), 
        [Dense(h_size, h_size, relu) for _ in 1:n_mlp_layers]...,
        Dense(h_size, 1), 
        vec)

    model = WaveControlModel(wave_encoder, design_encoder, iter, mlp)
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
    return hcat(u, v)
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
    return m(field.dim.x')
end

function build_hypernet_model(;
        grid_size,
        elements,
        pml_width,
        pml_scale,
        ambient_speed,
        dt,
        steps,
        h_channels,
        design_input_size,
        h_size,
        act
        )

    ## Defining integrator
    latent_dim = OneDim(grid_size, elements)
    grad = build_gradient(latent_dim)
    pml = build_pml(latent_dim, pml_width, pml_scale)
    bc = dirichlet(latent_dim)
    latent_dynamics = ForceLatentDynamics(ambient_speed, 1.0f0, grad, pml, bc)
    iter = Integrator(runge_kutta, latent_dynamics, 0.0f0, dt, steps)

    ## Model for producing fields over the latent dim
    wave_emb = Chain(
        Dense(1, h_size, act), 
        Dense(h_size, h_size, act), 
        Dense(h_size, h_size, act), 
        Dense(h_size, h_size, act),
        Dense(h_size, 2, identity))

    ps, re = destructure(wave_emb)

    wave_encoder = Chain(
        WaveEncoder(1, h_channels, 1, tanh),
        Dense(1024, h_size, act),
        Dense(h_size, length(ps), bias = false), ## sometimes the weight matrix is too big and bson fails
        vec, 
        re,
        Field(latent_dim),
        LatentWaveActivation(ambient_speed))
        
    design_encoder = Chain(
        Dense(design_input_size, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, length(ps), bias = false),
        vec,
        re,
        Field(latent_dim),
        LatentDesignActivation())

    mlp = Chain(
        flatten,
        Dense(4 * elements, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, 1),
        vec
        )

    return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
end

function FileIO.save(model::WaveControlModel, path::String)
    BSON.bson(joinpath(path, "wave_encoder.bson"), wave_encoder = cpu(model.wave_encoder))
    BSON.bson(joinpath(path, "design_encoder.bson"), design_encoder = cpu(model.design_encoder))
    BSON.bson(joinpath(path, "iter.bson"), iter = cpu(model.iter))
    BSON.bson(joinpath(path, "mlp.bson"), mlp = cpu(model.mlp))
end

function WaveControlModel(;path::String)
    wave_encoder = BSON.load(joinpath(path, "wave_encoder.bson"))[:wave_encoder]
    design_encoder = BSON.load(joinpath(path, "design_encoder.bson"))[:design_encoder]
    iter = BSON.load(joinpath(path, "iter.bson"))[:iter]
    mlp = BSON.load(joinpath(path, "mlp.bson"))[:mlp]
    return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
end