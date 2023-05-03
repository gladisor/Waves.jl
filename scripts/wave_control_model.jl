function train(model::WaveControlModel, train_loader::DataLoader, val_loader::DataLoader, epochs::Int, lr)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), model)

    for i in 1:epochs

        train_loss = 0.0f0
        @showprogress for (s, a, σ) in train_loader
            s, a, σ = gpu(s[1]), gpu(a[1]), gpu(σ[1])

            loss, back = pullback(_model -> mse(_model(s, a), σ), model)
            gs = back(one(loss))[1]

            opt_state, model = Optimisers.update(opt_state, model, gs)
            train_loss += loss
        end

        val_loss = 0.0f0
        @showprogress for (s, a, σ) in val_loader
            s, a, σ = gpu(s[1]), gpu(a[1]), gpu(σ[1])
            val_loss += mse(model(s, a), σ)
        end

        train_loss = train_loss / length(train_loader)
        val_loss = val_loss / length(val_loader)

        println("Epoch: $i, Train Loss: $train_loss, Val Loss: $val_loss")
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
        z -> hcat(z[:, 1], z[:, 2] * 0.0f0)
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