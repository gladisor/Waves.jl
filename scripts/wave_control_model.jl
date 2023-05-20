function visualize_predictions!(model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0, title = "Prediction of Scattered Energy Versus Ground Truth")

    sigma_pred = model(s, a)

    for i in axes(tspan, 2)
        lines!(ax, cpu(tspan[:, i]), cpu(sigma[:, i]), color = :blue, label = "True")
        lines!(ax, cpu(tspan[:, i]), cpu(sigma_pred[:, i]), color = :orange, label = "Predicted")
    end

    save(path, fig)
end

function visualize!(model::WaveControlModel, dim::OneDim, s::WaveEnvState, actions::Vector{<: AbstractDesign}, tspan::AbstractMatrix, sigma::AbstractMatrix; path::String)

    tspan = cpu(tspan)
    tspan_flat = vcat(tspan[1], vec(tspan[2:end, :]))

    z_wave = model.wave_encoder(s.wave_total)
    design = s.design

    zs = []
    for (i, a) in enumerate(actions)
        z, z_wave, design = propagate(model, z_wave, design, a)

        if i == 1
            push!(zs, cpu(z))
        else
            push!(zs, cpu(z[:, :, 2:end]))
        end
    end

    z = cat(zs..., dims = ndims(zs[1]))
    pred_sigma = cpu(model(s, a))

    fig = Figure(resolution = (1920, 1080), fontsize = 30)
    z_grid = fig[1, 1] = GridLayout()

    ax1, hm1 = heatmap(z_grid[1, 1], dim.x, tspan_flat, z[:, 1, :], colormap = :ice)
    Colorbar(z_grid[1, 2], hm1)
    ax1.title = "Displacement"
    ax1.ylabel = "Time (s)"
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false

    ax2, hm2 = heatmap(z_grid[1, 3], dim.x, tspan_flat, z[:, 2, :], colormap = :ice)
    Colorbar(z_grid[1, 4], hm2)
    ax2.title = "Velocity"
    ax2.xticklabelsvisible = false
    ax2.xticksvisible = false
    ax2.yticklabelsvisible = false
    ax2.yticksvisible = false

    ax3, hm3 = heatmap(z_grid[2, 1], dim.x, tspan_flat, z[:, 3, :], colormap = :ice)
    Colorbar(z_grid[2, 2], hm3)
    ax3.title = "Force"
    ax3.xlabel = "Distance (m)"
    ax3.ylabel = "Time (s)"

    ax4, hm4 = heatmap(z_grid[2, 3], dim.x, tspan_flat, z[:, 4, :], colormap = :ice)
    Colorbar(z_grid[2, 4], hm4)
    ax4.title = "Wave Speed"
    ax4.xlabel = "Distance (m)"
    ax4.yticklabelsvisible = false
    ax4.yticksvisible = false

    p_grid = fig[1, 2] = GridLayout()
    p_axis = Axis(p_grid[1, 1], title = "Prediction of Scattered Energy Versus Ground Truth", xlabel = "Time (s)", ylabel = "Scattered Energy (σ)")

    for i in axes(tspan, 2)
        lines!(p_axis, tspan[:, i], cpu(sigma[:, i]), color = :blue, label = "True", linewidth = 3)
        lines!(p_axis, tspan[:, i], pred_sigma[:, i], color = :orange, label = "Predicted", linewidth = 3)
    end

    save(joinpath(path, "latent.png"), fig)
    render!(dim, z, path = joinpath(path, "latent.mp4"))
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

struct FrequencyDomain
    domain::AbstractMatrix{Float32}
end

Flux.@functor FrequencyDomain
Flux.trainable(::FrequencyDomain) = (;)

function FrequencyDomain(dim::OneDim, nfreq::Int)
    dim = cpu(dim)
    L = dim.x[end] - dim.x[1]
    frequencies = (Float32.(collect(1:nfreq)) .* dim.x') / L
    domain = vcat(sin.(2.0f0 * pi * frequencies), cos.(2.0f0 * pi * frequencies))
    return FrequencyDomain(domain)
end

function (freq::FrequencyDomain)(m)
    return permutedims(m(freq.domain), (2, 1))
    # return m(freq.domain)
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

function LatentDynamics(dim::OneDim; ambient_speed, freq, pml_width, pml_scale)
    pml = build_pml(dim, pml_width, pml_scale)
    grad = build_gradient(dim)
    bc = dirichlet(dim)
    return LatentDynamics(ambient_speed, freq, pml, grad, bc)
end

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