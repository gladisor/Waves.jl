struct LatentTransformationModel
    inner_model::AcousticEnergyModel
    basis::AbstractMatrix
    coefficients::AbstractMatrix
end

Flux.@functor LatentTransformationModel
Flux.trainable(model::LatentTransformationModel) = (;model.inner_model, model.coefficients)
# Flux.trainable(model.inner_model::AcousticEnergyModel) = (;model.inner_model.wave_encoder, model.inner_model.design_encoder, model.inner_model.F)

function LatentTransformationModel(;
    env::WaveEnv, 
    latent_dim::OneDim,
    in_channels::Int,
    h_size::Int, 
    nfreq::Int, 
    pml_width::Float32,
    pml_scale::Float32,
    base_function::Function = build_cnn_base
    )
    
    num_basis_functions = 16
    num_g_functions = 16
    basis_matrix = generate_basis(size(latent_dim.x)[1], num_basis_functions)
    coefficients = generate_coefficients(num_basis_functions, num_g_functions)

    return LatentTransformationModel(AcousticEnergyModel(;env, h_size, in_channels, nfreq, pml_width, pml_scale, latent_dim, base_function), basis_matrix, coefficients)
end

function (model::LatentTransformationModel)(s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = generate_latent_solution(model.inner_model, s, a, t)
    tot = z[:, 1, :, :]
    inc = z[:, 3, :, :]
    sc = tot .- inc

    tot_energy = sum(tot .^ 2, dims = 1) * model.inner_model.dx
    inc_energy = sum(inc .^ 2, dims = 1) * model.inner_model.dx
    sc_energy  = sum(sc  .^ 2, dims = 1) * model.inner_model.dx
    
    latent_energy = permutedims(vcat(tot_energy, inc_energy, sc_energy), (3, 1, 2))

    g_i_matrix = model.basis * model.coefficients
    g_i_matrix = reshape(g_i_matrix, size(g_i_matrix, 2), 1, 1, size(g_i_matrix, 1))
    sc = Flux.unsqueeze(permutedims(sc, [2, 3, 1]), 1)

    transformation = g_i_matrix .* sc
    transformed_energy = sum(transformation  .^ 2, dims = 4) * model.inner_model.dx
    transformed_energy = dropdims(transformed_energy, dims = 4)
    transformed_energy = permutedims(transformed_energy, [3, 1, 2])

    return hcat(latent_energy, transformed_energy)
end

function generate_basis(axis_length::Int, num_basis_functions::Int)
    basis_functions = []
    basis_length = Int(axis_length / num_basis_functions)
    for i in 1:basis_length:axis_length
        basis_function_i = zeros(Float32, axis_length)
        basis_function_i[i:(i + basis_length - 1)] .= 1.0f0
        push!(basis_functions, basis_function_i)
    end
    scale = sqrt(sum(basis_functions[1] .^ 2))
    basis_matrix = reshape(vcat(basis_functions...), axis_length, num_basis_functions) ./ scale
    return basis_matrix
end

function generate_coefficients(num_basis_functions::Int, g_functions_num::Int)
    reshape(vcat([randn(Float32, num_basis_functions) ./ Float32(sqrt(num_basis_functions)) for _ in 1:g_functions_num]...), num_basis_functions, g_functions_num)
end

function make_plots(
    model::LatentTransformationModel, 
    batch; path::String, 
    samples::Int = 1)

    s, a, t, y = batch
    z = cpu(generate_latent_solution(model.inner_model, s, a, t))
    latent_dim = cpu(model.inner_model.iter.dynamics.dim)
    render_latent_solution!(latent_dim, cpu(t[:, 1]), z[:, :, 1, :], path = path)

    z0, (C, F, PML) = Waves.get_parameters_and_initial_condition(model.inner_model, s, a, t)

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, latent_dim.x, cpu(PML[:, 1]))
    save(joinpath(path, "pml.png"), fig)

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, latent_dim.x, cpu(F.shape[:, 1]))
    save(joinpath(path, "force.png"), fig)

    y_hat = cpu(model(s, a, t))
    y = cpu(y)
    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        plot_predicted_energy(tspan, y[:, 1, i], y_hat[:, 1, i], title = "Total Energy", path = joinpath(path, "tot$i.png"))
        plot_predicted_energy(tspan, y[:, 2, i], y_hat[:, 2, i], title = "Incident Energy", path = joinpath(path, "inc$i.png"))
        plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end

    masked_plots_path = mkpath(joinpath(path, "masked_plots"))
    for i in 1:min(length(s), samples)
        cur_sample_path = mkpath(joinpath(masked_plots_path, "$i"))
        tspan = cpu(t[:, i])
        for j in 4:size(y_hat)[2]
            plot_predicted_energy(tspan, y[:, j, i], y_hat[:, j, i], title = "Scattered Energy in Region $j", path = joinpath(cur_sample_path, "region$j.png"))
        end
    end

    # g_plots_path = mkpath(joinpath(path, "g_plots"))
    # plot_g_functions(model, path=g_plots_path)

    return nothing
end

function plot_g_functions(model::LatentTransformationModel; path::String)
    g_matrix = model.basis * model.coefficients
    for i in 1:size(g_matrix[1,:])[1]
        plot_function(g_matrix[:, i], path=joinpath(mkpath(path), "g_$i.png"))
    end
    plot_function(vec(sum(g_matrix, dims=2)), path=joinpath(path, "G.png"))
end

function plot_function(g_function::AbstractVector; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, 1:size(g_function)[1], g_function)
    save(path, fig)
end

function build_encoder()
    
    return Metalhead.ViT(:tiny;
        inchannels = 1,
        nclasses = 3,
        imsize = (100, 100),
        patch_size = (20, 20))
end