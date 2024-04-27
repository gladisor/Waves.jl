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
    pml_scale::Float32)
    
    num_basis_functions = 16
    num_g_functions = 16
    basis_matrix = generate_basis(size(latent_dim.x)[1], num_basis_functions)
    coefficients = generate_coefficients(num_basis_functions, num_g_functions)

    return LatentTransformationModel(AcousticEnergyModel(;env, h_size, in_channels, nfreq, pml_width, pml_scale, latent_dim), basis_matrix, coefficients)
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
    basis_matrix = reshape(vcat(basis_functions...), axis_length, num_basis_functions)
    return basis_matrix
end

function generate_coefficients(num_basis_functions::Int, g_functions_num::Int)
    reshape(vcat([randn(Float32, num_basis_functions) ./ Float32(sqrt(num_basis_functions)) for _ in 1:g_functions_num]...), num_basis_functions, g_functions_num)
end

# g_functions_num = 10
# num_basis_functions = 16
# latent_space_axis_length = 64
# basis_matrix = reshape(vcat(generate_basis_functions(latent_space_axis_length, num_basis_functions)...), latent_space_axis_length, num_basis_functions)
# coefs_matrix = reshape(vcat([randn(Float32, num_basis_functions) ./ Float32(sqrt(num_basis_functions)) for _ in 1:g_functions_num]...), num_basis_functions, g_functions_num)

# g_functions_matrix = basis_matrix * coefs_matrix

# G = sum(g_functions_matrix, dims=2)

