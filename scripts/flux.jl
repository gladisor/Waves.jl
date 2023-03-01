using Waves
using Waves: field
using SparseArrays
using CairoMakie
using Flux
using Statistics: mean

function laplacian(x::Vector{Float32})
    laplace = zeros(Float32, length(x), length(x))
    dx = (x[end] - x[1]) / (length(x) - 1)

    laplace[1, [1, 2, 3, 4]] .= [2.0f0, -5.0f0, 4.0f0, -1.0f0] / dx^3
    laplace[end, [end-3, end-2, end-1, end]] .= [-1.0f0, 4.0f0, -5.0f0, 2.0f0] / dx^3

    for i ∈ 2:(size(laplace, 2) - 1)
        laplace[i, [i-1, i, i+1]] .= [1.0f0, -2.0f0, 1.0f0] / dx^2
    end

    return sparse(laplace)
end

function circle_mask(dim::TwoDim, radius::Float32)
    g = grid(dim)
    return dropdims(sum(g .^ 2, dims = 3), dims = 3) .< radius ^2
end

function flux(u::AbstractMatrix{Float32}, laplace::SparseMatrixCSC{Float32}, mask::BitMatrix)
    f = (laplace * u .+ (laplace * u')')
    return sum(f .* mask)
end

function energy(u::AbstractArray{Float32})
    return u .^ 2
end

function split_wave(wave::Wave{OneDim}, t::Float32, dyn::WaveDynamics)
    U = displacement(wave)
    Vx = field(wave, 2)

    # U[[1, end]] .= 0.0f0

    ∇ = dyn.grad
    dU = dyn.ambient_speed * ∇ * Vx
    dVx = ∇ * U

    return Wave{OneDim}(cat(dU, dVx, dims = 2))
end

function reset!(dyn::WaveDynamics)
    dyn.t = 0
end

function reset!(iter::WaveIntegrator)
    reset!(iter.dyn)
end

dim = TwoDim(5.0f0, 0.05f0)
pulse = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)
wave = Wave(dim, 6)
wave = pulse(wave)

kwargs = Dict(
    :pml_width => 1.0f0, 
    :pml_scale => 100.0f0,
    :ambient_speed => 2.0f0,
    :dt => 0.01f0)

dyn = WaveDynamics(dim = dim; kwargs...)
iter = WaveIntegrator(wave, split_wave_pml, runge_kutta, dyn)
@time sol = integrate(iter, 400)
@time render!(sol, path = "vid.mp4")

# z_dim = OneDim(1.0f0, 100)
# z_dyn = WaveDynamics(dim = z_dim; kwargs...)

# encoder = Chain(
#     Conv((4, 4), 1 => 1, pad = SamePad(), relu),
#     MaxPool((4, 4)),
#     Conv((4, 4), 1 => 1, pad = SamePad(), relu),
#     MaxPool((4, 4)),
#     Flux.flatten,
#     Dense(144, 200, relu),
#     x -> reshape(x, 100, 2, :)
#     )

# hypernet = Chain(
#     Dense(2, 128, relu),
#     Dense(128, 1))

# _, Ψ = Flux.destructure(hypernet)

# decoder = Chain(
#     Dense(200, 128, relu),
#     Dense(128, length(Ψ)))

# x = cat(sol.u..., dims = 3)
# x = reshape(x, size(x, 1), size(x, 2), 1, size(x, 3))
# x = x[:, :, :, 1, :]

# g = grid(dim)
# points = reshape(g, :, 2)

# opt = Descent(0.05)
# θ = Flux.params(encoder, decoder)

# for i ∈ 1:100

#     reset!(z_dyn)

#     gs = Flux.gradient(θ) do

#         loss = 0.0f0
#         z = Wave{OneDim}(dropdims(encoder(x), dims = 3))

#         for i ∈ 1:(length(sol) - 70)
#             Φ = Ψ(decoder(vec(z.u)))
#             û = reshape(Φ(points'), size(dim)...)
#             loss += mean((sol.u[i] .- û) .^ 2)
#             z = z + runge_kutta(split_wave, z, z_dyn)
#             z_dyn.t += 1
#         end

#         Flux.ignore() do 
#             println("Loss: $loss")
#         end
        
#         return loss
#     end

#     Flux.Optimise.update!(opt, θ, gs)
# end