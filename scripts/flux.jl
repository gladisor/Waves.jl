using Waves
using CairoMakie
using Flux
using Statistics: mean
using Flux.Optimisers: Restructure

function latent_wave(wave::AbstractMatrix{Float32}, t::Float32, dynamics::WaveDynamics)
    U = selectdim(wave, 2, 1)
    V = selectdim(wave, 2, 2)

    ∇ = dynamics.grad
    σx = dynamics.pml
    C = dynamics.ambient_speed
    b = C ^ 2

    dU = b * ∇ * V .- σx .* U
    dV = ∇ * U .- σx .* V

    return cat(dU, dV, dims = 2)
end

struct WaveEncoder
    cell::WaveCell
    dynamics::WaveDynamics
    steps::Int
    layers::Chain
end

Flux.@functor WaveEncoder
Flux.trainable(encoder::WaveEncoder) = (encoder.layers,)

function (encoder::WaveEncoder)(wave::AbstractArray{Float32, 3})
    z = dropdims(encoder.layers(Flux.batch([wave])), dims = 3)
    latents = integrate(encoder.cell, z, encoder.dynamics, encoder.steps)
    latents = Flux.flatten(cat(latents..., dims = 3))
    return latents
end

function (encoder::WaveEncoder)(sol::WaveSol{TwoDim})
    return encoder(first(sol.u))
end

struct WaveDecoder
    dim::TwoDim
    points::AbstractMatrix{Float32}
    fields::Int
    restructure::Restructure
    hypernetwork::Chain
end

Flux.@functor WaveDecoder
Flux.trainable(decoder::WaveDecoder) = (decoder.hypernetwork,)

function WaveDecoder(dim::AbstractDim, in_size::Int, h_size::Int, fields::Int)

    points = reshape(grid(dim), :, 2)'
    Φ = Chain(
        Dense(ndims(dim), h_size, relu), 
        Dense(h_size, h_size, relu),
        Dense(h_size, fields))

    _, restructure = Flux.destructure(Φ)
    hypernetwork = Chain(Dense(in_size, h_size, relu), Dense(h_size, length(restructure), bias = false))
    return WaveDecoder(dim, points, fields, restructure, hypernetwork)
end

function (decoder::WaveDecoder)(z)
    θ = decoder.hypernetwork(z)
    Φ = decoder.restructure.(eachcol(θ))
    return map(Φ -> reshape(Φ(decoder.points), size(decoder.dim)..., decoder.fields), Φ)
end

function training_data(sol::WaveSol, k::Int)
    x = []
    y = []

    for i ∈ 1:(length(sol) - k)
        push!(x, sol.u[i])
        push!(y, sol.u[i+1:i+k])
    end

    return (x, y)
end

dim = TwoDim(5.0f0, 0.025f0)
pulse = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)
pulse2 = Pulse(dim, -3.0f0, 0.0f0, 5.0f0)
wave = zeros(Float32, size(dim)..., 6)
wave = pulse(wave) .+ pulse2(wave)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)

dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)

n = 10
@time u = integrate(cell, wave, dynamics, n)
pushfirst!(u, wave) ## add the initial state
t = collect(range(0.0f0, dynamics.dt * n, n + 1))
sol = WaveSol(dim, t, u)

z_grid_size = 5.0f0
z_elements = 100
z_fields = 2

z_cell = WaveCell(latent_wave, runge_kutta)
z_dim = OneDim(z_grid_size, z_elements)
z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...)

layers = Chain(
    Conv((4, 4), 6 => 1, relu, pad = SamePad()), MaxPool((4, 4)),
    Conv((4, 4), 1 => 1, relu, pad = SamePad()), MaxPool((4, 4)),
    Flux.flatten,
    Dense(625, z_elements * z_fields, relu),
    z -> reshape(z, z_elements, z_fields, :))

encoder = WaveEncoder(z_cell, z_dynamics, length(sol)-1, layers) |> gpu
decoder = WaveDecoder(dim, z_elements * z_fields, 128, 6) |> gpu

ps = Flux.params(encoder, decoder)
opt = Adam(0.01)

u_true = cat(sol.u[2:end]..., dims = 4)

for i ∈ 1:50
    
    gs = Flux.gradient(ps) do

        latents = encoder(sol)
        u_pred = cat(decoder(latents)..., dims = 4)
        loss = mean((u_pred .- u_true) .^ 2)

        Flux.ignore() do 
            println("Loss: $loss")
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)

    p = WavePlot(dim)
    plot_wave!(p, dim, decoder(encoder(sol))[end])
    save("training/u$i.png", p.fig)
end