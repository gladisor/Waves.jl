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

function (encoder::WaveEncoder)(x)
    z = encoder.layers(x)
    zs = [z[:, :, i] for i ∈ axes(z, 3)]
    latents = integrate.([encoder.cell], zs, [encoder.dynamics], [encoder.steps])
    return latents
end

dim = TwoDim(5.0f0, 0.025f0)
pulse = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)
wave = zeros(Float32, size(dim)..., 6)
wave = pulse(wave)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)
dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)

n = 100
u = integrate(cell, wave, dynamics, n)
pushfirst!(u, wave)
t = collect(range(0.0f0, dynamics.dt * n, n + 1))
sol = WaveSol(dim, t, displacement.(u))

z_grid_size = 5.0f0
z_elements = 100
z_fields = 2

layers = Chain(
    Conv((4, 4), 1 => 1, relu, pad = SamePad()), MaxPool((4, 4)),
    Conv((4, 4), 1 => 1, relu, pad = SamePad()), MaxPool((4, 4)),
    Flux.flatten,
    Dense(625, z_elements * z_fields, tanh),
    z -> reshape(z, z_elements, z_fields, :)
    )

z_cell = WaveCell(latent_wave, runge_kutta)
z_dim = OneDim(z_grid_size, z_elements)
z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...)
encoder = WaveEncoder(z_cell, z_dynamics, 6, layers)

x = cat(sol.u[1:3]..., dims = 4)

println(size(x))
latents = encoder(x)

ps = Flux.params(encoder)

# ps = Flux.params(wave)

# gs = Flux.gradient(ps) do 
#     latents = integrate(cell, wave, dynamics, 1)
#     e = sum(energy(displacement(latents[end])))

#     Flux.ignore() do 
#         println(e)
#     end

#     return e
# end

# p = WavePlot(dim)
# # lines!(p.ax, dim.x, displacement(gs[wave]))
# heatmap!(p.ax, dim.x, dim.y, displacement(gs[wave]))
# save("u.png", p.fig)
