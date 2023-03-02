using Waves
using CairoMakie
using Flux
using Statistics: mean
using Flux.Optimisers: Restructure

function latent_wave(wave::AbstractMatrix{Float32}, t::Float32, dyn::WaveDynamics)
    U = selectdim(wave, 2, 1)
    Vx = selectdim(wave, 2, 2)

    ∇ = dyn.grad
    σx = dyn.pml

    dU = ∇ * Vx .- σx .* U
    dVx = ∇ * U .- σx .* Vx

    return cat(dU, dVx, dims = 2)
end

function wave_cell(z::AbstractMatrix{Float32}, dyn::WaveDynamics)
    z′ = z .+ runge_kutta(latent_wave, z, dyn)
    dyn.t += 1
    return z′, z′
end

struct WaveNet
    encoder::Chain
    restructure::Restructure
    hypernet::Chain
end

function WaveNet(dim::AbstractDim, z_size::Int, z_fields::Int)

    encoder = Chain(
        Conv((4, 4), 1 => 1, pad = SamePad(), tanh),
        MaxPool((4, 4)),
        Conv((4, 4), 1 => 1, pad = SamePad(), tanh),
        MaxPool((4, 4)),
        Flux.flatten,
        Dense(144, z_size * z_fields, tanh),
        z -> reshape(z, z_size, z_fields, :))

    Φ = Chain(Dense(length(size(dim)), 128, relu), Dense(128, 1))
    _, restructure = Flux.destructure(Φ)
    hypernet = Chain(Dense(z_size * z_fields, 128, relu), Dense(128, length(restructure)))

    return WaveNet(encoder, restructure, hypernet)
end

dim = OneDim(5.0f0, 0.05f0)
pulse = Pulse(dim, 0.0f0, 10.0f0)
wave = zeros(Float32, size(dim)..., 2)
wave = pulse(wave)

cyl = Cylinder([-2.0f0, 0.0f0], 0.5f0, 0.0f0)

kwargs = Dict(
    :pml_width => 1.0f0, 
    :pml_scale => 100.0f0,
    :ambient_speed => 2.0f0,
    :dt => 0.01f0)

dyn = WaveDynamics(dim = dim, design = cyl; kwargs...)

# iter = WaveIntegrator(wave, split_wave_pml, runge_kutta, dyn)
# n = 600
# @time sol = integrate(iter, n)
# @time render!(sol, path = "vid.mp4")

p = WavePlot(dim)
lines!(p.ax, dim.x, displacement(wave))
save("ui.png", p.fig)

iter = WaveIntegrator(wave, latent_wave, runge_kutta, dyn)
Waves.reset!(iter)
sol = integrate(iter, 30)

p = WavePlot(dim)
lines!(p.ax, dim.x, sol.u[end])
save("uf.png", p.fig)

ps = Flux.params(wave)
opt = Descent(0.01)

for i ∈ 1:100
    gs = Flux.gradient(ps) do

        m = Flux.Recur(wave_cell, wave)
        latents = [m(dyn) for i ∈ 1:30]
        e = sum(energy(displacement(latents[end])))

        Flux.ignore() do 
            println(e)
        end

        return e
    end

    Flux.Optimise.update!(opt, ps, gs)
end

p = WavePlot(dim)
lines!(p.ax, dim.x, displacement(wave))
save("optimized_ui.png", p.fig)

iter = WaveIntegrator(wave, latent_wave, runge_kutta, dyn)
Waves.reset!(iter)
sol = integrate(iter, 30)

p = WavePlot(dim)
lines!(p.ax, dim.x, sol.u[end])
save("optimized_uf.png", p.fig)

nodes = 100
fields = 2
z_dim = OneDim(5.0f0, nodes)
z_dyn = WaveDynamics(dim = z_dim; kwargs...)

# x = cat(sol.u..., dims = 3)
# x = reshape(x, size(x, 1), size(x, 2), 1, size(x, 3))
# x = x[:, :, 1, 1:10]

# g = grid(dim)
# points = reshape(g, :, 2)

# opt = Adam(0.01)
# ps = Flux.params(encoder, Φ)

# for i ∈ 1:100

#     Waves.reset!(z_dyn)

#     gs = Flux.gradient(ps) do

#         z = dropdims(encoder(x[:, :, 1, :, :]), dims = 3)
#         m = Flux.Recur(wave_rnn, z)
#         latents = cat([m(z_dyn) for _ ∈ 1:size(x, 3)]..., dims = 3)
#         latents = Flux.flatten(latents)
#         θ = Ψ(latents)
#         Φ = restructure.(eachcol(θ)) ## restructuring weights into Φ
#         û = cat(map(Φ -> Φ(points'), Φ)..., dims = 3) ## evaluating Φ on the domain
#         û = reshape(û, size(dim)..., :)
#         loss = mean((x .- û) .^ 2)

#         Flux.ignore() do 
#             println("Loss: $loss")
#         end
        
#         return loss
#     end

#     Flux.Optimise.update!(opt, ps, gs)
# end

# Waves.reset!(z_dyn)
# z = dropdims(encoder(x[:, :, 1, :, :]), dims = 3)
# z_iter = WaveIntegrator(z, split_wave, runge_kutta, z_dyn)
# @time z_sol = integrate(z_iter, n)
# @time render!(z_sol, path = "z_vid.mp4")
