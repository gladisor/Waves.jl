using Waves
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

function dirichlet(dim::OneDim)
    bc = ones(Float32, size(dim)...)
    bc[[1, end]] .= 0.0f0
    return bc
end

function dirichlet(dim::TwoDim)
    bc = ones(Float32, size(dim)...)
    bc[[1, end], :] .= 0.0f0
    bc[:, [1, end]] .= 0.0f0
    return bc
end

function split_wave(wave::AbstractMatrix{Float32}, t::Float32, dyn::WaveDynamics)
    # bc = selectdim(wave, 2, 3)
    U = selectdim(wave, 2, 1)# .* bc
    Vx = selectdim(wave, 2, 2)

    ∇ = dyn.grad
    σx = dyn.pml

    dU = ∇ * Vx .- σx .* U
    dVx = ∇ * U .- σx .* Vx

    return cat(dU, dVx, dims = 2)
end

function reset!(dyn::WaveDynamics)
    dyn.t = 0
end

function reset!(iter::WaveIntegrator)
    reset!(iter.dyn)
end

function Waves.displacement(sol::WaveSol{TwoDim})
    return cat(sol.u..., dims = 3)
end

function wave_rnn(z::Matrix{Float32}, dyn::WaveDynamics)
    z′ = z .+ runge_kutta(split_wave, z, dyn)
    dyn.t += 1
    return z′, z′
end

dim = TwoDim(5.0f0, 0.05f0)
pulse = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)
wave = zeros(Float32, size(dim)..., 6)
wave = pulse(wave)

kwargs = Dict(
    :pml_width => 1.0f0, 
    :pml_scale => 100.0f0,
    :ambient_speed => 2.0f0,
    :dt => 0.01f0)

dyn = WaveDynamics(dim = dim; kwargs...)
iter = WaveIntegrator(wave, split_wave_pml, runge_kutta, dyn)

n = 200
@time sol = integrate(iter, n)
@time render!(sol, path = "vid.mp4")

nodes = 100
fields = 2
z_dim = OneDim(5.0f0, nodes)
z_dyn = WaveDynamics(dim = z_dim; kwargs...)

encoder = Chain(
    Conv((4, 4), 1 => 1, pad = SamePad(), relu),
    MaxPool((4, 4)),
    Conv((4, 4), 1 => 1, pad = SamePad(), relu),
    MaxPool((4, 4)),
    Flux.flatten,
    Dense(144, nodes * fields, tanh),
    x -> reshape(x, nodes, fields, :)
    )

Φ = Chain(
    Dense(2, 128, relu),
    Dense(128, 1))

_, restructure = Flux.destructure(hypernet)

Ψ = Chain(
    Dense(nodes * fields, 128, relu),
    Dense(128, length(restructure)))

x = cat(sol.u..., dims = 3)
x = reshape(x, size(x, 1), size(x, 2), 1, size(x, 3))
x = x[:, :, 1, 1:10]

g = grid(dim)
points = reshape(g, :, 2)

opt = Adam(0.01)
ps = Flux.params(encoder, decoder)

for i ∈ 1:100

    reset!(z_dyn)

    gs = Flux.gradient(ps) do

        z = dropdims(encoder(x[:, :, 1, :, :]), dims = 3)
        m = Flux.Recur(wave_rnn, z)
        latents = cat([m(z_dyn) for _ ∈ 1:size(x, 3)]..., dims = 3)
        latents = Flux.flatten(latents)
        θ = Ψ(latents)
        Φ = restructure.(eachcol(θ)) ## restructuring weights into Φ
        û = cat(map(Φ -> Φ(points'), Φ)..., dims = 3) ## evaluating Φ on the domain
        û = reshape(û, size(dim)..., :)
        loss = mean((x .- û) .^ 2)

        Flux.ignore() do 
            println("Loss: $loss")
        end
        
        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end

reset!(z_dyn)
z = dropdims(encoder(x[:, :, 1, :, :]), dims = 3)
z_iter = WaveIntegrator(z, split_wave, runge_kutta, z_dyn)
@time z_sol = integrate(z_iter, n)
@time render!(z_sol, path = "z_vid.mp4")
