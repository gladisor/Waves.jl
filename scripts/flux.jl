using Waves
using Waves: field
using SparseArrays
using CairoMakie

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

function energy(u::AbstractMatrix{Float32})
    return u .^ 2
end

function split_wave(wave::Wave{OneDim}, t::Float32, dyn::WaveDynamics)
    U = displacement(wave)
    Vx = field(wave, 2)

    U[[1, end]] .= 0.0f0

    ∇ = dyn.grad
    dU = dyn.ambient_speed * ∇ * Vx
    dVx = ∇ * U

    return Wave{OneDim}(cat(dU, dVx, dims = 2))
end

# dim = TwoDim(8.0f0, 0.05f0)
dim = OneDim(8.0f0, 0.05f0)
# pulse = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)
pulse = Pulse(dim, 0.0f0, 10.0f0)
# wave = Wave(dim, 6)
wave = Wave(dim, 2)

wave = pulse(wave)

dyn = WaveDynamics(
    dim = dim,
    pml_width = 1.0f0,
    pml_scale = 100.0f0,
    ambient_speed = 3.0f0,
    dt = 0.01f0)

iter = WaveIntegrator(
    wave, 
    # split_wave_pml,
    split_wave,
    runge_kutta,
    dyn)

@time sol = integrate(iter, 500)
@time render!(sol, path = "vid.mp4")
# mask = circle_mask(dim, 3.0f0)
# laplace = laplacian(dim.x)

# f = [flux(energy(u), laplace, mask) for u in sol.u]
# println("Plotting Flux")
# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, sol.t[1], sol.t[end])
# lines!(ax, sol.t, f, color = :blue)
# save("flux.png", fig)

# using Flux

