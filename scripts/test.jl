using SparseArrays
using GLMakie
using DifferentialEquations
using LinearAlgebra
using Waves

function Waves.OneDim(grid_size::Float64, n::Int)
    return OneDim(collect(range(-grid_size, grid_size, n)))
end

function Waves.TwoDim(grid_size::Float64, n::Int)
    return TwoDim(
        collect(range(-grid_size, grid_size, n)),
        collect(range(-grid_size, grid_size, n)))
end

"""
Function for constructing a gradient operator for a one dimensional scalar field.
"""
function gradient(x::Vector)
    grad = zeros(size(x, 1), size(x, 1))
    Δ = (x[end] - x[1]) / (length(x) - 1)

    grad[[1, 2, 3], 1] .= [-3.0, 4.0, -1.0] ## left boundary edge case
    grad[[end-2, end-1, end], end] .= [1.0, -4.0, 3.0] ## right boundary edge case

    for i ∈ 2:(size(grad, 2) - 1)
        grad[[i - 1, i + 1], i] .= [-1.0, 1.0]
    end

    return sparse((grad / (2 * Δ))')
end

function pulse(dim::OneDim, x = 0.0, intensity = 1.0)
    return exp.(- intensity * (dim.x .- x) .^ 2)
end

function pulse(dim::TwoDim, x = 0.0, y = 0.0, intensity = 1.0)
    u = zeros(length(dim.x), length(dim.y))

    for i ∈ axes(u, 1)
        for j ∈ axes(u, 2)
            u[i, j] = exp(- intensity * ((dim.x[i] - x) .^ 2 + (dim.y[j] - y) .^ 2))
        end
    end

    return u
end

function wave!(du::Matrix{Float64}, u::Matrix{Float64}, p, t::Float64)
    grad, pml = p

    U = u[:, 1]
    V = u[:, 2]

    du[:, 1] .= grad * V .- U .* pml
    du[:, 2] .= grad * U .- V .* pml
end

function wave!(du::Array{Float64, 3}, u::Array{Float64, 3}, p, t::Float64)
    grad, C, pml = p

    U = u[:, :, 1]
    Vx = u[:, :, 2]
    Vy = u[:, :, 3]

    du[:, :, 1] .= C(t) .* (grad * Vx .+ (grad * Vy')') .- U .* pml
    du[:, :, 2] .= (grad * U) .- Vx .* pml
    du[:, :, 3] .= (grad * U')' .- Vy .* pml
end

gs = 10.0
n = 300

dim = TwoDim(gs, n)
grad = gradient(dim.x)
u = pulse(dim)
u0 = cat(u, zeros(size(u)..., 2), dims = 3)

pml = build_pml(dim, 4.0) * 10.0
prob = ODEProblem(wave!, u0, (0.0, 10.0), [grad, 2.0, pml])
sol = solve(prob, Midpoint())
wavesol = Waves.interpolate(sol, dim, 0.05)
Waves.render!(wavesol, path = "vid.mp4")

# dx_u = (grad * u')'
# fig = Figure(resolution = (1920, 1080))
# ax = Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5)
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, dim.y[1], dim.y[end])
# zlims!(ax, -1.0, 4.0)
# surface!(ax, dim.x, dim.y, pml, colormap = :ice)
# save("pulse.png", fig)