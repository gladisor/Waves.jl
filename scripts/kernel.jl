using GLMakie
using DifferentialEquations

using Waves
using Waves: ∇, ∇x, ∇y

mutable struct WaveParams
    Δ::Float64
    C::WaveSpeed
    pml::AbstractArray
end

function pml_2d!(du::Array{Float64, 3}, u::Array{Float64, 3}, p::WaveParams, t::Float64)
    Δ = p.Δ
    C = p.C
    pml = p.pml

    U = u[:, :, 1]
    Vx = u[:, :, 2]
    Vy = u[:, :, 3]
    Ψ = u[:, :, 4]

    du[:, :, 1] .= C(t) .* (∇x(Vx, Δ) .+ ∇y(Vy, Δ)) .- (pml .* U) .+ Ψ
    du[:, :, 2] .= ∇x(U, Δ) .- (pml .* Vx)
    du[:, :, 3] .= ∇y(U, Δ)
    du[:, :, 4] .= pml .* ∇y(Vy, Δ)
end

gs = 5.0
Δ = 0.1
C0 = 2.0
pml_width = 4.0
pml_scale = 10.0
tspan = (0.0, 20.0)
dim = OneDim(gs, Δ)
u = sin.(dim.x)
du_old = ∇(u, Δ)

grad = zeros(size(u, 1), size(u, 1))
grad[[1, 2, 3], 1] .= [-3.0, 4.0, -1.0]
grad[[end-2, end-1, end], end] .= [1.0, -4.0, 3.0]

for i ∈ 2:(size(grad, 2) - 1)
    grad[[i - 1, i + 1], i] .= [-1.0, 1.0]
end

grad = grad ./ (2 * Δ)

du_new = grad' * u

fig = Figure(resolution = (1920, 1080), fontsize = 40)
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0, 1.0)
lines!(ax, dim.x, u, linewidth = 5, label = "Function")
lines!(ax, dim.x, du_old, linewidth = 5, label = "Old Derivative")
lines!(ax, dim.x, du_new, linewidth = 5, label = "New Derivative")

axislegend(ax)

GLMakie.save("func.png", fig)

# # u = gaussian_pulse(dim, 0.0, 0.0, 1.0)[:, :, 1]
# # u0 = cat(u, zeros(size(u)..., 3), dims = 3)

# # C = WaveSpeed(dim, C0)
# # pml = build_pml(dim, pml_width) * pml_scale

# # ps = WaveParams(Δ, C, pml)

# # prob = ODEProblem(pml_2d!, u0, tspan, ps)
# # @time sol = solve(prob, RK4())
# # @time sol = interpolate(sol, dim, 0.05)
# # @time render!(sol, path = "new.mp4")