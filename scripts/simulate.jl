using Waves, CairoMakie
using Flux
include("model_modifications.jl")

grid_size = 15.0f0
elements = 1024
dim = OneDim(grid_size, elements)
grad = build_gradient(dim)

pml = build_pml(dim, 2.0f0, 10000.0f0)

pulse = build_normal(build_grid(dim), [-12.0f0], [0.4f0], [1.0f0])
dynamics = AcousticDynamics(dim, WATER, 2.0f0, 10000.0f0)
c = gpu(ones(Float32, size(dim)...))
# c[dim.x .>= -7.0f0] .= 0.0f0
C = t -> c
F = Source(pulse, 1000.0f0)
pml = build_pml(dim, 2.0f0, 1.0f0)
θ = gpu([C, F, pml[:, :]])

dt = 1f-5
iter = gpu(Integrator(runge_kutta, dynamics, dt))
tspan = gpu(build_tspan(iter, 0.0f0, 500)[:, :])

w0 = gpu(zeros(Float32, size(dim)..., 4, 1))
# @time w = iter(w0, tspan, θ)
loss, back = Flux.pullback(w0, θ) do _w0, _θ
    w = iter(_w0, tspan, _θ)
    return sum(w[:, 1, 1, end] .^ 2)
end

@time gs = back(one(loss))





# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)

# @time record(fig, "vid.mp4", axes(tspan, 1)) do i
#     empty!(ax)
#     lines!(ax, dim.x, w[:, 1, 1, i], color = :blue)
#     lines!(ax, dim.x, c, color = :red)
# end