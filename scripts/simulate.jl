using Waves, CairoMakie
using Flux

grid_size = 15.0f0
elements = 1024
dim = OneDim(grid_size, elements)
grad = build_gradient(dim)

pml = build_pml(dim, 2.0f0, 10000.0f0)
w0 = zeros(Float32, size(dim)..., 4, 1)
pulse = build_normal(build_grid(dim), [-12.0f0], [0.4f0], [1.0f0])
dynamics = AcousticDynamics(dim, WATER, 2.0f0, 10000.0f0)

c = ones(Float32, size(dim)...)
# c[dim.x .>= -7.0f0] .= 0.0f0
C = t -> c


F = Source(pulse, 1000.0f0)
θ = [C, F]

dt = 1f-5
iter = Integrator(runge_kutta, dynamics, dt)
tspan = build_tspan(iter, 0.0f0, 500)[:, :]

loss, back = Flux.pullback(w0) do _w0
    w = iter(_w0, tspan, θ)
    return sum(w[:, 1, 1, end] .^ 2)
end
# @time w = iter(w0, tspan, θ)
gs = back(one(loss))





# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)

# @time record(fig, "vid.mp4", axes(tspan, 1)) do i
#     empty!(ax)
#     lines!(ax, dim.x, w[:, 1, 1, i], color = :blue)
#     lines!(ax, dim.x, c, color = :red)
# end