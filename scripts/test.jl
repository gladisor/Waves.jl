using Waves
using Flux
using CairoMakie
using Interpolations: linear_interpolation

function (dyn::AcousticDynamics{OneDim})(x::AbstractArray, t, θ)
    F = θ
    U = x[:, 1]
    V = x[:, 2]
    ∇ = dyn.grad

    dU = dyn.c0 .^ 2 * ∇ * V .- dyn.pml .* U
    dV = ∇ * (U .+ F(t)) .- dyn.pml .* V
    return hcat(dU .* dyn.bc, dV)
end

dim = OneDim(15.f0, 512)
grid = build_grid(dim)
pulse = build_normal(grid, 
    [0.0f0, 0.5f0], 
    [0.3f0, 0.4f0], 
    [1.0f0, -0.5f0])

source = Source(pulse, 1000.0f0)

c0 = WATER
pml_width = 5.f0
pml_scale = 10000.0f0
dyn = AcousticDynamics(dim, c0, pml_width, pml_scale)
iter = Integrator(runge_kutta, dyn, 1f-5, 200)
x = build_wave(dim, 2)
tspan = build_tspan(iter, 0.0f0)

cost, back = Flux.pullback(x, source) do _x, _source
    sol = iter(_x, tspan, _source)
    return Flux.mean(sum(sol[:, 1, :] .^ 2, dims = 1))
end
gs = back(one(cost) / 1000)

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[1, 3])
lines!(ax1, dim.x, gs[1][:, 1])
lines!(ax2, dim.x, gs[1][:, 2])
lines!(ax3, dim.x, gs[2].shape)
save("gs.png", fig)



@time sol = iter(x, tspan, source)
fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

@time record(fig, "latent.mp4", axes(sol, 3)) do i
    empty!(ax)
    lines!(ax, dim.x, sol[:, 1, i], color = :blue)
end