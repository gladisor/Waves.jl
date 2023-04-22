include("dependencies.jl")

struct ForceLatentDynamics <: AbstractDynamics
    ambient_speed::Float32
    grad::AbstractMatrix{Float32}
    bc::AbstractVector{Float32}
end

Flux.@functor ForceLatentDynamics

function (dyn::ForceLatentDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]
    f = wave[:, 3]
    c = wave[:, 4]

    b = dyn.ambient_speed .^ 2 .* c

    du = b .* dyn.grad * v
    dv = dyn.grad * u .+ f

    df = f * 0.0f0
    dc = c * 0.0f0

    return hcat(dyn.bc .* du, dv, df, dc)
end

latent_elements = 512
dim = OneDim(grid_size, latent_elements)
grid = build_grid(dim)
grad = build_gradient(dim)
bc = dirichlet(dim)
force = randn(latent_elements) * 0.0f0

wave = build_wave(dim, fields = 4)
wave[:, 1] .= build_pulse(grid, x = -2.0f0) .+ build_pulse(grid, x = 2.0f0, intensity = 1.0f0)
wave[:, 2] .= 0.0f0
wave[:, 3] .= force
wave[:, 4] .= 1.0f0

dynamics = ForceLatentDynamics(ambient_speed, grad, bc)
iter = Integrator(runge_kutta, dynamics, ti, dt, 50)
@time z = iter(wave)
@time render!(dim, z, path = "latent.mp4")

mlp = Chain(Flux.flatten, Dense(4 * latent_elements, 1))
# cost, back = pullback(_wave -> mean(mlp(iter(_wave))), wave)
cost, back = pullback(_wave -> mean(sum(iter(_wave)[:, 1, :] .^ 2, dims = 1)), wave)

gs = back(one(cost))[1]

fig = Figure()
ax1 = Axis(fig[1, 1], title = "Displacement")
ax2 = Axis(fig[1, 2], title = "Velocity")
ax3 = Axis(fig[1, 3], title = "Force")
ax4 = Axis(fig[1, 4], title = "Speed")

lines!(ax1, dim.x, gs[:, 1])
lines!(ax2, dim.x, gs[:, 2])
lines!(ax3, dim.x, gs[:, 3])
lines!(ax4, dim.x, gs[:, 4])

save("gs.png", fig)