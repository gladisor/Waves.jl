using CairoMakie
using SparseArrays
using Flux
using Flux.Optimise: Optimiser

using Waves

mutable struct WassersteinDynamics <: AbstractDynamics
    dim::AbstractDim
    grad::SparseMatrixCSC
    v::AbstractArray
    t::Int
    dt::Float32
end

Flux.@functor WassersteinDynamics (v,)

function Waves.reset!(dynamics::WassersteinDynamics)
    dynamics.t = 0
end

function divergence(grad::SparseMatrixCSC{Float32}, u::Array{Float32, 3})
    dx = grad * u[:, :, 1]
    dy = (grad * u[:, :, 2]')'
    return dx .+ dy
end

function wasserstein_dynamics(u::Matrix, t::Float32, dynamics::WassersteinDynamics)
    dU = -divergence(dynamics.grad, u .* dynamics.v)
    return dU
end

function euler(f::Function, wave::AbstractArray{Float32}, dynamics::AbstractDynamics)
    return f(wave, dynamics.t * dynamics.dt, dynamics) * dynamics.dt
end

function Waves.plot_wave!(p::WavePlot, dim::TwoDim, u::Matrix{Float32})
    heatmap!(p.ax, dim.x, dim.y, u, colormap = :ice)
end

steps = 100
dt = 1.0f0 / steps

dim = TwoDim(5.0f0, 128)
grad = Waves.gradient(dim.x)

pulse = Pulse(dim, 0.0f0, 0.0f0, 1.0f0);
wave = pulse()

target_pulse1 = Pulse(dim, -2.0f0, -2.0f0, 5.0f0)
target_pulse2 = Pulse(dim, 2.0f0, 2.0f0, 5.0f0)
target_wave = (target_pulse1() .+ target_pulse2()) ./ 2.0f0

velocity = zeros(Float32, size(dim)..., 2)
cell = WaveCell(wasserstein_dynamics, runge_kutta)
dynamics = WassersteinDynamics(dim, grad, velocity, 0, dt)

opt = Adam(0.1)
ps = Flux.params(dynamics)

for i in 1:200

    Waves.reset!(dynamics)

    gs = Flux.gradient(ps) do

        traj = integrate(cell, wave, dynamics, steps) ## run simulator
        loss = Flux.Losses.mse(traj[end], target_wave) # + W

        Flux.ignore() do 
            println("Update: $i, Loss: $loss")
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)

end

Waves.reset!(dynamics)
traj = integrate(cell, wave, dynamics, steps)
fig = Figure(resolution = (1920, 1080), fontsize = 30)
ax1 = Axis(fig[1, 1], aspect = 1.0f0, title = "αₜ₌₀")
ax2 = Axis(fig[1, 2], aspect = 1.0f0, title = "αₜ₌₁")
ax3 = Axis(fig[1, 3], aspect = 1.0f0, title = "Simulated αₜ₌₁")

heatmap!(ax1, dim.x, dim.y, wave, colormap = :ice)
heatmap!(ax2, dim.x, dim.y, target_wave, colormap = :ice)
heatmap!(ax3, dim.x, dim.y, traj[end], colormap = :ice)
save("u.png", fig)

Waves.reset!(dynamics)
traj = integrate(cell, wave, dynamics, steps)
fig = Figure(resolution = (1920, 1080), fontsize = 30)
ax1 = Axis(fig[1, 1], aspect = 1.0f0, title = "Velocity Field (x component)")
ax2 = Axis(fig[1, 2], aspect = 1.0f0, title = "Velocity Field (y component)")
heatmap!(ax1, dim.x, dim.y, dynamics.v[:, :, 1], colormap = :ice)
heatmap!(ax2, dim.x, dim.y, dynamics.v[:, :, 2], colormap = :ice)
save("v.png", fig)

Waves.reset!(dynamics)
sol = solve(cell, wave, dynamics, steps)
render!(sol, path = "vid.mp4")