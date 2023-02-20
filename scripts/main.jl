using Flux
using Waves

function circle_mask(dim::TwoDim, radius::Float32)
    g = grid(dim)
    return dropdims(sum(g .^ 2, dims = 3), dims = 3) .< radius ^2
end

function flux(u::AbstractArray{Float32, 3}, grad::AbstractMatrix{Float32}, mask::AbstractMatrix)
    U = view(u, :, :, 1)
    dUxx = grad * grad * U
    dUy = (grad * U')'
    dUyy = (grad * dUy')'
    return sum((dUxx .+ dUyy) .* mask)
end

function flux(sol::WaveSol, grad::AbstractMatrix{Float32}, mask::AbstractMatrix)
    return [flux(sol[i], grad, mask) for i ∈ axes(sol.t, 1)]
end

dx = 0.1f0
ambient_speed = 1.0f0
dt = sqrt(dx^2/ambient_speed^2)
tmax = 20.0
n = Int(round(tmax / dt))

kwargs = Dict(:dim => TwoDim(15.0f0, dx), :pml_width => 4.0f0, :pml_scale => 20.0f0, :ambient_speed => ambient_speed, :dt => dt)
dyn = WaveDynamics(design = Cylinder([0.0f0, 0.0f0], 1.0f0, 0.1f0); kwargs...)
dyn_inc = WaveDynamics(;kwargs...)

u = pulse(dyn.dim, 0.0f0, 4.0f0, 1.0f0) .+ pulse(dyn.dim, 0.0f0, -4.0f0, 1.0f0)
@time sol_inc = cpu(integrate(gpu(u), gpu(dyn_inc), n))
env = gpu(WaveEnv(u, dyn, 5))
policy = action_space(env.dyn.C.design.initial, 1.0f0)

sol_tot = WaveSol{TwoDim}[]
design_traj = DesignTrajectory{Cylinder}[]

action = zero(env.dyn.C.design.initial)

@time while time(env) < tmax
    sol = env(gpu(rand(policy)))
    push!(sol_tot, cpu(sol))
    push!(design_traj, cpu(DesignTrajectory(env)))
    println(time(env))
end

sol_tot = vcat(sol_tot...)
design_traj = vcat(design_traj...)

sol_sc = sol_tot - sol_inc

mask = circle_mask(env.dyn.dim, 10.0f0)
sc_flux = flux(sol_sc, env.dyn.grad, mask)
inc_flux = flux(sol_inc, env.dyn.grad, mask)

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, sol_inc.t[1], sol_inc.t[end])
lines!(ax, sol_inc.t, inc_flux, color = :blue, label = "Incident")
lines!(ax, sol_sc.t, sc_flux, color = :red, label = "Scattered")
axislegend(ax)
save("flux.png", fig)

@time render!(sol_tot, design_traj, path = "vid.mp4")