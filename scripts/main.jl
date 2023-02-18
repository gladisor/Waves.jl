using Flux
using Waves

dx = 0.1f0
ambient_speed = 1.0f0
dt = sqrt(dx^2/ambient_speed^2)
tmax = 20.0
n = Int(round(tmax / dt))

kwargs = Dict(:dim => TwoDim(15.0f0, dx), :pml_width => 4.0f0, :pml_scale => 20.0f0, :ambient_speed => ambient_speed, :dt => dt)
dyn = WaveDynamics(design = Cylinder(-3.0f0, -3.0f0, 1.0f0, 0.1f0); kwargs...)
dyn_inc = WaveDynamics(;kwargs...)

u = pulse(dyn.dim)
sol_inc = cpu(integrate(u, dyn_inc, n))

env = WaveEnv(u, dyn, 5) |> gpu
policy = pos_action_space(env.dyn.C.design.initial, 1.0f0)

sol_tot = WaveSol{TwoDim}[]
design_traj = DesignTrajectory{Cylinder}[]

action = zero(env.dyn.C.design(0.0f0))

while time(env) < 20.0
    sol = env(rand(policy))
    push!(sol_tot, sol)
    push!(design_traj, DesignTrajectory(env))
    println(time(env))
end

sol_tot = vcat(sol_tot...)
design_traj = vcat(design_traj...)

sol_sc = sol_tot - sol_inc
render!(sol_sc, design_traj, path = "vid.mp4")