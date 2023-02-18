using Flux

using Waves

dx = 0.025f0
ambient_speed = 1.0f0
dt = sqrt(dx^2/ambient_speed^2)

kwargs = Dict(:pml_width => 4.0f0, :pml_scale => 20.0f0, :ambient_speed => ambient_speed, :dt => dt)
dyn = WaveDynamics(dim = TwoDim(10.0f0, dx), design = Cylinder(-3.0f0, -3.0f0, 1.0f0, 0.1f0); kwargs...)
u = pulse(dyn.dim)
env = WaveEnv(u, dyn, 20) |> gpu
policy = pos_action_space(env.dyn.C.design.initial, 1.0f0)

wave_traj = WaveSol{TwoDim}[]
design_traj = DesignTrajectory{Cylinder}[]

action = zero(env.dyn.C.design(0.0f0))

while time(env) < 10.0
    sol = env(rand(policy))
    push!(wave_traj, sol)
    push!(design_traj, DesignTrajectory(env))
    println(time(env))
end

wave_traj = cpu(vcat(wave_traj...))
design_traj = vcat(design_traj...)

render!(wave_traj, design_traj, path = "vid.mp4")