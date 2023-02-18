using Waves

kwargs = Dict(:pml_width => 4.0, :pml_scale => 20.0, :ambient_speed => 1.0, :dt => 0.05)
dyn = WaveDynamics(dim = TwoDim(10.0, 0.05), design = Cylinder(-3, -3, 1.0, 0.2); kwargs...)
u = pulse(dyn.dim, 0.0, 0.0, 1.0)
env = WaveEnv(u, dyn, 20)
policy = pos_action_space(env.dyn.C.design.initial, 1.0)

wave_traj = WaveSol{TwoDim}[]
design_traj = DesignTrajectory{Cylinder}[]

action = Cylinder(0.2, 0.2, 0.0, 0.0)

while env.dyn.t < 500
    # sol = env(rand(policy))
    sol  = env(action)
    push!(wave_traj, sol)
    push!(design_traj, DesignTrajectory(env))
    println(time(env))
end

wave_traj = vcat(wave_traj...)
design_traj = vcat(design_traj...)

render!(wave_traj, design_traj, path = "vid.mp4")