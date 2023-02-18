using Flux

using Waves

function Flux.gpu(dim::TwoDim)
    return TwoDim(gpu(dim.x), gpu(dim.y))
end

function Flux.gpu(C::SpeedField)
    dim = gpu(C.dim)
    g = gpu(C.g)
    ambient_speed = gpu(C.ambient_speed)
    return SpeedField(dim, g, ambient_speed, C.design)
end

function Flux.gpu(dyn::WaveDynamics)
    dim = gpu(dyn.dim)
    grad = gpu(dyn.grad)
    C = gpu(dyn.C)
    pml = gpu(dyn.pml)
    return WaveDynamics(dim, grad, C, pml, dyn.t, dyn.dt)
end

function Flux.gpu(env::WaveEnv)
    u = gpu(env.u)
    dyn = gpu(env.dyn)
    return WaveEnv(u, dyn, env.design_steps)
end

kwargs = Dict(:pml_width => 4.0, :pml_scale => 20.0, :ambient_speed => 1.0, :dt => 0.05)
dyn = WaveDynamics(dim = TwoDim(10.0, 0.05), design = Cylinder(-3, -3, 1.0, 0.1); kwargs...)
u = pulse(dyn.dim, 0.0, 0.0, 1.0)
env = WaveEnv(u, dyn, 10) |> gpu
policy = pos_action_space(env.dyn.C.design.initial, 1.0)

wave_traj = WaveSol{TwoDim}[]
design_traj = DesignTrajectory{Cylinder}[]

action = Cylinder(0.0, 0.0, 0.0, 0.0)

while env.dyn.t < 500
    sol = env(rand(policy))
    # sol = env(action)
    push!(wave_traj, sol)
    push!(design_traj, DesignTrajectory(env))
    println(time(env))
end

wave_traj = vcat(wave_traj...)
design_traj = vcat(design_traj...)

render!(wave_traj, design_traj, path = "vid.mp4")