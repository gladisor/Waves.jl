using IntervalSets
using ReinforcementLearning
using Distributions: Uniform

using Waves

function Base.rand(cyl::ClosedInterval{Cylinder})
    x = rand(Uniform(cyl.left.x, cyl.right.x))
    y = rand(Uniform(cyl.left.y, cyl.right.y))

    if cyl.right.r > cyl.left.r
        r = rand(Uniform(cyl.left.r, cyl.right.r))
    else
        r = 0.0
    end

    if cyl.right.c > cyl.left.c
        c = rand(Uniform(cyl.left.c, cyl.right.c))
    else
        c = 0.0
    end

    return Cylinder(x, y, r, c)
end

kwargs = Dict(:pml_width => 4.0, :pml_scale => 20.0, :ambient_speed => 1.0, :dt => 0.05)
dyn = WaveDynamics(dim = TwoDim(10.0, 0.05), design = Cylinder(-3, -3, 2.0, 0.2); kwargs...)
u = pulse(dyn.dim, 0.0, 0.0, 1.0)
env = WaveEnv(u, dyn, 5)
action_space = Cylinder(-1.0, -1.0, 0.0, 0.0)..Cylinder(1.0, 1.0, 0.0, 0.0)

wave_traj = WaveSol{TwoDim}[]
design_traj = DesignTrajectory{Cylinder}[]

while env.dyn.t < env.design_steps * 100
    sol = env(rand(action_space))

    push!(wave_traj, sol)
    push!(design_traj, DesignTrajectory(env))
    println(time(env))
end

wave_traj = vcat(wave_traj...)
design_traj = vcat(design_traj...)

render!(wave_traj, design_traj, path = "vid.mp4")