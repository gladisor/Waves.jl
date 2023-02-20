using Waves

include("scatterers.jl")

gs = 15.0f0
dx = 0.1f0
ambient_speed = 1.0f0
dt = sqrt(dx^2/ambient_speed^2)
tmax = 20.0
n = Int(round(tmax / dt))
M = 4
pml_width = 4.0f0

config = Scatterers(M = 4, r = 1.0f0, disk_r = gs - pml_width, c = 0.1f0)

kwargs = Dict(:dim => TwoDim(gs, dx), :pml_width => pml_width, :pml_scale => 20.0f0, :ambient_speed => ambient_speed, :dt => dt)
dyn = WaveDynamics(design = config; kwargs...)

u = pulse(dyn.dim, -9.0f0, 9.0f0, 1.0f0)
env = gpu(WaveEnv(u, dyn, 5))
policy = action_space(config, 1.0f0)

sol_tot = WaveSol{TwoDim}[]
design_traj = DesignTrajectory{Scatterers}[]

@time while time(env) < tmax
    sol = env(gpu(rand(policy)))

    push!(sol_tot, cpu(sol)); push!(design_traj, cpu(DesignTrajectory(env)))
    println(time(env))
end

sol_tot = vcat(sol_tot...); design_traj = vcat(design_traj...)
@time render!(sol_tot, design_traj, path = "vid.mp4")
