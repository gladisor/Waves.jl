using ReinforcementLearning
using JLD2
using CairoMakie

using Waves

design_kwargs = Dict(:width => 1, :hight => 1, :spacing => 1.0f0, :c => 0.20f0, :center => [0.0f0, 0.0f0])
config = random_radii_scatterer_formation(;design_kwargs...)

grid_size = 4.0f0
elements = 512
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)

env = gpu(WaveEnv(
    initial_condition = PlaneWave(dim, -2.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    random_design = () -> random_radii_scatterer_formation(;design_kwargs...),
    space = radii_design_space(config, 0.05f0),
    design_steps = 20,
    tmax = 5.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))
# traj = episode_trajectory(env)
# agent = Agent(policy, traj)
# @time run(agent, env, StopWhenDone())

# sol_tot = WaveSol([s.sol.total for s in traj.traces.state]...)
# sol_inc = WaveSol([s.sol.incident for s in traj.traces.state]...)
# sol_sc = sol_tot - sol_inc

# sc_energy = sol_sc.u .|> displacement .|> energy .|> sum

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, sc_energy, label = "Scattered")
# save("energy_elements=$(elements)_speed=$(env.total_dynamics.ambient_speed)_design_steps=$(env.design_steps).png", fig)
# @time render!(traj, path = "vid_elements=$(elements)_speed=$(env.total_dynamics.ambient_speed)_design_steps=$(env.design_steps).mp4")

for i in 1:10
    @time data = generate_episode_data(policy, env, 1)
    data_path = mkpath("data/elements=$(elements)_speed=$(env.total_dynamics.ambient_speed)_design_steps=$(env.design_steps)/episode$i")
    save_episode_data!(data..., path = data_path)
end