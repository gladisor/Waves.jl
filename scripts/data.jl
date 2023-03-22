using ReinforcementLearning
using Flux
using JLD2

using Waves

design_kwargs = Dict(:width => 1, :hight => 1, :spacing => 1.0f0, :c => 0.20f0, :center => [0.0f0, 0.0f0])
config = random_radii_scatterer_formation(;design_kwargs...)

grid_size = 4.0f0
elements = 256
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

env = gpu(WaveEnv(
    initial_condition = PlaneWave(dim, -2.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    random_design = () -> random_radii_scatterer_formation(;design_kwargs...),
    space = radii_design_space(config, 0.2f0),
    design_steps = 100,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))
@time train_data = generate_episode_data(policy, env, 2)
@time test_data = generate_episode_data(policy, env, 2)

train_s, train_a = train_data;
train_path = mkpath("data/train")

for (i, (s, a)) in enumerate(zip(train_s, train_a))
    println(i)
    jldsave(joinpath(train_path, "data$i.jld2"); s, a)
end

test_s, test_a = test_data;
test_path = mkpath("data/test")

for (i, (s, a)) in enumerate(zip(test_s, test_a))
    println(i)
    jldsave(joinpath(test_path, "data$i.jld2"); s, a)
end




