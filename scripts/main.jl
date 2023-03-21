using Flux
using ReinforcementLearning

using Waves

include("design_encoder.jl")

struct WaveNet
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    wave_decoder::WaveDecoder
    cell::AbstractWaveCell
    z_dynamics::WaveDynamics
end

Flux.@functor WaveNet (wave_encoder, design_encoder, wave_decoder, cell)

function (net::WaveNet)(s::WaveEnvState, a::AbstractDesign)
    z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
    return z
end

design_kwargs = Dict(
    :width => 1, 
    :hight => 1, 
    :spacing => 1.0f0, 
    :r => 0.5f0, 
    :c => 0.20f0, 
    :center => [0.0f0, 0.0f0])

function random_radii_scatterer_formation(;kwargs...)
    config = scatterer_formation(;kwargs...)
    r = rand(Float32, size(config.r))
    r = r * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    return Scatterers(config.pos, r, config.c)
end

config = random_radii_scatterer_formation(;design_kwargs...)
grid_size = 4.0f0
elements = 700
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

@time data = generate_episode_data(policy, env, 1)
;
@time render!(policy, env, path = "data/vid.mp4")

s, a = data;
x = s[end].sol.total.u[50][:, :, 1]

using CairoMakie
fig = Figure(resolution = (1920, 1080))
ax = Axis(
    fig[1, 1],
    backgroundcolor = :transparent,
    leftspinevisible = false,
    rightspinevisible = false,
    bottomspinevisible = false,
    topspinevisible = false,
    xticklabelsvisible = false, 
    yticklabelsvisible = false,
    xgridcolor = :transparent,
    ygridcolor = :transparent,
    xminorticksvisible = false,
    yminorticksvisible = false,
    xticksvisible = false,
    yticksvisible = false,
    xautolimitmargin = (0.0,0.0),
    yautolimitmargin = (0.0,0.0),
    aspect = 1.0)

heatmap!(ax, dim.x, dim.y, x, colormap = :ice, grid = false, ticks = false)
save("u.png", fig)

# p = WavePlot(dim)
# heatmap!(p.ax, dim.x, dim.y, cpu(x), colormap = :ice)
# save("u.png", p.fig)