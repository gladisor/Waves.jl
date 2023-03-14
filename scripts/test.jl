using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using IntervalSets

using Waves

include("wave_encoder.jl")
include("wave_decoder.jl")

function radii_design_space(config::Scatterers, scale::Float32)

    pos = zeros(Float32, size(config.pos))

    radii_low = - scale * ones(Float32, size(config.r))
    radii_high =  scale * ones(Float32, size(config.r))
    c = zeros(Float32, size(config.c))

    return Scatterers(pos, radii_low, c)..Scatterers(pos, radii_high, c)
end

function square_formation()
    points = [
         0.0f0   0.0f0;
        -1.0f0   0.0f0;
        -1.0f0   1.0f0;
         0.0f0   1.0f0;
         1.0f0   1.0f0;
         1.0f0   0.0f0;
         1.0f0  -1.0f0;
         0.0f0  -1.0f0;
        -1.0f0  -1.0f0;  
        ]

    return points * 2
end

struct DesignEncoder
    dense1::Dense
    dense2::Dense
    dense3::Dense
end

Flux.@functor DesignEncoder

function DesignEncoder(in_size::Int, h_size::Int, out_size::Int, activation::Function)
    dense1 = Dense(in_size, h_size, activation)
    dense2 = Dense(h_size, h_size, activation)
    dense3 = Dense(h_size, out_size, sigmoid)
    return DesignEncoder(dense1, dense2, dense3)
end

function (encoder::DesignEncoder)(design::AbstractDesign, action::AbstractDesign)
    x = vcat(vec(design), vec(action))
    return x |> encoder.dense1 |> encoder.dense2 |> encoder.dense3
end

grid_size = 5.0f0
elements = 128
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

pos = square_formation()

r = ones(Float32, size(pos, 1)) * 0.5f0
c = ones(Float32, size(pos, 1)) * 0.5f0

translation = [-1.0f0, 0.0f0]
config = Scatterers(pos .- translation', r, c)

env = gpu(WaveEnv(
    initial_condition = Pulse(dim, -4.0f0, 0.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    space = radii_design_space(config, 0.2f0),
    design_steps = 100,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))
action = policy(env)
env(action)
s = state(env)

# traj = episode_trajectory(env)
# agent = Agent(policy, traj)
# @time run(agent, env, StopWhenDone())
# @time render!(traj, path = "vid.mp4")

z_size = Int.(size(dim) ./ (2 ^ 3))
h_fields = 64 ## wave_encoder
h_size = 128 ## design_encoder
z_fields = 2
activation = relu

design_size = 2 * length(vec(s.design))

cell = WaveCell(nonlinear_latent_wave, runge_kutta)
z_dim = OneDim(grid_size, prod(z_size))
z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...) |> gpu

wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation) |> gpu
wave_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation) |> gpu
design_encoder = DesignEncoder(design_size, h_size, prod(z_size), activation) |> gpu

opt = Adam(0.0005)
ps = Flux.params(wave_encoder, wave_decoder, design_encoder)

u = cat(s.sol.total.u[2:end]..., dims = 4) |> gpu

for _ ∈ 1:20
    Waves.reset!(z_dynamics)

    gs = Flux.gradient(ps) do

        z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, action))
        latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
        u_pred = wave_decoder(reshape(latents, z_size..., z_fields + 1, env.design_steps))

        loss = sqrt(Flux.Losses.mse(u, u_pred))

        Flux.ignore() do
            println(loss)
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end

z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, action))
Waves.reset!(z_dynamics)
z_sol = solve(cell, z, z_dynamics, env.design_steps)
render!(z_sol, path = "z_sol.mp4")