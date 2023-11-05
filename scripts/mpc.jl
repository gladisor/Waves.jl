using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
using ReinforcementLearning

println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int)
    return [policy(env) for i in 1:horizon]
end

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int, shots::Int)
    return hcat([build_action_sequence(policy, env, horizon) for i in 1:shots]...)
end

struct RandomShooting <: AbstractPolicy
    policy::AbstractPolicy
    model::AcousticEnergyModel
    horizon::Int
    shots::Int
    beta::Float32
end

function compute_action_penalty(a::Matrix{<: AbstractDesign})
    x = cat([hcat(vec.(a)[:, i]...) for i in axes(a, 2)]..., dims = 3)
    return vec(sum(sqrt.(sum(x .^ 2, dims = 1)), dims = 2))
end

env = BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/env.bson")[:env]

pml_checkpoint = 1760
pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal/checkpoint_step=$pml_checkpoint/checkpoint.bson")[:model])
# no_pml_checkpoint = 1120
# no_pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_no_pml/checkpoint_step=760/checkpoint.bson")[:model])

policy = RandomDesignPolicy(action_space(env))

reset!(env)

horizon = 3
shots = 1
beta = 1.0

s = gpu(fill(state(env), shots))
a = gpu(build_action_sequence(policy, env, horizon, shots))
t = build_tspan(time(env), env.dt, env.integration_steps * horizon)
t = gpu(hcat(fill(t, shots)...))

# @time y_hat = cpu(pml_model(s, a, t))

@time z = generate_latent_solution(pml_model, s, a, t);
@time z0, theta = embed(pml_model, s, a, t)
@time gs = Waves.adjoint_sensitivity(pml_model.iter, z, t, theta, z)

# @time begin
#     loss, back = Flux.pullback(a) do _a
#         Flux.mean(pml_model(s, _a, t))
#     end
#     gs = back(one(loss))[1]
# end
# ;

# @time begin
#     loss, back = Flux.pullback(a) do _a
#         Flux.mean(pml_model(s, _a, t))
#     end
#     gs = back(one(loss))[1]
# end
# ;

# @time begin
#     loss, back = Flux.pullback(a) do _a
#         Flux.mean(pml_model(s, _a, t))
#     end
#     gs = back(one(loss))[1]
# end
# ;

# @time begin
#     loss, back = Flux.pullback(a) do _a
#         Flux.mean(pml_model(s, _a, t))
#     end
#     gs = back(one(loss))[1]
# end
# ;

# @time begin
#     loss, back = Flux.pullback(a) do _a
#         Flux.mean(pml_model(s, _a, t))
#     end
#     gs = back(one(loss))[1]
# end
# ;

# # fig = Figure()
# # ax = Axis(fig[1, 1])
# # lines!(ax, y_hat[:, 1, 1])
# # lines!(ax, y_hat[:, 2, 1])
# # lines!(ax, y_hat[:, 3, 1])
# # save("energy_pml_$pml_checkpoint.png", fig)