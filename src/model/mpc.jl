function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int)
    return [policy(env) for i in 1:horizon]
end

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int, shots::Int)
    return hcat([build_action_sequence(policy, env, horizon) for i in 1:shots]...)
end

struct RandomShooting <: AbstractPolicy
    policy::AbstractPolicy
    model
    horizon::Int
    shots::Int
    alpha::Float32
end

function compute_action_cost(a::Matrix{<: AbstractDesign})
    x = cat([hcat(vec.(a)[:, i]...) for i in axes(a, 2)]..., dims = 3)
    return vec(sum(sqrt.(sum(x .^ 2, dims = 1)), dims = 2))
end