export episode_trajectory, generate_episode_data, load_episode_data, save_episode_data!

function episode_trajectory(env::WaveEnv)
    traj = CircularArraySARTTrajectory(
        capacity = num_steps(env),
        state = Vector{WaveEnvState} => (),
        action = Vector{typeof(initial_design(env.total_dynamics.design))} => ())

    return traj
end

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv)
    traj = episode_trajectory(env)
    agent = Agent(policy, traj)
    run(agent, env, StopWhenDone())

    states = traj.traces.state[2:end]
    actions = traj.traces.action[1:end-1]

    return (states, actions)
end

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv, episodes::Int)
    states = []
    actions = []

    for _ ∈ 1:episodes
        s, a = generate_episode_data(policy, env)
        push!(states, s)
        push!(actions, a)
    end

    return (vcat(states...), vcat(actions...))
end

function load_episode_data(path::String)
    file = jldopen(path)
    s = file["s"]
    a = file["a"]
    return (s, a)
end

function save_episode_data!(states::Vector{WaveEnvState}, actions::Vector{<:AbstractDesign}; path)
    @showprogress for (i, (s, a)) in enumerate(zip(states, actions))
        jldsave(joinpath(path, "data$i.jld2"); s, a)
    end
end