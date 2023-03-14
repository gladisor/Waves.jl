export episode_trajectory

function episode_trajectory(env::WaveEnv)
    traj = CircularArraySARTTrajectory(
        capacity = num_steps(env),
        state = Vector{WaveEnvState} => (),
        action = Vector{typeof(initial_design(env.total_dynamics.design))} => ())

    return traj
end
