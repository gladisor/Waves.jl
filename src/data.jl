export episode_trajectory

function episode_trajectory(env::WaveEnv)
    traj = CircularArraySARTTrajectory(
        capacity = num_steps(env),
        state = Vector{TotalWaveSol} => (),
        action = Vector{typeof(env.total_dynamics.design(0.0f0))} => ())

    return traj
end