struct ActionSequence{D <: AbstractDesign}
    traj::Vector{D}
end

function ActionSequence(env::WaveEnv)
    design = env.dyn.C.design
    t = collect(range(design.ti, design.tf, env.design_steps + 1))
    return ActionSequence(design.(t))
end