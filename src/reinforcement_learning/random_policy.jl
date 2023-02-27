export RandomDesignPolicy

mutable struct RandomDesignPolicy <: AbstractPolicy
    action::ClosedInterval{<: AbstractDesign}
end

function (policy::RandomDesignPolicy)(env::WaveEnv)
    return gpu(rand(policy.action))
end