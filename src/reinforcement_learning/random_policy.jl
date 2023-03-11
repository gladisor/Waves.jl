export RandomDesignPolicy

mutable struct RandomDesignPolicy <: AbstractPolicy
    action::ClosedInterval{<: AbstractDesign}
end

function (policy::RandomDesignPolicy)(::WaveEnv)
    return gpu(rand(policy.action))
end