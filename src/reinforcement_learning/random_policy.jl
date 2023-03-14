export RandomDesignPolicy

mutable struct RandomDesignPolicy <: AbstractPolicy
    action::ClosedInterval{<: AbstractDesign}
end

function (policy::RandomDesignPolicy)(::WaveEnv)
    return rand(policy.action)
end