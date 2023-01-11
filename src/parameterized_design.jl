export ParameterizedDesign

mutable struct ParameterizedDesign{D <: AbstractDesign}
    design::D
    initial::D
    final::D
    t_initial::Num
    t_final::Num
end

function ParameterizedDesign(design::AbstractDesign, initial::AbstractDesign, final::AbstractDesign)
    @parameters t_initial, t_final
    return ParameterizedDesign(design, initial, final, t_initial, t_final)
end

function ParameterizedDesign(design::AbstractDesign)
    @named initial = typeof(design)()
    @named final = typeof(design)()
    return ParameterizedDesign(design, initial, final)
end

function Waves.wave_equation(wave::Wave, design::ParameterizedDesign)::Equation
    return wave_equation(wave, wave_speed(wave, design))
end