export ParameterizedDesign, design_parameters

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

function ParameterizedDesign(design::AbstractDesign; kwargs...)
    @named initial = typeof(design)(;kwargs...)
    @named final = typeof(design)(;kwargs...)
    return ParameterizedDesign(design, initial, final)
end

function Waves.wave_equation(wave::Wave, design::ParameterizedDesign)::Equation
    return wave_equation(wave, wave_speed(wave, design))
end

function Base.:+(pd::ParameterizedDesign, action::AbstractDesign)
    return pd.design + action
end

function design_parameters(::AbstractDesign)::Vector end

function design_parameters(design::ParameterizedDesign, new_design::AbstractDesign, t0, tf)
    [
        (design_parameters(design.initial) .=> design_parameters(design.design))...,
        (design_parameters(design.final) .=> design_parameters(new_design))...,
        design.t_initial => t0,
        design.t_final => tf
    ]
end

function design_parameters(design::ParameterizedDesign)
    return design_parameters(design.design)
end

function get_t_norm(design::ParameterizedDesign, t::Num)
    return (t - design.t_initial) / (design.t_final - design.t_initial)
end

function interpolate(initial::Num, final::Num, t::Num)
    return initial + (final - initial) * t
end