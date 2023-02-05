export Design, design_parameters

mutable struct Design{D <: AbstractDesign}
    design::D
    initial::D
    final::D
    ti::Num
    tf::Num
end

function Design(design::AbstractDesign; kwargs...)
    @named initial = typeof(design)(;kwargs...)
    @named final = typeof(design)(;kwargs...)
    @parameters ti, tf
    return Design(design, initial, final, ti, tf)
end

function design_parameters(design::Design)
    return design_parameters(design.design)
end

function design_parameters(design::Design, new_design::AbstractDesign, ti, tf)
    return [
        (design_parameters(design.initial) .=> design_parameters(design))...,
        (design_parameters(design.final) .=> design_parameters(new_design))...,
        design.ti => ti,
        design.tf => tf
    ]
end

"""
Interpolates between initial and final. Time t is expected to be between 0.0 and 1.0
"""
function interpolate(initial::Num, final::Num, t::Num)
    return initial + (final - initial) * t
end

function reset!(design::Design, dim::AbstractDim; kwargs...)
    design.design = typeof(design.design)(dim; kwargs...)
end

# function Base.:+(design::Design, action::AbstractDesign)
#     return design.design + action
# end