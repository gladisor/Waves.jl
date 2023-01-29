export Design

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

function Waves.design_parameters(design::Design)
    return Waves.design_parameters(design.design)
end

function Waves.design_parameters(design::Design, new_design::AbstractDesign, ti, tf)
    return [
        (design_parameters(design.initial) .=> design_parameters(design))...,
        (design_parameters(design.final) .=> design_parameters(new_design))...,
        design.ti => ti,
        design.tf => tf
    ]
end