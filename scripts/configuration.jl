using Waves: AbstractDesign, Scatterer

struct Configuration <: AbstractDesign
    scatterers::Vector{Scatterer}
end

function Base.:+(config1::Configuration, config2::Configuration)
    return Configuration(config1.scatterers .+ config2.scatterers)
end