include("dependencies.jl")

struct DesignSpace{D <: AbstractDesign}
    low::D
    high::D
end