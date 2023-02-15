struct Configuration <: AbstractDesign
    scatterers::Vector{<:AbstractDesign}
end

function Configuration(dim::TwoDim; M::Int, r::Float64, c::Float64, offset::Float64)
    scatterers = Cylinder[]

    for _ ∈ 1:M
        push!(scatterers, Cylinder(dim, r = r, c = c, offset = offset))
    end

    return Configuration(scatterers)
end

function Base.:+(c1::Configuration, c2::Configuration)
    return Configuration([s1 + s2 for (s1, s2) ∈ zip(c1.scatterers, c2.scatterers)])
end

function Base.:-(c1::Configuration, c2::Configuration)
    return Configuration([s1 - s2 for (s1, s2) ∈ zip(c1.scatterers, c2.scatterers)])
end

function Base.:*(config::Configuration, n::Float64)
    return Configuration([s * n for s ∈ config.scatterers])
end

function Base.:*(n::Float64, config::Configuration)
    return config * n
end

function Base.:/(config::Configuration, n::Float64)
    return Configuration([s / n for s ∈ config.scatterers])
end

function Waves.speed(dim::TwoDim, config::Configuration, C0::Matrix{Float64})
    C = ones(size(dim)) .* C0

    for scatterer ∈ config.scatterers
        C .= speed(dim, scatterer, C)
    end

    return C
end

function Waves.plot!(fig::GLMakie.Figure, config::Configuration)
    for scatterer ∈ config.scatterers
        plot!(fig, scatterer)
    end
end
