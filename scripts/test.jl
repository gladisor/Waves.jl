include("dependencies.jl")

# function hexagon(::Type{Scatterers}, r::Float32, c::Float32)
#     @assert r >= 2.0 * Waves.MAX_RADII
#     pos = Vector{Vector{Float32}}()
#     for i in 1:6
#         push!(pos, [r * cos((i-1) * 2pi/6.0f0), r * sin((i-1) * 2pi/6.0f0)])
#     end

#     pos = hcat(pos...)'
#     M = size(pos, 1)
#     r = rand(Float32, M) * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
#     # r = ones(Float32, M)
#     return Scatterers(pos, r, fill(c, M))
# end

function hexagon(r::Float32)
    @assert r >= 2.0 * Waves.MAX_RADII

    pos = Vector{Vector{Float32}}()
    for i in 1:6
        push!(pos, [r * cos((i-1) * 2pi/6.0f0), r * sin((i-1) * 2pi/6.0f0)])
    end

    return Matrix{Float32}(hcat(pos...)')
end

struct Radii <: AbstractInitialDesign
    pos::AbstractMatrix{Float32}
    c::Float32
end

function (radii::Radii)()
    M = size(radii.pos, 1)
    r = rand(Float32, M) * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    return Scatterers(radii.pos, r, fill(radii.c, M))
end

function Waves.design_space(radii::Radii, scale::Float32)
    return radii_design_space(radii(), scale)
end

radii = Radii(hexagon(3.0f0), BRASS)

cloak = RandomCloak(
    radii,
    Scatterers([0.0f0 0.0f0], [1.0f0], [BRASS]))

config = cloak()

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)
mesh!(ax, config)
save("config.png", fig)


