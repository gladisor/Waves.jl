include("dependencies.jl")

function Base.getindex(config::Scatterers, inds)
    return Scatterers(config.pos[[inds...], :], config.r[[inds...]], config.c[[inds...]])
end

struct RandomRadiiScattererRing <: AbstractInitialDesign
    core::Scatterers
    ring_radius::Float32
    spacing::Float32
    n::Int
    c::Float32
    center::Vector{Float32}
end

function (reset_design::RandomRadiiScattererRing)()
    R = reset_design.ring_radius
    S = reset_design.spacing
    θ = 2 * asin((2 * Waves.MAX_RADII + S) / (2 * R))
    N = reset_design.n

    pos = [[R * cos(pi), R * sin(pi)]]
    for i in 1:N
        push!(pos, [R * cos(pi - i * θ), R * sin(pi - i * θ)])
        push!(pos, [R * cos(pi + i * θ), R * sin(pi + i * θ)])
    end
    push!(pos, [R * cos(0.0f0), R * sin(0.0f0)])

    pos = hcat(pos...)'

    M = 2 + 2 * N
    r = rand(Float32, M) * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    c = fill(reset_design.c, M)

    pos = vcat(pos, reset_design.core.pos)
    r = vcat(r, reset_design.core.r)
    c = vcat(c, reset_design.core.c)

    return Scatterers(pos, r, c)
end

function design_space(reset_design::RandomRadiiScattererRing, scale::Float32)
    
    ds = radii_design_space(reset_design(), scale)
end

dim = TwoDim(15.0f0, 512)

core = Scatterers([0.0 0.0], [1.0], [BRASS])
reset_design = RandomRadiiScattererRing(core, 5.0f0, 1.0f0, 4, BRASS, [0.0f0, 0.0f0])
ds = design_space(reset_design, 1.0f0)

config = reset_design()

# pulse = build_pulse(build_grid(dim), -12.0f0, 0.0f0, 10.0f0)

# ds = radii_design_space(reset_design(), 1.0f0)

# env = WaveEnv(
#     dim, 
#     reset_wave = Silence(),
#     reset_design = reset_design,
#     action_space = ds,
#     source = Source(pulse, freq = 300.0f0),
#     actions = 2) |> gpu

# policy = RandomDesignPolicy(action_space(env))
# @time render!(policy, env, path = "vid.mp4", seconds = env.actions * 1.0f0)