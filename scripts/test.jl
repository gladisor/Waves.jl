include("dependencies.jl")

function stack(config1::Scatterers, config2::Scatterers)
    pos = vcat(config1.pos, config2.pos)
    r = vcat(config1.r, config2.r)
    c = vcat(config1.c, config2.c)
    return Scatterers(pos, r, c)
end

function ring_points(ring_radius::Float32, spacing::Float32, n::Int)::AbstractMatrix{Float32}
    R = ring_radius
    θ = 2 * asin((2 * Waves.MAX_RADII + spacing) / (2 * R))

    pos = [[R * cos(pi), R * sin(pi)]]
    for i in 1:n
        push!(pos, [R * cos(pi - i * θ), R * sin(pi - i * θ)])
        push!(pos, [R * cos(pi + i * θ), R * sin(pi + i * θ)])
    end
    push!(pos, [R * cos(0.0f0), R * sin(0.0f0)])

    return hcat(pos...)'
end

struct RandomRadiiScattererRing <: AbstractInitialDesign
    ring_radius::Float32
    spacing::Float32
    n::Int
    c::Float32
    center::Vector{Float32}
end

function (reset_design::RandomRadiiScattererRing)()
    pos = ring_points(reset_design.ring_radius, reset_design.spacing, reset_design.n)
    r = rand(Float32, size(pos, 1)) * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    c = fill(reset_design.c, size(pos, 1))
    return Scatterers(pos, r, c)
end

function design_space(reset_design::RandomRadiiScattererRing, scale::Float32)
    return radii_design_space(reset_design(), scale)
end

struct ScattererCloak <: AbstractInitialDesign
    ring::RandomRadiiScattererRing
    core::Scatterers
end

function (cloak::ScattererCloak)()
    return stack(cloak.ring(), cloak.core)
end

function design_space(cloak::ScattererCloak, scale::Float32)
    ds = design_space(cloak.ring, scale)

    left = stack(ds.left, zero(cloak.core))
    right = stack(ds.right, zero(cloak.core))

    return left..right
end

dim = TwoDim(15.0f0, 512)
ring = RandomRadiiScattererRing(5.0f0, 1.0f0, 4, BRASS, [0.0f0, 0.0f0])
core = Scatterers([0.0f0 0.0f0], [1.0f0], [BRASS])
cloak = ScattererCloak(ring, core)
pulse = build_pulse(build_grid(dim), -12.0f0, 0.0f0, 10.0f0)

env = WaveEnv(
    dim, 
    reset_wave = Silence(),
    reset_design = cloak,
    action_space = design_space(cloak, 1.0f0),
    source = Source(pulse, freq = 300.0f0),
    actions = 10) |> gpu

policy = RandomDesignPolicy(action_space(env))
@time render!(policy, env, path = "vid.mp4", seconds = env.actions * 1.0f0)