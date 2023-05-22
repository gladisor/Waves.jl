include("../scripts/dependencies.jl")

using Test

@testset begin

    dim = TwoDim(15.0f0, 512)
    grid = build_grid(dim)

    pos = hcat([
        [0.0f0, 0.0f0],
        [0.0f0, 5.0f0],
        [0.0f0, -5.0f0]
    ]...)'

    r_low = fill(0.2f0, size(pos, 1))
    r_high = fill(1.0f0, size(pos, 1))
    c = fill(AIR, size(pos, 1))

    core = Cylinders([4.0f0, 0.0f0]', [2.0f0], [AIR])
    design_low = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_low, c)), core)
    design_high = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_high, c)), core)
    space = DesignSpace(design_low, design_high)

    
end