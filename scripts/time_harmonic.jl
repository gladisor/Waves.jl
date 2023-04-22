include("dependencies.jl")

struct ForceDesignEncoder
    input_layer::Dense
    hidden_layers::Chain
end

Flux.@functor ForceDesignEncoder

function (model::ForceDesignEncoder)(design::AbstractDesign, action::AbstractDesign)
    x = vcat(vec(design), vec(action))
    return x |> model.input_layer |> model.hidden_layers
end

actions = 10
tf = ti + steps * actions * dt

design = DesignInterpolator(
    Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0]),
    Scatterers([0.0f0 0.0f0], [1.0f0], [0.0f0]), ti, tf)

dim = TwoDim(grid_size, elements)
grid = build_grid(dim)
grad = build_gradient(dim)
pml = build_pml(dim, 2.0f0, 20000.0f0)
source = build_pulse(grid, intensity = 10.0f0, x = -5.0f0, y = -5.0f0)
freq = 0.005f0
bc = dirichlet(dim)

dynamics = TimeHarmonicWaveDynamics(design, ambient_speed, grid, grad, pml, source, freq, bc)
iter = Integrator(runge_kutta, dynamics, ti, dt, steps * actions)

wave = build_wave(dim, fields = 6)
@time sol = iter(wave)
tspan = build_tspan(iter)

u = linear_interpolation(tspan, unbatch(sol))
@time render!(dim, tspan, u, path = "vid.mp4", seconds = 5.0f0)