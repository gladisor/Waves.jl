using Waves
using CairoMakie: save
using Flux

## Set some parameters
grid_size = 5.0f0
elements = 300
steps = 200

## Create the dimensional space
dim = OneDim(grid_size, elements)

## Make initial condition
pulse = Pulse(dim, 0.0f0, 10.0f0) |> gpu

## Initialize wave
wave = build_wave(dim, fields = 3) |> gpu
wave = pulse(wave)

## Split wave integration scheme with runge_kutta integrator
activation = tanh

layers =  Chain(
    Dense(length(dim.x), length(dim.x), activation),
    b -> sum(b, dims = 2),
    sigmoid
    )

cell = WaveRNNCell(latent_wave, runge_kutta, layers) |> gpu
dynamics = WaveDynamics(dim = dim, pml_width = 1.0f0, pml_scale = 50.0f0, ambient_speed = 1.0f0, dt = 0.01f0) |> gpu

opt = Adam(0.001)

sol = solve(cell, wave, dynamics, steps) |> cpu
render!(sol, path = "vid.mp4")

ps = Flux.params(cell)

for i ∈ 1:100

    Waves.reset!(dynamics)
    gs = Flux.gradient(ps) do 

        latents = integrate(cell, wave, dynamics, steps)
        e = sum(sum.(energy.(displacement.(latents))))

        Flux.ignore() do 
            println(e)
        end

        return e
    end

    Flux.Optimise.update!(opt, ps, gs)
end

# wave = pulse(wave)
sol = solve(cell, wave, dynamics, steps) |> cpu
render!(sol, path = "vid_opt.mp4")