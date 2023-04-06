using Flux
using Flux.Losses: huber_loss
using CairoMakie
using Waves

include("models.jl")

function plot_energy!(t::Vector{Float32}, sigma::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Energy")
    lines!(ax, t, sigma, linewidth = 3)
    save(path, fig)
end

function plot_energy!(sol::WaveSol; kwargs...)
    sigma = sum.(energy.(displacement.(sol.u)))
    t = sol.t
    plot_energy!(t, sigma; kwargs...)
end

s, a = load_episode_data("data/episode1/data10.jld2");
sol_sc = s.sol.total - s.sol.incident

elements = size(sol_sc.u[1], 1)
z_elements = 4096
h_size = 512

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

model = Chain(
    WaveEncoder(6, 32, 2, tanh),
    Parallel(
        hcat,
        Dense(4096, h_size, tanh),
        Chain(Dense(4096, h_size, sigmoid), z -> prod(z, dims = 2))),
    FEMIntegrator(h_size, 20; grid_size = 5.0f0, dynamics_kwargs...),
    Flux.flatten,
    Dense(h_size * 3, 1),
    vec
    )

opt = Adam(0.0001)
ps = Flux.params(model)

sigma = sol_sc.u .|> displacement .|> energy .|> sum
sigma_true = sigma[2:end]

for i in 1:30

    gs = Flux.gradient(ps) do 
        sigma_pred = model(s.sol.total)
        loss = huber_loss(sigma_true, sigma_pred)

        Flux.ignore() do
            println(loss)
        end

        return loss
    end

    Flux.update!(opt, ps, gs)
end

sigma_pred = model(s.sol.total)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, s.sol.total.t[2:end], sigma_pred, label = "Predicted")
lines!(ax, s.sol.total.t[2:end], sigma_true, label = "Actual")
axislegend(ax)
save("energy.png", fig)

iter = model[3]
dim = iter.dynamics.dim

z = s.sol.total |> model[1] |> model[2]
z_sol = solve(iter.cell, z, iter.dynamics, iter.steps)

render!(z_sol, path = "vid.mp4")