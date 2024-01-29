export build_pinn_grid, build_pinn_wave_encoder_head

function build_pinn_grid(latent_dim::OneDim, t::Vector{Float32})
    latent_gs = maximum(latent_dim.x)
    elements = length(latent_dim.x)
    dt = Flux.mean(diff(vec(t)))
    integration_steps = length(t)

    t_grid = repeat(reshape(t, 1, 1, integration_steps), 1, size(latent_dim.x, 1), 1) / (dt * integration_steps)
    x_grid = repeat(reshape(latent_dim.x, 1, elements, 1), 1, 1, integration_steps) / latent_gs
    pinn_grid = vcat(x_grid, t_grid)
    return reshape(pinn_grid, 2, :, 1)
end

function build_pinn_wave_encoder_head(h_size::Int, activation::Function, nfreq::Int, latent_dim::OneDim)
    return Chain(
        Parallel(
            vcat,
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq))
            ),
        b -> reshape(b, nfreq, 6, :),
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(
            x[:, [1], :],       # u_tot
            x[:, [2], :],       # v_tot
            x[:, [3], :],       # u_inc
            x[:, [4], :],       # v_inc
            x[:, [5], :],       # f
            x[:, [6], :] .^ 2
            )
        )
end