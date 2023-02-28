export split_wave_pml

function split_wave_pml(wave::Wave{TwoDim}, t::Float32, dyn::WaveDynamics)

    U = displacement(wave)
    Vx = field(wave, 2)
    Vy = field(wave, 3)
    Ψx = field(wave, 4)
    Ψy = field(wave, 5)
    Ω = field(wave, 6)

    C = Waves.speed(dyn, t)
    b = C .^ 2
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'

    Vxx = ∇ * Vx
    Vyy = (∇ * Vy')'
    Ux = ∇ * U
    Uy = (∇ * U')'

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return Wave{TwoDim}(cat(dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3))
end