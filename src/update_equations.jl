export split_wave_pml, runge_kutta

function split_wave_pml(wave::AbstractMatrix{Float32}, t::Float32, dynamics::WaveDynamics)
    # U = selectdim(wave, 2, 1)
    # V = selectdim(wave, 2, 2)
    U = wave[:, 1]
    V = wave[:, 2]

    ∇ = dynamics.grad
    σx = dynamics.pml
    C = Waves.speed(dynamics, t)
    b = C .^ 2

    dU = b .* (∇ * V) .- σx .* U
    dVx = ∇ * U .- σx .* V

    return cat(dU, dVx, dims = 2)
end

"""
Update rule for a two dimensional wave with a pml. Assumes the dimention is a square with
the same number of discretization points in each dimension.
"""
function split_wave_pml(wave::AbstractArray{Float32, 3}, t::Float32, dyn::WaveDynamics)
    # U = selectdim(wave, 3, 1)
    # Vx = selectdim(wave, 3, 2)
    # Vy = selectdim(wave, 3, 3)
    # Ψx = selectdim(wave, 3, 4)
    # Ψy = selectdim(wave, 3, 5)
    # Ω = selectdim(wave, 3, 6)

    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

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

    return cat(dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

"""
Runge Kutta integration scheme for more accuratly estimating the rate of change of the
wave over time.
"""
function runge_kutta(f::Function, wave::AbstractArray{Float32}, dyn::WaveDynamics)
    h = dyn.dt
    t = dyn.t * h

    k1 = f(wave,                   t,            dyn) ## Euler
    k2 = f(wave .+ 0.5f0 * h * k1, t + 0.5f0 * h, dyn) ## Midpoint
    k3 = f(wave .+ 0.5f0 * h * k2, t + 0.5f0 * h, dyn)
    k4 = f(wave .+         h * k3, t +         h, dyn) ## Endpoint
    return 1/6f0 * h * (k1 .+ 2*k2 .+ 2*k3 .+ k4)
end