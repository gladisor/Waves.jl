using Waves, Flux, CairoMakie

function δ⁺(T::AbstractArray{Float32}, T_ign::Float32, X₁₂::AbstractArray{Float32}, X₁₂ₑ::AbstractArray{Float32})
    return (T .> T_ign) .| (X₁₂ .> X₁₂ₑ)
end

CH4_a = [5.14987613f0, -1.36709788f-2, 4.9100599f-5, -4.84743026f-8, 1.66693956f-11]
O2_a  = [3.78246636f0, -2.99673415f-3, 9.84730200f-6, -9.68129608f-9, 3.24372836f-12]
CO2_a = [2.35677352f0, 8.98459677f-3, -7.12356269f-6, 2.45919022f-9, -1.43699548f-13]
H2O_a = [4.19864056f0, -2.03643410f-3, 6.52040211f-6, -5.48797062f-9, 1.77197817f-12]
N2_a  = [3.53100528f0, -1.23660987f-4, -5.02999437f-7, 2.43530612f-9, -1.40881235f-12]

θ_i =   [1.0f0,     2.0f0,      1.0f0,      2.0f0,      0.0f0]
M_i =   [16.04f0,   32.00f0,    44.01f0,    18.02f0,    28.02f0]
initial_mass_fractions = [0.1f0, 0.6876f0, 0.2016f0, 0.0108f0]

ρ = 1.2172f0
cₕ = 0.803f0
Aᵣ = 4.154f-5
k = 0.255f0
δ_x = 2.89f-2
δ_z = 1.91f0
χ = 2.16f-2
T_ig = 431.6f0

function heat_capacity(coefs::AbstractVector{Float32}, T::AbstractMatrix{Float32})
    T̃ = cat([T .^ i for i in 0:(length(coefs)-1)]..., dims = ndims(T) + 1)
    return dropdims(sum(permutedims(coefs[:, :, :], (2, 3, 1)) .* T̃, dims = 3), dims =  3)
end

function heat_capacity(coefs::AbstractVector{Float32}, T::Float32)
    T̃ = cat([T .^ i for i in 0:(length(coefs)-1)]..., dims = ndims(T) + 1)
    return coefs' * T̃
end

Flux.device!(0)
Flux.CUDA.allowscalar(false)

dim = TwoDim(100.0f0, 512)
dx = get_dx(dim)
dy = get_dy(dim)

grid = build_grid(dim)
fire_centers = [
    0.0f0   0.0f0; 
    10.0f0 -10.0f0]

num_centers = size(fire_centers, 1)
T = build_normal(grid, fire_centers, fill(1.0, num_centers), fill(1.0, num_centers))
# cp = heat_capacity(CO2_a, T)
# heat_capacity(CO2_a, 500.0f0)

t = collect(range(100.0f0, 2000.0f0, 500))
cp = heat_capacity.([CO2_a], t)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, t, cp)
save("cp.png", fig)

# fig = Figure()
# ax = Axis(fig[1, 1], aspect = 1.0f0)
# heatmap!(ax, dim.x, dim.y, cp)
# save("wildfire.png", fig)