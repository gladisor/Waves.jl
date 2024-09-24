### keep confident, ST  ###

using Plots
using LinearAlgebra
using ForwardDiff

import Base.:∘; (∘)(x::Vector, y::Vector) = sum( x .* y)[1]

dist(a, b) = (d = a-b; (d ∘ d)^0.5)

Scalar   = Float64
ArrayType    = Array{Scalar, 1}
State        = ArrayType
BallPair = Tuple{Int, Int}
∞ = 1e+10

abstract type Model end

mutable struct PoolTorus <: Model
arena::ArrayType
M::ArrayType # mass 
R::ArrayType # radius 
bp::BitArray # ball or player
d::Int # state dimension
N::Int # number of balls
ballPairs::Vector{BallPair}
end

function visualize(pool::Model, x::Vector, msg)::Nothing
	
	y = deepcopy(ForwardDiff.value.(x))

	if isa(msg[1], Dual)
		t_ = msg[1].value
		bp = ForwardDiff.value.(pool.ballPairs[msg[3]])
	else
		t_ = msg[1]
		bp = pool.ballPairs[msg[3]]
	end

	plotsPath = "/Users/,,,,,,,/plots/collisions/"
	arena = pool.arena
	dim = pool.d
	N = pool.N 
	agents = pool.bp


	plotArgs = (	size=(600, 600), lims=1.0*[-arena[3], arena[3]], 
	axes=:square, aspect_ratio=:equal, legend=:none, 
	title="t=$(round(t_; digits=2))s, #T=$(msg[2]), bp=$(bp)", 
	minorgrid=true, lw=5, lc=:black)

	plot( [arena[3], arena[3]], [arena[2], arena[4]]; plotArgs...)
	plot!([arena[1], arena[3]], [arena[4], arena[4]]; plotArgs...)
	plot!([arena[1], arena[1]], [arena[2], arena[4]]; plotArgs...)
	plot!([arena[1], arena[3]], [arena[2], arena[2]]; plotArgs...)
# 	mu = 18#73/5
	mu = 10#73/5
	for i in 1:N
		i₁, i₂ = (dim*(i-1)+1), dim*i
		q₁, q₂, p₁, p₂ = y[i₁:i₂]..., y[dim*N .+ (i₁:i₂)]... 

		if agents[i]
			plot!( [q₁], [q₂]; lt=:scatter, mc=:red, ms=mu, plotArgs...)
			if p₁ ≥ 0 && p₂ ≥ 0 
				plot!( [q₁, q₁ + p₁], [q₂, q₂ + p₂]; lt=:line, plotArgs..., lc=:red, lw=1, arrow=true)
			elseif p₁ ≥ 0 && p₂ ≤ 0 
				plot!( [q₁, q₁ + p₁], [q₂, q₂ - p₂]; lt=:line, plotArgs..., lc=:red, lw=1, arrow=true)
			elseif p₁ ≤ 0 && p₂ ≥ 0 
				plot!( [q₁, q₁ - p₁], [q₂, q₂ + p₂]; lt=:line, plotArgs..., lc=:red, lw=1, arrow=true)
			else   p₁ ≤ 0 && p₂ ≤ 0 
				plot!( [q₁, q₁ - p₁], [q₂, q₂ - p₂]; lt=:line, plotArgs..., lc=:red, lw=1, arrow=true)
			end

			annotate!(q₁, q₂, text("$(i)", :black, :center, 12))
		else
			plot!( [q₁], [q₂]; lt=:scatter, mc=:green, ms=mu, plotArgs...)
			if p₁ ≥ 0 && p₂ ≥ 0 
				plot!( [q₁, q₁ + p₁], [q₂, q₂ + p₂]; lt=:line, plotArgs..., lc=:green, lw=1, arrow=true)
			elseif p₁ ≥ 0 && p₂ ≤ 0 
				plot!( [q₁, q₁ + p₁], [q₂, q₂ - p₂]; lt=:line, plotArgs..., lc=:green, lw=1, arrow=true)
			elseif p₁ ≤ 0 && p₂ ≥ 0 
				plot!( [q₁, q₁ - p₁], [q₂, q₂ + p₂]; lt=:line, plotArgs..., lc=:green, lw=1, arrow=true)
			else   p₁ ≤ 0 && p₂ ≤ 0 
				plot!( [q₁, q₁ - p₁], [q₂, q₂ - p₂]; lt=:line, plotArgs..., lc=:green, lw=1, arrow=true)
			end
			annotate!(q₁, q₂, text("$(i)", :black, :center, 12))
		end
	end
# 	"visualize to $(plotsPath)$(msg[2])_θ$(θ).png" |> display
	@show savefig(plotsPath * "T$(msg[2])" * ".png")
	nothing
end

function Pool(arena::ArrayType, m::ArrayType, r::ArrayType, bp::BitArray, dim::Int) 

	N = length(m)

	### number of wall ###
	numW = 4
	ballPairs = [(i, j) for i in 1:N for j in i+1:(N+numW)] # + 4 for the walls

	PoolTorus(arena, m, r, bp, dim, N, ballPairs)

end

function GeneratePool(dyn::Model, eltype::Type)
	N = dyn.N
	dim = dyn.d
	arena = dyn.arena
	r = dyn.R[1]
	x = zeros(N, dim) # + 0.1*randn(dim)

	### state structure  ###
	### X = [x₁; y₁; x₂; y₂, vx₁; vy₁; vx₂; vy₂ ] ###

	### random positions for N balls  ###
	while Int(length(x)/dim) < N
		qⱼ = zeros(eltype, dim) + (arena[4]-2r)*randn(dim)*0.5
		add = true
		for i in 1:Int( length(x)/dim )
			qᵢ = view(x, dim*(i-1)+1:dim*i)
			if dist( qᵢ, qⱼ ) < 2r 
				add = false
				break
			end
			if !all(abs.(qⱼ) .<= (arena[4] - r))
				add = false
				break
			end
		end
		if add
			x = vcat(x, qⱼ)
		end
	end
	### add random or zero velocity to each ball ###
	return vcat(x, zeros(eltype, dim*N) + 0.5randn(dim*N))
end

function BallxWall(pool, q₁, p₁, b₂)

	arena = pool.arena
	N = pool.N

	slope = p₁[2] / p₁[1]
	c = -q₁[1]*slope + q₁[2]

	q₂ = zeros(eltype(q₁), pool.d)

	### N+1, N+2, N+3, N+4 = VL, HB, VR, HT ###
	if b₂ == N+1
		q₂ += [arena[3], slope*arena[3] + c]
	elseif b₂ == N+2
		q₂ += [(arena[2]-c)/slope, arena[2]]
	elseif b₂ == N+3
		q₂ += [arena[1], slope*arena[1] + c]
	elseif b₂ == N+4
		q₂ += [(arena[4]-c)/slope, arena[4]]
	end
	q₂
end

function NextCollisionTime(pool::Model, X)

	pairs = pool.ballPairs
	n = pool.N
	d = pool.d

	t = zeros(eltype(X), length(pairs))

	for (i, (b₁, b₂)) ∈ enumerate(pairs)
		q₁ = X[         ((d*b₁-1):(d*b₁))]
		p₁ = X[n*d .+   ((d*b₁-1):(d*b₁))]
		### b₂ is a ball ###
		if b₂ ≤ n
			q₂ = X[       ((d*b₂-1):(d*b₂))]
			p₂ = X[n*d .+ ((d*b₂-1):(d*b₂))]
		else
			q₂ = BallxWall(pool, q₁, p₁, b₂) 
			p₂ = zeros(eltype(X), size(p₁)...)
		end
		t[i] = τ(q₁, q₂, p₁, p₂)
	end

	val, idx = findmin(t)

end

function τ(qᵢ, qⱼ, pᵢ, pⱼ)

	t = ∞

	x₁, y₁ = qᵢ
	ẋ₁, ẏ₁ = pᵢ
	x₂, y₂ = qⱼ
	ẋ₂, ẏ₂ = pⱼ

	Δx = x₂ - x₁
	Δy = y₂ - y₁
	Δẋ = ẋ₂ - ẋ₁
	Δẏ = ẏ₂ - ẏ₁

	A = Δẋ*Δẋ  + Δẏ*Δẏ
	B =  Δx*Δẋ  +  Δy*Δẏ
	C =   Δx*Δx  +   Δy*Δy
	D = 2r

	F = (B^2 - A*(C-D^2))

	if F ≥ 0.0 && A ≠ 0.0

		G = √F

		t₁ =  (-B - G) / A
		t₂ =  (-B + G) / A

		t = max(min(t₁, t₂), 0.0)
		t =  B < 0 ? t : ∞ 
	end 
	t
end

function MomentTransfer(pool::Model, X, Idᵪ)

	n ,  d = pool.N, pool.d
	b₁, b₂ = pool.ballPairs[Idᵪ]
	r₁     = pool.R[b₁]

	q₁ = X[       ((d*b₁-1):(d*b₁))]
	p₁ = X[n*d .+ ((d*b₁-1):(d*b₁))]

	if b₂ ≤ n
		q₂ = X[        (d*b₂-1):(d*b₂)]
		p₂ = X[n*d .+ ((d*b₂-1):(d*b₂))]
	else
		# 		q₂ = BallxWall(pool, q₁, p₁, b₂) 
		if b₂ == n+1
			q₂ = q₁ + [+r₁, 0]
		elseif b₂ == n+2
			q₂ = q₁ + [0, -r₁]
		elseif b₂ == n+3
			q₂ = q₁ + [-r₁, 0]
		elseif b₂ == n+4
			q₂ = q₁ + [0, +r₁]
		end
		p₂ = zeros(eltype(X), size(p₁)...)
	end

	Δq = q₁ - q₂
	Δp = p₁ - p₂
	
	Δ = ( (Δp ∘ Δq) / (Δq ∘ Δq) ) * Δq

	Δ
end

function ControlGain(pool::Model, ballIdx)
	n     = pool.N # all balls
	d     = pool.d

	ñ     = length(ballIdx) # acting balls / agents
	G     = zeros(2*d*n, d*ñ)

	for (i, bᵢ) in enumerate(ballIdx |> sort!)
		G[(n*d + (d*bᵢ-1)):(n*d + d*bᵢ), (d*i-1):(d*i)] = LinearAlgebra.I(d)
	end

	G
end

### pool, time of simulation in seconds, state, control gain  ###
function Simulate(pool, tᶠ, X, Gᵤ)

	d = pool.d
	Trj = X
	Tᵪ  = [0.0]

	t  = 0.0 # current time in sec
	Tᵢ = 0 # number of collusions
	K = 100 # length of action sequence	

	U = zeros( d*length(actingBalls), K ) 
	### U[:, 1] = [1; 1; 0; 0] ###

	k = 0 
	while t ≤ tᶠ	

		X, tᵪ = dynamics(pool::Model, Gᵤ, X, U, Tᵢ)
		
		Trj = hcat(Trj, X)
		Tᵪ  = hcat(Tᵪ, tᵪ)
		t += tᵪ
		k += 1
		if k > K
			break
		end
	end
	Trj, Tᵪ
end

# linear dynamics between collisions
A(n) = (A_ = zeros(2n, 2n); A_[1:n, (n+1):(2n)] .= I(n); A_) 

function dynamics(pool::Model, Gᵤ, X, U, Tᵢ)

	n, d = pool.N, pool.d
	pairs = pool.ballPairs

	### apply force ###
	### U[:, k] = [fx₁; fy₁; fx₂; fy₂] ###
	X = X + Gᵤ * U[:, Tᵢ+1] 

	tᵪ, Idᵪ	= NextCollisionTime(pool, X)

	# state propagation without friction and/or any force 
	X = (I + A(2n)*tᵪ)*X
	# display(X)

	Δ = MomentTransfer(pool, X, Idᵪ)

	b₁, b₂ = pairs[Idᵪ]

	bpᵪ = [b₁]
	bpᵪ = b₂ ≤ n ? [bpᵪ..., b₂] : bpᵪ

	m₁ = M[b₁]
	m₂ = b₂ ≤ n ? M[b₂] : ∞

	Ũ = -2m₂*Δ/(m₁ + m₂)	
	if b₂ ≤ n #ball to ball
		Ũ = [Ũ..., +2m₁*Δ/(m₁ + m₂)...]
	end

	#Gₘ - changing moment transfer gain matrix
	Gₘ = ControlGain(pool, bpᵪ)

	X = X + Gₘ * Ũ

	X, tᵪ
end

### GENERATE POOL ###
d  = 2
r  = 0.25
m  = 1.0
M = m*ones(4)
R = r*ones(4)
n = length(R)
arenaSide = 10.0 # meters
arena  = [-arenaSide, -arenaSide, +arenaSide, +arenaSide]*1.25

# all cylinders are active  
agents = BitVector(rand(n) .> 0)

pool = Pool(arena, M, R, agents, d) 
@show pool

actingBalls = findall(agents)
actingBallIdx = []
for i in actingBalls
	push!(actingBallIdx, ((d*i-1):(d*i))...)
end
for i in actingBalls
	push!(actingBallIdx, ((d*n + (d*i-1)):((d*n + d*i)))...)
end

tᶠ = 20.0 # dt = 1e-3 time in sec of simulation
X  =  GeneratePool(pool, Float64)
# visualize(pool, X, (0, 0, 1)) ### ### ### ### ### ### ###
Gᵤ = ControlGain(pool::Model, actingBalls)

Trj, CollisionTimes = Simulate(pool, tᶠ, X, Gᵤ)
