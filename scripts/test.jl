using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

