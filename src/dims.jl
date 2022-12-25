export OneDim, TwoDim, ThreeDim, dims

struct OneDim <: AbstractDim
    x::Num
end

struct TwoDim <: AbstractDim
    x::Num
    y::Num
end

struct ThreeDim <: AbstractDim
    x::Num
    y::Num
    z::Num
end

function TwoDim(x_min, x_max, y_min, y_max)
    @parameters x [bounds = (x_min, x_max)]
    @parameters y [bounds = (y_min, y_max)]
    return TwoDim(x, y)
end

function dims(dim::TwoDim)::Tuple
    return (dim.x, dim.y)
end