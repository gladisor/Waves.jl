export AbstractDim, OneDim, TwoDim, ThreeDim, dims

struct OneDim <: AbstractDim
    x::Num
end

function OneDim(x_min, x_max)
    @parameters x [bounds = (x_min, x_max)]
    return OneDim(x)
end

struct TwoDim <: AbstractDim
    x::Num
    y::Num
end

function TwoDim(x_min, x_max, y_min, y_max)
    @parameters x [bounds = (x_min, x_max)]
    @parameters y [bounds = (y_min, y_max)]
    return TwoDim(x, y)
end

struct ThreeDim <: AbstractDim
    x::Num
    y::Num
    z::Num
end

function ThreeDim(x_min, x_max, y_min, y_max, z_min, z_max)
    @parameters x [bounds = (x_min, x_max)]
    @parameters y [bounds = (y_min, y_max)]
    @parameters z [bounds = (z_min, z_max)]
    return ThreeDim(x, y, z)
end


dims(dim::OneDim)::Tuple = (dim.x,)
dims(dim::TwoDim)::Tuple = (dim.x, dim.y)
dims(dim::ThreeDim)::Tuple = (dim.x, dim.y, dim.z)