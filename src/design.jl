export DesignInterpolator

struct DesignInterpolator
    initial::AbstractDesign
    Δ::AbstractDesign
    ti::Float64
    tf::Float64
end

function (interp::DesignInterpolator)(t::Float64)
    d = (interp.tf - interp.ti)
    t = ifelse(d > 0.0, (t - interp.ti) / d, 0.0)
    return interp.initial + t * interp.Δ
end