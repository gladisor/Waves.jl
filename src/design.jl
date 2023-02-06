export DesignInterpolator

struct DesignInterpolator
    initial::Scatterer
    Δ::Scatterer
    ti::Float64
    tf::Float64
end

function (interp::DesignInterpolator)(t::Float64)
    t = (t - interp.ti) / (interp.tf - interp.ti)
    return interp.initial + t * interp.Δ
end