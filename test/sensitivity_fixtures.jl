# A valid user curve need not return continuously compounded zero rates.
struct PeriodicZeroSensitivityCurve{T} <: FM.Yield.AbstractYieldModel
    rate::T
end
Base.zero(c::PeriodicZeroSensitivityCurve, t) = FC.Periodic(c.rate, 1)
FC.discount(c::PeriodicZeroSensitivityCurve, t) = FC.discount(zero(c, t), t)

# Zero streams query the curve once, at time zero, for their result's numeric type; they
# never value a payment.
struct ZeroCashflowTestCurve <: FM.Yield.AbstractYieldModel end
FC.discount(::ZeroCashflowTestCurve, t) = iszero(t) ? one(float(t)) : error("zero cashflows do not value payments")

# Callable valuations: fixed cashflows on one curve or on `base + credit`, and the mean of a
# valuation across simulated paths.
struct CashflowValue{C, T}
    cashflows::C
    times::T
end
(v::CashflowValue)(curve) = FC.pv(curve, v.cashflows, v.times)
(v::CashflowValue)(base, credit) = v(base + credit)

struct ScenarioValue{V}
    value::V
end
(v::ScenarioValue)(scenarios) = sum(v.value, scenarios) / length(scenarios)

_same_sensitivity(a, b) = isapprox(a, b; rtol = 1.0e-12, atol = 1.0e-12)
_same_sensitivity(a::NamedTuple, b::NamedTuple) =
    keys(a) == keys(b) && all(map(_same_sensitivity, values(a), values(b)))

# An iterator that can be read only once, so reading it twice, or losing its first item, shows.
mutable struct OnePass{T}
    items::Vector{T}
    read::Bool
end
OnePass(items) = OnePass(collect(items), false)
function Base.iterate(p::OnePass, i = 1)
    if i == 1
        p.read && error("OnePass iterator read twice")
        p.read = true
    end
    return i > length(p.items) ? nothing : (p.items[i], i + 1)
end
Base.IteratorSize(::Type{<:OnePass}) = Base.SizeUnknown()
Base.eltype(::Type{OnePass{T}}) where {T} = T
